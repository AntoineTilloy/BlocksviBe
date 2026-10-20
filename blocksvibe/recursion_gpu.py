"""Batched GPU radial recursion of blocks_3d (prototype).

blocks_3d computes, for one four-point structure and one lambda-derivative order n, the residues
res[p][m][j][P][a][b] level by level in m (fill_residues.cxx), one (j, pole) task at a time:

    H(src, m, j', D')  = h_inf[src][m][j'] + sum_i 2/(D' - D_i) res[src][m - shift_i][j'][i]
    res[p][m][j][P]    = Q(P, j) * L_P H(src, m, j', D'_P) R_P^T          (j' = P.two_j(j))

The operator (pole positions, Cauchy coefficients 2/(D'-D_i), L/R matrices, Q) depends only on
(j12, j43, Delta_12, Delta_43, pseudo-parities, order, j_max), not on the four-point structure q4
nor on n: all (q4, n) pairs of a blocks_3d family are independent right-hand sides.  Here they form
the batch dimension B, and each level m is

  1. Cauchy step: for every (src, j'): H = C X, C (rows D' x prime poles i), X (poles x S^2 B):
     one batched multiprecision GEMM (RNS/INT8 tensor cores, sdpvibe.mpf.gemm) over (src, j'),
     padded to common sizes;
  2. pole step: res = Q L H R^T for every (p, j, P): elementwise multiprecision products and sums
     (S <= 2 j12 + 1 structures), gathered from tables of L/R matrices and Q values.

Residues live in one slot buffer: slot = p * nslots + offset[m][jidx] + pole index, plus a zero slot.
All index tensors and constants are built once per operator (Plan, Recursion.__init__); run() only
does GPU work and can be called for any number of right-hand-side batches.
"""
from fractions import Fraction
import time
import mpmath as mp
import torch

from . import structure as S
from .cauchy import cauchy as cauchy_kernel
from .smallgemm import smallgemm
from .. import mpf
from ..rns import dev


def bgemm(A, B, chunk=384):
    """batched mpf.gemm, chunked over the batch (the RNS kernel's grid z = primes x batch < 2^16)"""
    nb = A.shape[0]
    if nb <= chunk:
        return mpf.gemm(A, B)
    parts = [mpf.gemm(A[i:i + chunk], B[i:i + chunk]) for i in range(0, nb, chunk)]
    return mpf.MPF(torch.cat([q.m for q in parts], 1), torch.cat([q.e for q in parts], 0))


def random_mpf(shape, L, spread=8):
    """random normalised MPF with |x| in ~[2^-spread, 2^spread] (timing runs)"""
    n = 1
    for d in shape: n *= d
    m = torch.randint(-2**31, 2**31 - 1, (L, n), dtype=torch.int32, device=dev)
    e = torch.randint(-spread, spread + 1, (n,), dtype=torch.int64, device=dev) - 32 * L
    return mpf.normalize(m, e, L).reshape(*shape)


def _frac_mp(x):
    return mp.mpf(x.numerator) / x.denominator


def _tick(prof, key):
    """accumulate GPU wall time since the previous tick into prof[key]"""
    torch.cuda.synchronize()
    t = time.time()
    if key is not None:
        prof[key] = prof.get(key, 0.0) + t - prof["_t"]
    prof["_t"] = t


def _take(A, idx):
    """A[idx] along the first entry dimension (idx: long tensor of any shape)"""
    return mpf.MPF(A.m[:, idx], A.e[idx])


def _permute(A, perm):
    return mpf.MPF(A.m.permute(0, *[p + 1 for p in perm]).contiguous(), A.e.permute(*perm).contiguous())


def _expand(A, shape):
    return mpf.MPF(A.m.expand(A.L, *shape).contiguous(), A.e.expand(*shape).contiguous())


class Plan:
    """CPU-side index plan of the recursion for one operator (depends on no number)."""

    def __init__(self, two_j12, two_j43, two_j_max, order, pseudo_parities):
        self.tj12, self.tj43, self.tjmax, self.order, self.pp = two_j12, two_j43, two_j_max, order, pseudo_parities
        self.SL = lambda p, tj: len(S.so3(pseudo_parities[p][0], two_j12, tj))
        self.SR = lambda p, tj: len(S.so3(pseudo_parities[p][1], two_j43, tj))
        self.Smax = max(max(self.SL(p, tj), self.SR(p, tj)) for p in range(2)
                        for tj in range(two_j12 % 2, two_j_max + 2 * order + 1, 2))
        self.Jh = len(S.two_j_values(0, order + 1, two_j12, two_j_max))   # spins of h_inf (as blocks_3d)
        self.offset, n = {}, 0
        for m in range(order):
            for jidx, tj in enumerate(S.two_j_values(m, order, two_j12, two_j_max)):
                self.offset[(m, jidx)] = n
                n += len(S.poles(tj, two_j12, two_j43, order - m - S.j_excess(tj, two_j_max)))
        self.nslots = n
        self.zero_slot = 2 * n
        self.Mkeys, self.Qkeys = {}, {}           # (P, delta, two_jNN, ppp, two_j) / (P, tj, sign) -> index
        self.dmax = 1
        self.levels = [self._level(m) for m in range(order)]

    def _mkey(self, key):
        return self.Mkeys.setdefault(key, len(self.Mkeys))

    def _level(self, m):
        tj12, tj43, tjmax, order = self.tj12, self.tj43, self.tjmax, self.order
        groups = {}
        tasks = []
        for jidx, tj in enumerate(S.two_j_values(m, order, tj12, tjmax)):
            pl = S.poles(tj, tj12, tj43, order - m - S.j_excess(tj, tjmax))
            for pidx, P in enumerate(pl):
                tjp = S.pole_two_j(P, tj)
                dprime = S.delta_2(P, tj) + 2 * S.shift(P)
                for p in range(2):
                    src = (p + S.parity(P)) % 2
                    if not (self.SL(p, tj) and self.SR(p, tj) and self.SL(src, tjp) and self.SR(src, tjp)):
                        continue
                    g = groups.get((src, tjp))
                    if g is None:
                        jpex = S.j_excess(tjp, tjmax)
                        cols = [(src * self.nslots + self.offset[(m - S.shift(i), tjp // 2)]
                                 + S.position(tjp, tj12, tj43, order - m + S.shift(i) - jpex, i),
                                 S.delta_2(i, tjp)) for i in S.poles(tjp, tj12, tj43, m)]
                        g = groups[(src, tjp)] = {"rows": {}, "cols": cols, "gi": len(groups)}
                    r = g["rows"].setdefault(dprime, len(g["rows"]))
                    sign = -1 if (tjp - tj) % 4 else 1
                    tasks.append((p * self.nslots + self.offset[(m, jidx)] + pidx, g["gi"], r,
                                  self._mkey((P, 0, tj12, self.pp[src][0], tj)),
                                  self._mkey((P, 1, tj43, self.pp[src][1], tj)),
                                  self.Qkeys.setdefault((P, tj, sign), len(self.Qkeys))))
        if not groups:
            return None
        gl = sorted(groups.items(), key=lambda kv: kv[1]["gi"])
        nb = len(gl)
        Mr = max(len(g["rows"]) for _, g in gl)
        Kc = max(1, max(len(g["cols"]) for _, g in gl))
        col = torch.full((nb, Kc), self.zero_slot, dtype=torch.long)
        cidx = torch.zeros((nb, Mr, Kc), dtype=torch.long)       # coefficient table index (0 -> zero)
        cdiff = torch.zeros((nb, Mr, Kc), dtype=torch.int32)     # D' - D_i (coefficient 2/diff; 0 -> zero)
        for gi, ((src, tjp), g) in enumerate(gl):
            for c, (slot, di) in enumerate(g["cols"]):
                col[gi, c] = slot
            for d, r in g["rows"].items():
                for c, (_, di) in enumerate(g["cols"]):
                    diff = d - di
                    self.dmax = max(self.dmax, abs(diff))
                    cidx[gi, r, c] = 0 if diff == 0 else (2 * abs(diff) - (1 if diff > 0 else 0))
                    cdiff[gi, r, c] = diff
        t = torch.tensor(tasks, dtype=torch.long)
        return {"m": m, "nb": nb, "Mr": Mr, "Kc": Kc,
                "gsrc": torch.tensor([k[0] for k, _ in gl]), "gj": torch.tensor([k[1] // 2 for k, _ in gl]),
                "col": col, "cidx": cidx, "cdiff": cdiff, "t_slot": t[:, 0], "t_hrow": t[:, 1] * Mr + t[:, 2],
                "t_L": t[:, 3], "t_R": t[:, 4], "t_q": t[:, 5]}

    def counts(self):
        """output entries of the Cauchy and pole steps per right-hand side (rows padded per level,
        structures padded to Smax^2) -- what run() computes"""
        h = sum(lev["nb"] * lev["Mr"] for lev in self.levels if lev) * self.Smax ** 2
        r = sum(len(lev["t_slot"]) for lev in self.levels if lev) * self.Smax ** 2
        return h, r


class Recursion:
    """Numbers (M matrices, Q values, Cauchy coefficients) on the GPU for one Plan.
    M: dict as in dump.read_M (strings) -- matrices for every key of plan.Mkeys; or None for
    random O(1) matrices (timing runs: the cost does not depend on the values)."""

    def __init__(self, plan, M, L, pole="cuda", cauchy="auto", task_chunk=8192, kcut=128):
        """pole: "cuda" (two batched S x S GEMMs on CUDA cores, blocks/smallgemm.py) | "gemm" (batched
        q L(x)R RNS GEMM) | "elementwise"; cauchy: "cuda" (CUDA-core kernel using that
        the coefficients are 2/integer, blocks/cauchy.py; cost ~ K) | "rns" (batched RNS GEMM; cost ~
        CRT per entry) | "auto" (cuda for levels with <= kcut prime poles, rns above)"""
        self.plan, self.L, self.pole, self.cauchy, self.task_chunk = plan, L, pole, cauchy, task_chunk
        self.kcut = kcut
        Sm = plan.Smax
        mp.mp.prec = 32 * L + 64
        nM = len(plan.Mkeys)
        if M is None:
            self.Mtab = random_mpf((nM, Sm, Sm), L)
        else:
            self._load_M(M, nM, Sm)
        self._constants()

    def _load_M(self, M, nM, Sm):
        L, plan = self.L, self.plan
        strs = ["0"] * (nM * Sm * Sm)
        for key, i in plan.Mkeys.items():
            mat = M.get(key) or []
            for a, row in enumerate(mat):
                for b, x in enumerate(row):
                    strs[(i * Sm + a) * Sm + b] = x
        self.Mtab = mpf.from_strings(strs, L, (nM, Sm, Sm))

    def _constants(self):
        L, plan = self.L, self.plan
        qv = [0] * len(plan.Qkeys)
        for (P, tj, sign), i in plan.Qkeys.items():
            qv[i] = _frac_mp(S.q_value(P, tj) * sign)
        self.Qtab = mpf.from_mp(qv, L, (len(qv),))
        tab = [mp.mpf(0)]
        for k in range(1, plan.dmax + 1):
            tab += [mp.mpf(2) / k, mp.mpf(-2) / k]       # index 2k-1: +2/k, 2k: -2/k
        self.Ctab = mpf.from_mp(tab, L, (len(tab),))
        self.levels = [None if lev is None else {k: (v.to(dev) if torch.is_tensor(v) else v)
                                                 for k, v in lev.items()} for lev in plan.levels]

    def run(self, h_inf, timer=None, prof=None):
        """h_inf: MPF (2, order, Jh, Smax^2, B).  Returns residues MPF (2 nslots + 1, Smax^2, B)."""
        pl, L, Sm = self.plan, self.L, self.plan.Smax
        S2, B = Sm * Sm, h_inf.shape[-1]
        res = mpf.zeros((2 * pl.nslots + 1, S2, B), L)
        if prof is not None:
            _tick(prof, None)
        for lev in self.levels:
            if lev is None:
                continue
            m, nb, Mr, Kc = lev["m"], lev["nb"], lev["Mr"], lev["Kc"]
            # 1. Cauchy step
            Hi = mpf.MPF(h_inf.m[:, lev["gsrc"], m, lev["gj"]], h_inf.e[lev["gsrc"], m, lev["gj"]])   # (nb, S2, B)
            Hi = _expand(Hi.reshape(nb, 1, S2 * B), (nb, Mr, S2 * B))
            if m > 0:
                X = _take(res, lev["col"]).reshape(nb, Kc, S2 * B)
                if self.cauchy == "cuda" or (self.cauchy == "auto" and Kc <= self.kcut):
                    H = mpf.add(cauchy_kernel(X, lev["cdiff"]), Hi)
                else:
                    C = _take(self.Ctab, lev["cidx"])                                 # (nb, Mr, Kc)
                    H = mpf.add(bgemm(C, X), Hi)
            else:
                H = Hi
            if prof is not None:
                _tick(prof, "cauchy")
            # 2. pole step: out[t,a0,b0] = q_t sum_a L[t,a0,a] sum_b R[t,b0,b] H[t,a,b]
            T = lev["t_slot"].numel()
            Ht = _take(H.reshape(nb * Mr, S2, B), lev["t_hrow"])                     # (T, a b, B)
            for t0 in range(0, T, self.task_chunk):
                sl = slice(t0, min(T, t0 + self.task_chunk))
                out = self._pole(Ht[sl], lev["t_L"][sl], lev["t_R"][sl], lev["t_q"][sl])
                res.m[:, lev["t_slot"][sl]] = out.m
                res.e[lev["t_slot"][sl]] = out.e
            if prof is not None:
                _tick(prof, "pole")
            if timer:
                timer(m)
        return res

    def _pole(self, Ht, iL, iR, iq):
        """out[t, a0 b0, :] = q_t sum_{a,b} L_t[a0,a] R_t[b0,b] Ht[t, a b, :]"""
        Sm, B = self.plan.Smax, Ht.shape[-1]
        T, S2 = Ht.shape[0], Sm * Sm
        Lt, Rt = _take(self.Mtab, iL), _take(self.Mtab, iR)                       # (T, a0, a), (T, b0, b)
        if self.pole == "cuda":           # two batched S x S GEMMs on CUDA cores: (q L) (R H^T)^T
            H4 = Ht.reshape(T, Sm, Sm, B)                                          # (t, a, b, k)
            Hb = _permute(H4, (0, 2, 1, 3)).reshape(T, Sm, Sm * B)                 # (t, b, a k)
            Y = smallgemm(Rt, Hb).reshape(T, Sm, Sm, B)                            # (t, b0, a, k)
            Y = _permute(Y, (0, 2, 1, 3)).reshape(T, Sm, Sm * B)                   # (t, a, b0 k)
            qL = mpf.mul(Lt, _expand(_take(self.Qtab, iq)[:, None, None], (T, Sm, Sm)))
            return smallgemm(qL, Y).reshape(T, S2, B)
        if self.pole == "gemm":           # batched (S^2 x S^2) GEMM with K = q (L (x) R), built on the GPU
            shK = (T, Sm, Sm, Sm, Sm)                                              # (a0, b0, a, b)
            K = mpf.mul(_expand(Lt[:, :, None, :, None], shK), _expand(Rt[:, None, :, None, :], shK))
            K = mpf.mul(K, _expand(_take(self.Qtab, iq)[:, None, None, None, None], shK))
            return bgemm(K.reshape(T, S2, S2), Ht)
        sh = (T, Sm, Sm, B)                                                        # elementwise variant
        Ht = Ht.reshape(T, Sm, Sm, B)
        inter = None                                                               # (T, a, b0, B)
        for b in range(Sm):
            term = mpf.mul(_expand(Ht[:, :, b:b + 1, :], sh), _expand(Rt[:, None, :, b, None], sh))
            inter = term if inter is None else mpf.add(inter, term)
        out = None                                                                 # (T, a0, b0, B)
        for a in range(Sm):
            term = mpf.mul(_expand(inter[:, a:a + 1, :, :], sh), _expand(Lt[:, :, a, None, None], sh))
            out = term if out is None else mpf.add(out, term)
        out = mpf.mul(out, _expand(_take(self.Qtab, iq)[:, None, None, None], sh))
        return out.reshape(T, S2, B)

    def slot(self, p, m, jidx, pidx):
        return p * self.plan.nslots + self.plan.offset[(m, jidx)] + pidx
