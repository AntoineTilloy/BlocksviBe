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

Residues live in one slot buffer: logical slot = p * nslots + offset[m][jidx] + pole index, plus a zero
slot.  With Plan(reuse=True) a logical slot is mapped to a physical one that is recycled once the residue
has been read as a Cauchy column (its only later use; derivative moments are taken right after the
level, blocksvibe/derivatives.py): the buffer holds the peak of live residues instead of all of them.
All index tensors and constants are built once per operator (Plan, Recursion.__init__); run() only
does GPU work and can be called for any number of right-hand-side batches.
"""
from fractions import Fraction
import time
import mpmath as mp
import numpy as np
import torch

from . import structure as S
from .cauchy import cauchy as cauchy_kernel
from .smallgemm import smallgemm
from sdpvibe import mpf
from sdpvibe.rns import dev


def _pad(A, dims):
    """zero-pad the last two entry dimensions of A to (dims[0], dims[1])"""
    r, c = A.shape[-2], A.shape[-1]
    if (r, c) == tuple(dims):
        return A
    m = torch.nn.functional.pad(A.m, (0, dims[1] - c, 0, dims[0] - r))
    e = torch.nn.functional.pad(A.e, (0, dims[1] - c, 0, dims[0] - r), value=mpf.EZ)
    return mpf.MPF(m, e)


_NOPAD = __import__("os").environ.get("SDPVIBE_BLOCKS_NOPAD") == "1"      # A/B timing of the padding only


def gemm(A, B, q=32):
    """mpf.gemm (2-d or batched); for K >= 64, K (and M, N when >= 64) zero-padded to multiples of q: the
    RNS kernel cannot vectorise its int8 loads when K is not a multiple of 16 and pays partial tiles in M, N
    (3x at K = 154).  Small products are left alone: there the padding costs more conversions than it saves."""
    M, K = A.shape[-2:]
    N = B.shape[-1]
    if _NOPAD:
        return mpf.gemm(A, B)
    if K < 64:
        return mpf.gemm(A, B)
    up = lambda x: -(-x // q) * q if x >= 64 else x
    Mp, Kp, Np = up(M), up(K), up(N)
    if (Mp, Kp, Np) == (M, K, N):
        return mpf.gemm(A, B)
    C = mpf.gemm(_pad(A, (Mp, Kp)), _pad(B, (Kp, Np)))
    if (Mp, Np) == (M, N):
        return C
    return mpf.MPF(C.m[..., :M, :N].contiguous(), C.e[..., :M, :N].contiguous())


def bgemm(A, B, chunk=384):
    """batched (aligned) mpf.gemm, chunked over the batch (the RNS kernel's grid z = primes x batch < 2^16)"""
    nb = A.shape[0]
    if nb <= chunk:
        return gemm(A, B)
    parts = [gemm(A[i:i + chunk], B[i:i + chunk]) for i in range(0, nb, chunk)]
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


class _Offsets:
    """offset[(m, jidx)]: first logical slot of spin jidx at level m"""

    def __init__(self, start, flat):
        self.start, self.flat = start, flat

    def __getitem__(self, key):
        return int(self.flat[self.start[key[0]] + key[1]])


def _first_rank(codes):
    """index of each code's value in order of first appearance (dict-insertion order), and the values"""
    uniq, first, inv = np.unique(codes, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(uniq), dtype=np.int64)
    rank[order] = np.arange(len(uniq))
    return rank[inv.reshape(-1)], uniq[order]


class Plan:
    """CPU-side index plan of the recursion for one operator (depends on no number).  Vectorised (numpy):
    builds exactly the tensors of the original per-task builder (plan_ref.PlanRef, tests/test_blocks_plan.py),
    except that the per-level Cauchy tables are kept as row/column pole positions (rowd/nrow, cold/ncol)
    and expanded on the GPU when used (cauchy_tables): dense, they would take tens of GB at order 152."""

    def __init__(self, two_j12, two_j43, two_j_max, order, pseudo_parities, reuse=False):
        self.tj12, self.tj43, self.tjmax, self.order, self.pp = two_j12, two_j43, two_j_max, order, pseudo_parities
        tj12, tj43, tjmax = two_j12, two_j43, two_j_max
        self.SL = lambda p, tj: len(S.so3(pseudo_parities[p][0], two_j12, tj))
        self.SR = lambda p, tj: len(S.so3(pseudo_parities[p][1], two_j43, tj))
        b = tj12 % 2
        tj_all = range(b, tjmax + 2 * order + 3, 2)                     # every two_j a task can touch
        self._SL = np.array([[self.SL(p, tj) for tj in tj_all] for p in range(2)])
        self._SR = np.array([[self.SR(p, tj) for tj in tj_all] for p in range(2)])
        nmax = (tjmax + 2 * order - b) // 2 + 1
        self.Smax = int(max(self._SL[:, :nmax].max(), self._SR[:, :nmax].max()))
        self.Jh = len(S.two_j_values(0, order + 1, tj12, tjmax))          # spins of h_inf (as blocks_3d)
        # logical slot layout: (m, jidx, pole) consecutive, parity p adds p * nslots
        self._tjs, self._ords, self._npol, start, flat, n, nent = [], [], [], [], [], 0, 0
        for m in range(order):
            tjs = np.arange(b, tjmax + 2 * (order - m - 1) + 1, 2, dtype=np.int64)
            ords = order - m - S.j_excess_v(tjs, tjmax)
            npol = S.counts_v(tjs, tj12, tj43, ords).sum(0)
            start.append(nent)
            nent += len(tjs)
            flat.append(n + np.concatenate([[0], np.cumsum(npol)[:-1]]) if len(npol) else np.zeros(0, np.int64))
            n += int(npol.sum())
            self._tjs.append(tjs); self._ords.append(ords); self._npol.append(npol)
        self._start = np.array(start, dtype=np.int64)
        self._flat = np.concatenate(flat) if flat else np.zeros(0, np.int64)
        self.offset = _Offsets(self._start, self._flat)
        self.nslots = n
        self.zero_slot = 2 * n
        self.dmax = 1
        raw = [self._level(m) for m in range(order)]
        self._keys(raw)
        self.levels = [None if r is None else r[0] for r in raw]
        self._physical(reuse)

    def _ji(self, tj):
        return (tj - self.tj12 % 2) // 2

    def _level(self, m):
        tj12, tj43, tjmax, order, ns = self.tj12, self.tj43, self.tjmax, self.order, self.nslots
        tjs, ords, npol = self._tjs[m], self._ords[m], self._npol[m]
        tot = int(npol.sum())
        if tot == 0:
            return None
        jidx = np.repeat(np.arange(len(tjs)), npol)
        pidx = np.arange(tot) - np.repeat(np.cumsum(npol) - npol, npol)
        tj = tjs[jidx]
        t, k = S.pole_at_v(tj, tj12, tj43, ords[jidx], pidx)
        jidx, pidx, tj, t, k = (np.repeat(a, 2) for a in (jidx, pidx, tj, t, k))   # parity p innermost
        p = np.tile(np.array([0, 1]), tot)
        tjp = S.pole_two_j_v(t, k, tj)
        sh = S.shift_v(t, k)
        dprime = S.delta_2_v(t, k, tj) + 2 * sh
        src = (p + (t == 3)) % 2
        ji, jip = self._ji(tj), self._ji(tjp)
        keep = ((self._SL[p, ji] > 0) & (self._SR[p, ji] > 0) & (self._SL[src, jip] > 0) & (self._SR[src, jip] > 0))
        if not keep.any():
            return None
        jidx, pidx, tj, t, k, p, tjp, dprime, src = (a[keep] for a in (jidx, pidx, tj, t, k, p, tjp, dprime, src))
        slot = p * ns + self._flat[self._start[m] + jidx] + pidx
        # groups (src, tjp) and their rows D' in order of first appearance
        gi, gkeys = _first_rank(src * (1 << 32) + tjp)
        nb = len(gkeys)
        gsrc, gtjp = gkeys >> 32, gkeys & ((1 << 32) - 1)
        dmin = int(dprime.min())
        rcode = gi * (1 << 32) + (dprime - dmin)
        uniq, first, inv = np.unique(rcode, return_index=True, return_inverse=True)
        ug = uniq >> 32
        o = np.lexsort((first, ug))                                    # by group, then first appearance
        cnt = np.bincount(ug, minlength=nb)
        gstart = np.cumsum(cnt) - cnt
        r_of = np.empty(len(uniq), dtype=np.int64)
        r_of[o] = np.arange(len(uniq)) - gstart[ug[o]]
        r = r_of[inv.reshape(-1)]
        Mr = int(cnt.max())
        rowd = np.zeros((nb, Mr), dtype=np.int64)
        rowd[ug, r_of] = (uniq & ((1 << 32) - 1)) + dmin
        # columns: poles of tjp at order m, residues of level m - shift
        ncol = S.counts_v(gtjp, tj12, tj43, m).sum(0)
        Kc = max(1, int(ncol.max()))
        cg = np.repeat(np.arange(nb), ncol)
        cc = np.arange(int(ncol.sum())) - np.repeat(np.cumsum(ncol) - ncol, ncol)
        ctj = gtjp[cg]
        ti, ki = S.pole_at_v(ctj, tj12, tj43, m, cc)
        si = S.shift_v(ti, ki)
        cslot = (gsrc[cg] * ns + self._flat[self._start[m - si] + ctj // 2]
                 + S.position_v(ctj, tj12, tj43, order - m + si - S.j_excess_v(ctj, tjmax), ti, ki))
        col = np.full((nb, Kc), self.zero_slot, dtype=np.int64)
        col[cg, cc] = cslot
        cold = np.zeros((nb, Kc), dtype=np.int64)
        cold[cg, cc] = S.delta_2_v(ti, ki, ctj)
        if len(cg):
            cmin = np.full(nb, np.iinfo(np.int64).max); np.minimum.at(cmin, cg, cold[cg, cc])
            cmax = np.full(nb, np.iinfo(np.int64).min); np.maximum.at(cmax, cg, cold[cg, cc])
            rmin = np.where(np.arange(Mr)[None] < cnt[:, None], rowd, np.iinfo(np.int64).max).min(1)
            rmax = np.where(np.arange(Mr)[None] < cnt[:, None], rowd, np.iinfo(np.int64).min).max(1)
            h = ncol > 0
            if h.any():
                self.dmax = max(self.dmax, int(np.abs(rmax[h] - cmin[h]).max()), int(np.abs(rmin[h] - cmax[h]).max()))
        sign = np.where((tjp - tj) % 4 != 0, -1, 1)
        ppa = np.array(self.pp)
        lev = {"m": m, "nb": nb, "Mr": Mr, "Kc": Kc, "gsrc": torch.from_numpy(gsrc), "gj": torch.from_numpy(gtjp // 2),
               "col": col, "rowd": torch.from_numpy(rowd), "nrow": torch.from_numpy(cnt),
               "cold": torch.from_numpy(cold), "ncol": torch.from_numpy(ncol), "t_slot": slot,
               "t_hrow": torch.from_numpy(gi * Mr + r)}
        return lev, (t, k, tj, ppa[src, 0], ppa[src, 1], sign)

    def _keys(self, raw):
        """M-matrix and Q keys, indexed in order of first appearance over all levels (L before R per task)"""
        lv = [x[1] for x in raw if x is not None]
        if not lv:
            self.Mkeys, self.Qkeys = {}, {}
            return
        t, k, tj, pL, pR, sign = (np.concatenate([x[i] for x in lv]) for i in range(6))
        KM, TJM = self.order + 2, self.tjmax + 2 * self.order + 4
        PPM = int(np.array(self.pp).max()) + 1
        code = lambda d, pq: (((t * KM + k) * 2 + d) * PPM + pq) * TJM + tj
        idx, uniq = _first_rank(np.stack([code(0, pL), code(1, pR)], 1).reshape(-1))
        u = uniq
        tju, u = u % TJM, u // TJM
        ppu, u = u % PPM, u // PPM
        du, u = u % 2, u // 2
        ku, tu = u % KM, u // KM
        self.Mkeys = {((int(a), int(b)), int(d), self.tj12 if d == 0 else self.tj43, int(c), int(e)): i
                      for i, (a, b, d, c, e) in enumerate(zip(tu, ku, du, ppu, tju))}
        qidx, quniq = _first_rank(((t * KM + k) * TJM + tj) * 2 + (sign > 0))
        u = quniq
        su, u = u % 2, u // 2
        tju, u = u % TJM, u // TJM
        self.Qkeys = {((int(u_ // KM), int(u_ % KM)), int(a), 1 if b else -1): i
                      for i, (u_, a, b) in enumerate(zip(u, tju, su))}
        o = 0
        for x in raw:
            if x is None:
                continue
            lev, n = x[0], len(x[1][0])
            lev["t_L"] = torch.from_numpy(idx[2 * o:2 * (o + n):2].copy())
            lev["t_R"] = torch.from_numpy(idx[2 * o + 1:2 * (o + n):2].copy())
            lev["t_q"] = torch.from_numpy(qidx[o:o + n].copy())
            o += n
        # distinct (L matrix, Q value) pairs: the pole step multiplies by the product q L, tabulated once
        pair = np.concatenate([idx[0::2] * len(self.Qkeys) + qidx])
        pidx, self.LQpairs = _first_rank(pair)
        self.LQpairs = np.stack([self.LQpairs // len(self.Qkeys), self.LQpairs % len(self.Qkeys)], 1)
        o = 0
        for x in raw:
            if x is None:
                continue
            n = len(x[1][0])
            x[0]["t_qL"] = torch.from_numpy(pidx[o:o + n].copy())
            o += n

    def _physical(self, reuse):
        """logical -> physical slots (self.phys: -1 for unborn slots, which are zero and read as the zero
        slot).  reuse: greedy recycling of a slot after the level that reads it as a Cauchy column (the
        gather copies the columns before the pole step writes, so that level may reuse it), or after its
        own level if it is never read; same allocation order as the reference builder (LIFO free list)."""
        z, N = self.zero_slot, 2 * self.nslots
        born = np.full(N, -1, dtype=np.int64)
        last = np.full(N, -1, dtype=np.int64)
        for lev in self.levels:
            if lev is None:
                continue
            born[lev["t_slot"]] = lev["m"]
            c = lev["col"].reshape(-1)
            c = c[c != z]
            np.maximum.at(last, c, lev["m"])
        phys = np.full(N, -1, dtype=np.int64)
        if not reuse:
            phys[born >= 0] = np.nonzero(born >= 0)[0]
            self.nphys = N
        else:
            order = np.argsort(born, kind="stable")
            order = order[born[order] >= 0]
            bl = born[order]
            cuts = np.flatnonzero(np.diff(bl)) + 1
            free, release, n = np.zeros(0, np.int64), {}, 0
            for seg in np.split(order, cuts):
                if not len(seg):
                    continue
                m = int(born[seg[0]])
                pend = [kk for kk in release if kk <= m]
                if pend:
                    free = np.concatenate([free] + [np.concatenate(release.pop(kk)) for kk in pend])
                take = min(len(seg), len(free))
                ph = np.empty(len(seg), dtype=np.int64)
                ph[:take] = free[::-1][:take]
                ph[take:] = n + np.arange(len(seg) - take)
                n += len(seg) - take
                free = free[:len(free) - take]
                phys[seg] = ph
                d = last[seg]
                rel = np.where(d > m, d, m + 1)
                _, fi = np.unique(rel, return_index=True)
                for kk in rel[np.sort(fi)]:
                    release.setdefault(int(kk), []).append(ph[rel == kk])
            self.nphys = n
        self.phys = phys
        self.zero_slot = self.nphys
        for lev in self.levels:
            if lev is None:
                continue
            lev["t_slot"] = torch.from_numpy(phys[lev["t_slot"]])
            c = lev["col"]
            ok = c != z
            pc = np.full(c.shape, self.nphys, dtype=np.int64)
            pc[ok] = phys[c[ok]]
            lev["col"] = torch.from_numpy(np.where(pc >= 0, pc, self.nphys))

    def phys_slots(self, logical):
        """physical slots of logical ones (numpy array; the zero slot where never written)"""
        logical = np.asarray(logical)
        ok = logical < len(self.phys)
        p = self.phys[np.where(ok, logical, 0)]
        return np.where(ok & (p >= 0), p, self.zero_slot)

    def phys_slot(self, logical):
        return int(self.phys_slots(np.array([logical]))[0])

    def counts(self):
        """output entries of the Cauchy and pole steps per right-hand side (rows padded per level,
        structures padded to Smax^2) -- what run() computes"""
        h = sum(lev["nb"] * lev["Mr"] for lev in self.levels if lev) * self.Smax ** 2
        r = sum(len(lev["t_slot"]) for lev in self.levels if lev) * self.Smax ** 2
        return h, r


def cauchy_tables(lev):
    """per-level Cauchy tables from the row/column pole positions: cdiff = D' - D_i (int32, 0 for padding
    and for coinciding positions) and cidx (index into the table of +-2/k: 0 -> zero, 2k-1 -> +2/k, 2k -> -2/k)"""
    rowd, cold = lev["rowd"], lev["cold"]
    vr = torch.arange(lev["Mr"], device=rowd.device)[None] < lev["nrow"][:, None]
    vc = torch.arange(lev["Kc"], device=cold.device)[None] < lev["ncol"][:, None]
    diff = torch.where(vr[:, :, None] & vc[:, None, :], rowd[:, :, None] - cold[:, None, :], 0)
    cidx = torch.where(diff == 0, 0, 2 * diff.abs() - (diff > 0).long())
    return diff.to(torch.int32), cidx


class Recursion:
    """Numbers (M matrices, Q values, Cauchy coefficients) on the GPU for one Plan.
    M: dict as in dump.read_M (strings) -- matrices for every key of plan.Mkeys; a setup.MTable (b3d_setup,
    already on the GPU); or None for random O(1) matrices (timing runs: the cost does not depend on the values)."""

    def __init__(self, plan, M, L, pole="cuda", cauchy="auto", task_chunk=8192, kcut=128, ncut=128):
        """pole: "cuda" (two batched S x S GEMMs on CUDA cores, blocks/smallgemm.py) | "gemm" (batched
        q L(x)R RNS GEMM) | "elementwise"; cauchy: "cuda" (CUDA-core kernel using that
        the coefficients are 2/integer, blocks/cauchy.py; cost ~ K) | "rns" (batched RNS GEMM; cost ~
        CRT per entry) | "auto" (cuda for levels with <= kcut prime poles or <= ncut columns S^2 B, rns above:
        at B = 1 the CUDA-core kernel wins at every level)"""
        self.plan, self.L, self.pole, self.cauchy, self.task_chunk = plan, L, pole, cauchy, task_chunk
        self.kcut, self.ncut = kcut, ncut
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
        if hasattr(M, "padded"):                        # setup.MTable: gather on the GPU, absent keys -> zero
            mats = M.padded(Sm)
            zero = mats.shape[0] - 1
            rows = [0] * nM
            for key, i in plan.Mkeys.items():
                rows[i] = M.index.get(key, zero)
            self.Mtab = _take(mats, torch.tensor(rows, dtype=torch.long, device=dev))
            return
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
        # pole-step operands, aligned per row once: R matrices, and q L for every (L, Q) pair of the plan
        self.Ralign = mpf._align_rows(self.Mtab)
        pr = torch.from_numpy(plan.LQpairs).to(dev)
        Sm = plan.Smax
        QL = mpf.mul(_take(self.Mtab, pr[:, 0]), _expand(_take(self.Qtab, pr[:, 1])[:, None, None], (len(pr), Sm, Sm)))
        self.QLalign = mpf._align_rows(QL)

    def run(self, h_inf, timer=None, prof=None, hook=None):
        """h_inf: MPF (2, >= order, Jh, Smax^2, B).  Returns residues MPF (nphys + 1, Smax^2, B) (physical
        slots).  hook(m, res) is called after each level (e.g. Derivs.accumulate)."""
        pl, L, Sm = self.plan, self.L, self.plan.Smax
        S2, B = Sm * Sm, h_inf.shape[-1]
        res = mpf.zeros((pl.nphys + 1, S2, B), L)
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
                cdiff, cidx = cauchy_tables(lev)
                if self.cauchy == "cuda" or (self.cauchy == "auto" and (Kc <= self.kcut or S2 * B <= self.ncut)):
                    H = mpf.add(cauchy_kernel(X, cdiff), Hi)
                else:
                    C = _take(self.Ctab, cidx)                                        # (nb, Mr, Kc)
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
                out = self._pole(Ht[sl], lev["t_L"][sl], lev["t_R"][sl], lev["t_q"][sl], lev["t_qL"][sl])
                res.m[:, lev["t_slot"][sl]] = out.m
                res.e[lev["t_slot"][sl]] = out.e
            if prof is not None:
                _tick(prof, "pole")
            if hook is not None:
                hook(m, res)
                if prof is not None:
                    _tick(prof, "moments")
            if timer:
                timer(m)
        return res

    def _pole(self, Ht, iL, iR, iq, iqL=None):
        """out[t, a0 b0, :] = q_t sum_{a,b} L_t[a0,a] R_t[b0,b] Ht[t, a b, :]"""
        Sm, B = self.plan.Smax, Ht.shape[-1]
        T, S2 = Ht.shape[0], Sm * Sm
        if self.pole == "cuda":           # two batched S x S GEMMs on CUDA cores: (q L) (R H^T)^T, pre-aligned tables
            H4 = Ht.reshape(T, Sm, Sm, B)                                          # (t, a, b, k)
            Hb = _permute(H4, (0, 2, 1, 3)).reshape(T, Sm, Sm * B)                 # (t, b, a k)
            Y = smallgemm(None, Hb, aligned=(self.Ralign[0][:, iR], self.Ralign[1][iR])).reshape(T, Sm, Sm, B)
            Y = _permute(Y, (0, 2, 1, 3)).reshape(T, Sm, Sm * B)                   # (t, a, b0 k)
            return smallgemm(None, Y, aligned=(self.QLalign[0][:, iqL], self.QLalign[1][iqL])).reshape(T, S2, B)
        Lt, Rt = _take(self.Mtab, iL), _take(self.Mtab, iR)                       # (T, a0, a), (T, b0, b)
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
        """physical slot of residue (p, m, jidx, pidx) (meaningful after run() only without reuse)"""
        return self.plan.phys_slot(p * self.plan.nslots + self.plan.offset[(m, jidx)] + pidx)
