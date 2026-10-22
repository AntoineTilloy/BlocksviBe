"""blocks_3d's post-recursion stages on the GPU: residues -> log(r) derivatives at r_c -> r^Delta restored
-> Delta -> x shift -> numerator polynomials (Log_r_Derivatives, before write_output's coordinate
transforms), for every internal spin, parity, structure and right-hand side of a batched recursion.

blocks_3d (conformal_block_derivatives.cxx, restore_r_delta.cxx, shift_block/*.cxx), per (spin j, parity,
structure), with w[k][n] = n^k r_c^n, r_c = 3 - 2 sqrt 2, R = Lambda - n_lambda, poles i of spin j at full
order (positions D_i, shifts s_i), x = Delta - ub_j:

  c[k]   = sum_{m <= order} w[k][m] h_inf[m]                    (h_inf part)
  v_i[k] = sum_m w[k][m + s_i] res[m][i]                        (moments; here accumulated level by level)
  g_e(Delta) = (d/dlog r)^e (r^Delta h) / r^Delta = sum_{d<=e} C(e,d) Delta^(e-d) (c[d] + sum_i v_i[d]/(Delta - D_i))
             = P_e(Delta) + sum_i a_i[e] / (Delta - D_i),
      a_i[e]  = sum_{d<=e} C(e,d) D_i^(e-d) v_i[d]                                     (residues, O(R^2))
      P_e[t]  = C(e,t) c[e-t] + sum_{u=t+1}^{e} C(e,u) G[e-u][u-t-1],  G[d][w] = sum_i D_i^w v_i[d]
  numerator_e(x) = P_e(x + ub) Pi(x) + sum_i a_i[e] Pi_i(x),  Pi = prod_i (x - x_i), Pi_i = Pi / (x - x_i)

(blocks_3d's restore_r_delta divides each pole's polynomial by (Delta - D_i), O(R^3) per pole; the form
above is the same quantity.)  Every map is linear with exact (dyadic rational) coefficients, so on the GPU:

  1. moments: after each recursion level, y = res r_c^n, then v[k] += n^k y for k <= R (one CUDA-core
     kernel, one-limb multiplications by n): no residue has to be kept for this stage;
  2. c: one GEMM (R+1) x (order+1);  a: batched (R+1) x (R+1) GEMMs over (spin, pole);
     G: batched GEMM over spins with K = poles; P: one GEMM with a fixed binomial matrix;
  3. numerators: batched GEMM over spins, A_j = [Pi_{j,i}(x) | Pi_j(x) (x+ub_j)^t (t <= R)], i.e. the
     Delta -> x shift and together_with_factors in one product, K = R + 1 + poles.

Pole shifting (kept_pole_order < order) is not implemented: the stress-tensor runs use kept_pole_order =
order.  All GEMMs are the RNS/INT8 tensor-core sdpvibe.mpf.gemm (normwise error, rounded once per entry).
"""
from math import comb
import cupy as cp
import mpmath as mp
import numpy as np
import torch

from . import structure as S
from .recursion_gpu import gemm as RG_gemm, bgemm, _take, _expand, _permute, _tick
from .. import mpf
from ..rns import dev

_SRC = r"""
// v[row_t, col, k] += n_t^k y[t, col] for k < R1 (y: normalised MPF, n_t < 2^31)
extern "C" __global__ void momacc(const int* Y, const i64* eY, const int* NN, const i64* rows,
                                  int* Mo, i64* eMo, i64 T, i64 N, i64 nrows, int R1) {
    i64 idx = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (idx >= T * N) return;
    i64 col = idx % N, t = idx / N;
    u32 y[L + 1], a[L], wa[L + 2], wb[L + 2], r[L + 1];
    for (int k = 0; k < L; k++) y[k] = (u32)Y[k * T * N + idx];
    bool ny = y[L - 1] >> 31;
    if (ny) negate(y, L);
    if (bitlen(y, L) == 0) return;
    i64 ey = eY[idx];
    u64 n = (u64)NN[t];
    i64 P = nrows * R1 * N, o0 = (rows[t] * N + col) * R1;      // moments (row, col, k), k innermost
    for (int kk = 0; kk < R1; kk++) {
        i64 o = o0 + kk;
        for (int k = 0; k < L; k++) a[k] = (u32)Mo[k * P + o];
        bool na = a[L - 1] >> 31;
        if (na) negate(a, L);
        i64 ea = eMo[o];
        if (bitlen(a, L) == 0) ea = EZ;
        i64 tt = (ea > ey ? ea : ey) - 32;
        i64 da = ea - tt, db = ey - tt;
        int sa = da < -(i64)(32 * (L + 3)) ? 32 * (L + 3) : (int)(-da);
        int sb = db < -(i64)(32 * (L + 3)) ? 32 * (L + 3) : (int)(-db);
        if (ea == EZ) { for (int k = 0; k < L + 2; k++) wa[k] = 0; } else shift_mag(a, L, sa, wa, L + 2);
        shift_mag(y, L, sb, wb, L + 2);
        if (na) negate(wa, L + 2);
        if (ny) negate(wb, L + 2);
        u64 c = 0;
        for (int k = 0; k < L + 2; k++) { u64 v = (u64)wa[k] + wb[k] + c; wa[k] = (u32)v; c = v >> 32; }
        bool ns = wa[L + 1] >> 31;
        if (ns) negate(wa, L + 2);
        int bl = bitlen(wa, L + 2);
        if (bl == 0) { for (int k = 0; k < L; k++) Mo[k * P + o] = 0; eMo[o] = EZ; }
        else {
            int s = bl - NB;
            shift_mag(wa, L + 2, s, r, L);
            if (ns) negate(r, L);
            for (int k = 0; k < L; k++) Mo[k * P + o] = (int)r[k];
            eMo[o] = tt + s;
        }
        if (kk + 1 < R1) {                       // y *= n, renormalised (|y| < 2^NB)
            u64 cc = 0;
            for (int k = 0; k < L; k++) { u64 v = (u64)y[k] * n + cc; y[k] = (u32)v; cc = v >> 32; }
            y[L] = (u32)cc;
            int s = bitlen(y, L + 1) - NB;
            if (s > 0) { shift_mag(y, L + 1, s, r, L); for (int k = 0; k < L; k++) y[k] = r[k]; ey += s; }
        }
    }
}

// r = a + b for sign-magnitude normalised operands (zero: magnitude 0); result normalised (or zero, EZ)
__device__ void mpadd(const u32* a, bool na, i64 ea, const u32* b, bool nb, i64 eb, u32* r, bool* nr, i64* er) {
    u32 wa[L + 2], wb[L + 2];
    bool za = bitlen(a, L) == 0, zb = bitlen(b, L) == 0;
    if (za && zb) { for (int k = 0; k < L; k++) r[k] = 0; *nr = false; *er = EZ; return; }
    if (za) ea = EZ;
    if (zb) eb = EZ;
    i64 tt = (ea > eb ? ea : eb) - 32;
    i64 da = ea - tt, db = eb - tt;
    int sa = da < -(i64)(32 * (L + 3)) ? 32 * (L + 3) : (int)(-da);
    int sb = db < -(i64)(32 * (L + 3)) ? 32 * (L + 3) : (int)(-db);
    if (za) { for (int k = 0; k < L + 2; k++) wa[k] = 0; } else shift_mag(a, L, sa, wa, L + 2);
    if (zb) { for (int k = 0; k < L + 2; k++) wb[k] = 0; } else shift_mag(b, L, sb, wb, L + 2);
    if (na) negate(wa, L + 2);
    if (nb) negate(wb, L + 2);
    u64 c = 0;
    for (int k = 0; k < L + 2; k++) { u64 v = (u64)wa[k] + wb[k] + c; wa[k] = (u32)v; c = v >> 32; }
    bool ns = wa[L + 1] >> 31;
    if (ns) negate(wa, L + 2);
    int bl = bitlen(wa, L + 2);
    if (bl == 0) { for (int k = 0; k < L; k++) r[k] = 0; *nr = false; *er = EZ; return; }
    int s = bl - NB;
    shift_mag(wa, L + 2, s, r, L);
    *nr = ns; *er = tt + s;
}

// windowed moments, one thread per (row, k, col):
//   v[row, col, k] += sum_w y[w, row, col] * n_w^k,  n_w = m0 + w + srow[row]
// (y: residues of W consecutive levels already scaled by r_c^n, zero where a row got none; n^k exact
// integers of KL limbs from the table NK).  The W exact products are aligned to a common exponent (an upper
// bound of the largest), truncated 32 bits below an (L+1)-limb window, summed exactly in L + 3 limbs and
// rounded once, then added to v.
extern "C" __global__ void momflush(const int* Yb, const i64* eYb, const int* srow, const int* NK, const int* NKb,
                                    int m0, int W, int* Mo, i64* eMo, i64 nrows, i64 N, int R1, i64 nW) {
    i64 idx = blockIdx.x * (i64)blockDim.x + threadIdx.x; if (idx >= nrows * R1 * N) return;
    i64 kk = idx % R1, col = (idx / R1) % N, row = idx / (N * R1);     // k innermost: a warp shares its y loads
    i64 PY = (i64)W * nrows * N, PW = nW * R1;        // limb-plane strides of the staging buffer and of NK
    i64 E = EZ;
    for (int w = 0; w < W; w++) {                      // pass 1: exponent bound of the largest product
        i64 o = (i64)w * nrows * N + row * N + col;
        if (Yb[(L - 1) * PY + o] == 0) continue;       // normalised: zero iff the top limb is 0
        i64 t = eYb[o] + NB + NKb[(i64)(m0 + w + srow[row]) * R1 + kk];
        if (t > E) E = t;
    }
    if (E == EZ) return;
    i64 T = E - 32 * (L + 1);                          // weight of the accumulator's lowest limb
    u32 acc[L + 3];
    for (int k = 0; k < L + 3; k++) acc[k] = 0;
    for (int w = 0; w < W; w++) {
        i64 o = (i64)w * nrows * N + row * N + col;
        u32 a[L];
        for (int k = 0; k < L; k++) a[k] = (u32)Yb[k * PY + o];
        bool na = a[L - 1] >> 31;
        if (na) negate(a, L);
        if (bitlen(a, L) == 0) continue;
        i64 ow = (i64)(m0 + w + srow[row]) * R1 + kk;
        i64 sh = T - eYb[o];                            // product limbs >> sh land on acc's grid
        if (sh >= 32 * (L + KL)) continue;              // entirely below the window
        u32 p[L + KL];
        for (int k = 0; k < L + KL; k++) p[k] = 0;
        for (int x = 0; x < KL; x++) {
            u32 bx = (u32)NK[x * PW + ow];
            if (!bx) continue;
            u64 c = 0;
            for (int y = 0; y < L; y++) { u64 v = (u64)bx * a[y] + p[x + y] + c; p[x + y] = (u32)v; c = v >> 32; }
            p[x + L] = (u32)c;
        }
        u32 q[L + 2];
        shift_mag(p, L + KL, (int)sh, q, L + 2);
        if (na) { u64 bo = 0; for (int k = 0; k < L + 2; k++) { u64 v = (u64)acc[k] - q[k] - bo; acc[k] = (u32)v; bo = (v >> 32) & 1; }
                  acc[L + 2] -= (u32)bo; }
        else { u64 c = 0; for (int k = 0; k < L + 2; k++) { u64 v = (u64)acc[k] + q[k] + c; acc[k] = (u32)v; c = v >> 32; }
               acc[L + 2] += (u32)c; }
    }
    bool ns = acc[L + 2] >> 31;
    if (ns) negate(acc, L + 3);
    int bl = bitlen(acc, L + 3);
    if (bl == 0) return;
    u32 r[L], mo[L], t[L + 1];
    int s = bl - NB;
    shift_mag(acc, L + 3, s, r, L);
    i64 o = (row * N + col) * R1 + kk, P = nrows * R1 * N;
    for (int k = 0; k < L; k++) mo[k] = (u32)Mo[k * P + o];
    bool nm = mo[L - 1] >> 31;
    if (nm) negate(mo, L);
    bool nr; i64 er;
    mpadd(mo, nm, eMo[o], r, ns, T + s, t, &nr, &er);
    if (nr) negate(t, L);
    for (int k = 0; k < L; k++) Mo[k * P + o] = (int)t[k];
    eMo[o] = er;
}
"""
_mods = {}


def _kernel(L, name="momacc", KL=1):
    key = (L, KL)
    if key not in _mods:
        src = (mpf._SRC + _SRC).replace("__L__", str(L)).replace("__LMAX__", str(mpf.LMAX_WIDE))
        src = src.replace("#define NB (32 * L - 2)", f"#define NB (32 * L - 2)\n#define KL {KL}")
        _mods[key] = cp.RawModule(code=src, options=("-std=c++17",))
    return _mods[key].get_function(name)


def _cat(parts, dim):
    """concatenate MPFs along an entry dimension"""
    return mpf.MPF(torch.cat([p.m for p in parts], dim + 1), torch.cat([p.e for p in parts], dim))


def exact_mpf(ints, exps, L, shape):
    """exact values ints[i] * 2^exps[i] (Python ints of any size) -> MPF, rounded once to NB bits (half up
    on the magnitude, like mpf.from_mp) -- without going through mpmath"""
    NB = mpf.NBITS(L)
    out_i, out_e = [], []
    for v, e in zip(ints, exps):
        if v:
            a = -v if v < 0 else v
            sh = a.bit_length() - NB
            if sh > 0:
                a = (a + (1 << (sh - 1))) >> sh
                e += sh
            v = -a if v < 0 else a
        out_i.append(v); out_e.append(e)
    return mpf.from_ints(out_i, out_e, L, shape)


def _polymul_lin(p, a):
    """p(y) * (y + a), integer coefficients (lowest first)"""
    out = [0] * (len(p) + 1)
    for k, c in enumerate(p):
        out[k] += a * c
        out[k + 1] += c
    return out


def _syndiv_all(p, roots):
    """[p(y) / (y - r) for r in roots] (exact, integer coefficients), vectorised over the roots"""
    n = len(p) - 1
    r = np.array(roots, dtype=object)
    acc = np.zeros(len(roots), dtype=object)
    q = np.empty((n, len(roots)), dtype=object)
    for k in range(n, 0, -1):
        acc = p[k] + r * acc
        q[k - 1] = acc
    assert all(p[0] + r * acc == 0)
    return [list(q[:, i]) for i in range(len(roots))]


def _syndiv(p, root):
    """p(y) / (y - root) for a root of p (exact), integer coefficients"""
    n = len(p) - 1
    q = [0] * n
    acc = 0
    for k in range(n, 0, -1):
        acc = p[k] + root * acc
        q[k - 1] = acc
    assert p[0] + root * acc == 0
    return q


class Derivs:
    """Post-recursion stages for one recursion Plan (any batch of right-hand sides with lambda orders
    n >= Lambda - R).  two_js: internal spins to output (default: all two_j <= two_j_max)."""

    def __init__(self, plan, R, L, two_js=None):
        self.plan, self.R, self.L = plan, R, L
        pl, order = plan, plan.order
        tj12, tj43, tjmax = pl.tj12, pl.tj43, pl.tjmax
        self.tjs = list(two_js) if two_js is not None else list(range(tj12 % 2, tjmax + 1, 2))
        Jo = self.Jo = len(self.tjs)
        self.poles = [S.poles(tj, tj12, tj43, order) for tj in self.tjs]
        NP = self.NP = max(len(p) for p in self.poles)
        self.Dout = NP + R + 1
        # moment rows ((p * Jo + jo) * NP + i), per level: (physical slot, row, n = m + s_i)
        tjo = np.array(self.tjs, dtype=np.int64)
        self.levels = []
        for m in range(order):
            tjs = pl._tjs[m]
            jsel = np.flatnonzero(np.isin(tjs, tjo))
            jo = np.searchsorted(tjo, tjs[jsel]) if np.all(np.diff(tjo) > 0) else \
                np.array([self.tjs.index(int(x)) for x in tjs[jsel]], dtype=np.int64)
            npol = pl._npol[m][jsel]
            tot = int(npol.sum())
            if tot == 0:
                self.levels.append(None)
                continue
            rep = np.repeat(np.arange(len(jsel)), npol)
            pidx = np.arange(tot) - np.repeat(np.cumsum(npol) - npol, npol)
            jidx, tj, jo_ = jsel[rep], tjs[jsel][rep], jo[rep]
            t, k = S.pole_at_v(tj, tj12, tj43, pl._ords[m][jidx], pidx)
            i = S.position_v(tj, tj12, tj43, order, t, k)
            nn = m + S.shift_v(t, k)
            jidx, pidx, jo_, i, nn = (np.repeat(a, 2) for a in (jidx, pidx, jo_, i, nn))
            p = np.tile(np.array([0, 1]), tot)
            ph = pl.phys_slots(p * pl.nslots + pl._flat[pl._start[m] + jidx] + pidx)
            ok = ph != pl.zero_slot
            if not ok.any():
                self.levels.append(None)
                continue
            self.levels.append((torch.from_numpy(ph[ok]).to(dev), torch.from_numpy(((p * Jo + jo_) * NP + i)[ok]).to(dev),
                                torch.from_numpy(nn[ok].astype(np.int32)).to(dev)))
        self._constants()

    def _constants(self):
        L, R, order, Jo, NP = self.L, self.R, self.plan.order, self.Jo, self.NP
        mp.mp.prec = 32 * L + 64
        rc = 3 - 2 * mp.sqrt(2)
        self.rcpow = mpf.from_mp([rc ** n for n in range(order + 1)], L, (order + 1,))
        self.Wh = mpf.from_mp([(mp.mpf(n) ** k if (n or k) else 1) * rc ** n
                               for k in range(R + 1) for n in range(order + 1)], L, (R + 1, order + 1))
        # pole positions D_i = d/2 per (spin, pole); table of C(e,k) D^(e-k) = C(e,k) d^(e-k) 2^-(e-k) per distinct d
        d2 = sorted({S.delta_2(P, tj) for tj, pl in zip(self.tjs, self.poles) for P in pl})
        di = {d: k + 1 for k, d in enumerate(d2)}                     # 0: padding (all zero)
        ints, exps = [0] * ((R + 1) * (R + 1)), [0] * ((R + 1) * (R + 1))
        for d in d2:
            pw = [d ** q for q in range(R + 1)]
            for e in range(R + 1):
                ints += [comb(e, k) * pw[e - k] if k <= e else 0 for k in range(R + 1)]
                exps += [k - e for k in range(R + 1)]
        self.Ttab = exact_mpf(ints, exps, L, (len(d2) + 1, R + 1, R + 1))
        didx = np.zeros((Jo, NP), dtype=np.int64)
        for jo, (tj, pl) in enumerate(zip(self.tjs, self.poles)):
            didx[jo, :len(pl)] = [di[S.delta_2(P, tj)] for P in pl]
        self.didx = torch.from_numpy(didx).to(dev)                    # T, Vand gathered per spin chunk
        self._small_constants()
        # numerator matrix per spin: columns Pi_i(x), then Pi(x) (x + ub)^t (t <= R); in y = 2x the roots
        # y_i = 2 x_i are integers (<= 0), so everything is exact integer arithmetic, scaled at the end
        Dout, W = self.Dout, R + 1 + NP
        ints, exps = [0] * (Jo * Dout * W), [0] * (Jo * Dout * W)
        for jo, (tj, pl) in enumerate(zip(self.tjs, self.poles)):
            two_ub = tj + 1 if tj <= 1 else tj + 2
            roots = [S.delta_2(P, tj) - two_ub for P in pl]
            npj = len(roots)
            Pi = [1]
            for y0 in roots:
                Pi = _polymul_lin(Pi, -y0)
            cols = []
            q = Pi
            for t in range(R + 1):                                    # 2^-(npj+t) Pi_y(y) (y + 2ub)^t
                cols.append((NP + t, q, npj + t))
                q = _polymul_lin(q, two_ub)
            cols += [(i, qi, npj - 1) for i, qi in enumerate(_syndiv_all(Pi, roots))]
            for cc, poly, sc in cols:                                 # x^k coefficient: v 2^(k - sc)
                base = jo * Dout * W + cc
                ints[base:base + len(poly) * W:W] = poly
                exps[base:base + len(poly) * W:W] = range(-sc, len(poly) - sc)
        self.Anum = exact_mpf(ints, exps, L, (Jo, Dout, W))

    def _small_constants(self):
        """R-dependent index tables of the polynomial part (finish_iter): H[e][s] = sum_d C(e,d) G[s-1-d][d],
        P_e[t] = C(e,t) c[e-t] + H[e][e-t]"""
        L, R = self.L, self.R
        self.Cb = exact_mpf([comb(e, d) for e in range(R + 1) for d in range(R + 1)], [0] * (R + 1) ** 2, L,
                            (R + 1, R + 1))
        Rw = max(R, 1)
        sd = [(sv, d) for sv in range(R + 1) for d in range(R + 1)]
        self.gidx = torch.tensor([(sv - 1 - d) * (R + 1) + d if 1 <= sv and d <= sv - 1 else Rw * (R + 1)
                                  for sv, d in sd], device=dev)       # G[w][d] -> G'[s][d] (last: zero)
        te = [(t, e) for t in range(R + 1) for e in range(R + 1)]
        self.cidx_te = torch.tensor([e - t if t <= e else R + 1 for t, e in te], device=dev)     # c[e-t] (R+1: zero)
        self.hidx_te = torch.tensor([e * (R + 1) + (e - t) if t <= e else (R + 1) ** 2 for t, e in te], device=dev)
        self.cb_te = exact_mpf([comb(e, t) if t <= e else 0 for t, e in te], [0] * len(te), L, (len(te),))

    def for_order(self, R):
        """the same stages for a smaller R (a higher lambda order of the task): constants are views of this
        object's (Pi(x)(x+ub)^t for t <= R, C(e,k) D^(e-k) for e <= R, ...), so no extra GPU memory"""
        assert R <= self.R
        d = object.__new__(Derivs)
        d.__dict__.update(self.__dict__)
        d.R, d.Dout = R, self.NP + R + 1
        R1, K = R + 1, self.NP + R + 1                                # columns: Pi_i, then t <= R
        d.Anum = mpf.MPF(self.Anum.m[:, :, :d.Dout, :K], self.Anum.e[:, :d.Dout, :K])
        d.Ttab = mpf.MPF(self.Ttab.m[:, :, :R1, :R1].contiguous(), self.Ttab.e[:, :R1, :R1].contiguous())
        d.Wh = mpf.MPF(self.Wh.m[:, :R1].contiguous(), self.Wh.e[:R1].contiguous())
        d._small_constants()
        return d

    def start(self, B, window=8):
        """zero moment accumulators for a batch of B right-hand sides.  window: levels staged before the
        moments are updated (momflush: accumulator traffic / window); 1 = per-level kernel (momacc)"""
        self.N = self.plan.Smax ** 2 * B
        self.B = B
        nrows = 2 * self.Jo * self.NP
        self.mom = mpf.zeros((nrows, self.N, self.R + 1), self.L)           # (row, col, k)
        self.W = window
        if window > 1:
            self.stage = torch.zeros((self.L, window, nrows, self.N), dtype=torch.int32, device=dev)
            self.stage_e = torch.zeros((window, nrows, self.N), dtype=torch.int64, device=dev)
            if not hasattr(self, "srow"):
                srow = np.zeros((2, self.Jo, self.NP), dtype=np.int32)
                for jo, pl in enumerate(self.poles):
                    srow[:, jo, :len(pl)] = [S.shift(P) for P in pl]
                self.srow = torch.from_numpy(srow.reshape(-1)).to(dev)
            self.m0, self.pending = 0, False
            if getattr(self, "NKR", None) != self.R:      # exact n^k (limb planes) and bit lengths
                order, R1 = self.plan.order, self.R + 1
                KL = max(1, ((order ** self.R).bit_length() + 31) // 32)
                v = [n ** k for n in range(order + 1) for k in range(R1)]
                self.NK = torch.from_numpy(np.array([[(x >> (32 * l)) & 0xFFFFFFFF for x in v] for l in range(KL)],
                                                    dtype=np.uint32).view(np.int32)).to(dev)
                self.NKb = torch.tensor([x.bit_length() for x in v], dtype=torch.int32, device=dev)
                self.NKR = self.R

    def _flush(self):
        nrows, R1 = self.mom.shape[0], self.R + 1
        th = 128
        tot = nrows * R1 * self.N
        _kernel(self.L, "momflush", self.NK.shape[0])(((tot + th - 1) // th,), (th,),
            (cp.from_dlpack(self.stage), cp.from_dlpack(self.stage_e), cp.from_dlpack(self.srow),
             cp.from_dlpack(self.NK), cp.from_dlpack(self.NKb), np.int32(self.m0), np.int32(self.W),
             cp.from_dlpack(self.mom.m), cp.from_dlpack(self.mom.e), np.int64(nrows), np.int64(self.N),
             np.int32(R1), np.int64(self.plan.order + 1)))
        self.stage.zero_()
        self.pending = False

    def accumulate(self, m, res):
        """recursion hook: add level m's residues of output spins to the moments"""
        lev = self.levels[m]
        if self.W > 1 and self.pending and m - self.m0 >= self.W:
            self._flush()
        if lev is None:
            return
        sl, rw, nn = lev
        T = sl.numel()
        X = _take(res, sl).reshape(T, self.N)
        Y = mpf.mul(X, _expand(_take(self.rcpow, nn.long())[:, None], (T, self.N)))
        if self.W > 1:                                   # stage; the flush multiplies by n^k
            if not self.pending:
                self.m0, self.pending = m, True
            w = m - self.m0
            self.stage[:, w, rw] = Y.m
            self.stage_e[w, rw] = Y.e
            if w == self.W - 1:
                self._flush()
            return
        th = 128
        _kernel(self.L)(((T * self.N + th - 1) // th,), (th,),
                        (cp.from_dlpack(Y.m.contiguous()), cp.from_dlpack(Y.e.contiguous()), cp.from_dlpack(nn),
                         cp.from_dlpack(rw), cp.from_dlpack(self.mom.m), cp.from_dlpack(self.mom.e),
                         np.int64(T), np.int64(self.N), np.int64(self.mom.shape[0]), np.int32(self.R + 1)))

    def offload(self):
        """move the finished moments to pinned host memory (finish_iter streams them back per spin chunk):
        frees GPU memory for the final GEMMs on small cards"""
        if getattr(self, "W", 1) > 1 and self.pending:
            self._flush()
        self.stage = self.stage_e = None
        self.mom = mpf.MPF(self.mom.m.cpu().pin_memory(), self.mom.e.cpu().pin_memory())

    def finish(self, h_inf, chunk=None):
        """h_inf: MPF (2, order + 1, Jh, Smax^2, B).  Returns numerators MPF (Jo, Dout, R+1, 2, Smax^2, B):
        [spin, x^k coefficient, log r derivative e, parity, structure a*Smax+b, rhs]"""
        return _cat([num for _, num in self.finish_iter(h_inf, chunk)], 0)

    def finish_iter(self, h_inf, chunk=None, prof=None):
        """as finish, by chunks of output spins (bounded temporaries): yields (first spin index, numerators).
        prof: dict accumulating GPU seconds per stage"""
        tick = (lambda k: _tick(prof, k)) if prof is not None else (lambda k: None)
        if getattr(self, "W", 1) > 1 and self.pending:
            self._flush()
        tick(None)
        R, Jo, NP, order = self.R, self.Jo, self.NP, self.plan.order
        N, Rw = self.N, max(R, 1)
        S2 = self.plan.Smax ** 2
        Nc = 2 * N
        jidx = torch.tensor([(tj - self.plan.tj12 % 2) // 2 for tj in self.tjs], device=dev)
        H = mpf.MPF(h_inf.m[:, :, :order + 1][:, :, :, jidx], h_inf.e[:, :order + 1][:, :, jidx])  # (2, o+1, Jo, S2, B)
        H = _permute(H, (1, 2, 0, 3, 4)).reshape(order + 1, Jo * Nc)
        c = RG_gemm(self.Wh, H).reshape(R + 1, Jo, Nc)                                  # c[d]
        c = _permute(c, (1, 0, 2))                                                       # (Jo, R+1, Nc)
        tick("h_inf part")
        mom = self.mom.reshape(2, Jo, NP, N, R + 1)
        chunk = chunk or Jo
        for j0 in range(0, Jo, chunk):
            j1 = min(Jo, j0 + chunk)
            J = j1 - j0
            v = mom[:, j0:j1]
            if v.m.device != dev:
                v = mpf.MPF(v.m.to(dev), v.e.to(dev))
            v = _permute(v, (1, 2, 4, 0, 3)).reshape(J, NP, R + 1, Nc)                   # v_i[d]
            tick("moments in")
            T = _take(self.Ttab, self.didx[j0:j1].reshape(-1))                        # (J NP, e, d): C(e,d) D_i^(e-d)
            a = bgemm(T, v.reshape(J * NP, R + 1, Nc)).reshape(J, NP, R + 1, Nc)         # a_i[e]
            tick("residues a")
            T = T.reshape(J, NP, R + 1, R + 1)                                           # [.., w, 0] = D_i^w
            Vand = mpf.MPF(T.m[:, :, :, :Rw, 0].permute(0, 1, 3, 2).contiguous(),
                           T.e[:, :, :Rw, 0].permute(0, 2, 1).contiguous())             # (J, Rw, NP)
            del T
            G = bgemm(Vand, v.reshape(J, NP, (R + 1) * Nc)).reshape(J, Rw * (R + 1), Nc)  # G[w][d]
            del v
            tick("G")
            Gz = _cat([G.reshape(J, Rw * (R + 1), Nc), mpf.zeros((J, 1, Nc), self.L)], 1)
            del G
            Gs = _permute(_take(_permute(Gz, (1, 0, 2)), self.gidx).reshape(R + 1, R + 1, J, Nc), (1, 2, 0, 3))
            del Gz                                                                       # G'[s][d] as (d, jo, s, col)
            Hm = RG_gemm(self.Cb, Gs.reshape(R + 1, -1)).reshape(R + 1, J, R + 1, Nc)  # H[e][s] (e, jo, s, col)
            del Gs
            Hm = _cat([_permute(Hm, (0, 2, 1, 3)).reshape((R + 1) ** 2, J, Nc), mpf.zeros((1, J, Nc), self.L)], 0)
            cz = _cat([_permute(c[j0:j1], (1, 0, 2)), mpf.zeros((1, J, Nc), self.L)], 0)  # (d | zero, jo, col)
            T2 = (R + 1) ** 2
            P = mpf.mul(_take(cz, self.cidx_te), _expand(self.cb_te[:, None, None], (T2, J, Nc)))
            P = mpf.add(P, _take(Hm, self.hidx_te))                                      # ((t, e), jo, col)
            del Hm, cz
            P = _permute(P.reshape(R + 1, R + 1, J, Nc), (2, 0, 1, 3)).reshape(J, R + 1, (R + 1) * Nc)  # (jo, t, e col)
            tick("polynomial part")
            Bn = _cat([a.reshape(J, NP, (R + 1) * Nc), P], 1)                            # (jo, i | t, e col)
            del P, a
            tick("layout")
            num = bgemm(self.Anum[j0:j1], Bn)                                            # (J, Dout, (R+1) Nc)
            tick("numerators")
            yield j0, num.reshape(J, self.Dout, R + 1, 2, S2, self.B)
