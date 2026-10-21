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
  3. numerators: batched GEMM over spins, A_j = [Pi_j(x) (x+ub_j)^t (t <= R) | Pi_{j,i}(x)], i.e. the
     Delta -> x shift and together_with_factors in one product, K = R + 1 + poles.

Pole shifting (kept_pole_order < order) is not implemented: the stress-tensor runs use kept_pole_order =
order.  All GEMMs are the RNS/INT8 tensor-core sdpvibe.mpf.gemm (normwise error, rounded once per entry).
"""
from fractions import Fraction
from math import comb
import cupy as cp
import mpmath as mp
import numpy as np
import torch

from . import structure as S
from .recursion_gpu import bgemm, _take, _expand, _permute, _tick
from .. import mpf
from ..rns import dev

_SRC = r"""
// v[row_t, k, col] += n_t^k y[t, col] for k < R1 (y: normalised MPF, n_t < 2^31)
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
    i64 P = nrows * R1 * N, o0 = rows[t] * R1 * N + col;
    for (int kk = 0; kk < R1; kk++) {
        i64 o = o0 + kk * N;
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
"""
_mods = {}


def _kernel(L):
    if L not in _mods:
        src = (mpf._SRC + _SRC).replace("__L__", str(L)).replace("__LMAX__", str(mpf.LMAX_WIDE))
        _mods[L] = cp.RawModule(code=src, options=("-std=c++17",)).get_function("momacc")
    return _mods[L]


def _cat(parts, dim):
    """concatenate MPFs along an entry dimension"""
    return mpf.MPF(torch.cat([p.m for p in parts], dim + 1), torch.cat([p.e for p in parts], dim))


def _dyadic(vals, L, shape):
    """exact rationals with power-of-two denominators (or ints) -> MPF, rounded to working precision"""
    return mpf.from_mp([mp.mpf(v) if isinstance(v, int) else mp.mpf(v.numerator) / v.denominator
                        for v in vals], L, shape)


def _polymul_lin(p, a):
    """p(y) * (y + a), integer coefficients (lowest first)"""
    out = [0] * (len(p) + 1)
    for k, c in enumerate(p):
        out[k] += a * c
        out[k + 1] += c
    return out


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
        # pole positions D_i (doubled, exact) per (spin, pole); table of C(e,d) D^(e-d) per distinct D
        d2 = sorted({S.delta_2(P, tj) for tj, pl in zip(self.tjs, self.poles) for P in pl})
        di = {d: k + 1 for k, d in enumerate(d2)}                     # 0: padding (all zero)
        tab = [0] * ((R + 1) * (R + 1))
        for d in d2:
            D = Fraction(d, 2)
            tab += [comb(e, k) * D ** (e - k) if k <= e else 0 for e in range(R + 1) for k in range(R + 1)]
        self.Ttab = _dyadic(tab, L, (len(d2) + 1, R + 1, R + 1))
        didx = torch.zeros((Jo, NP), dtype=torch.long)
        for jo, (tj, pl) in enumerate(zip(self.tjs, self.poles)):
            for i, P in enumerate(pl):
                didx[jo, i] = di[S.delta_2(P, tj)]
        self.didx = didx.to(dev)                                      # T, Vand gathered per spin chunk
        Rw = max(R, 1)
        # P_e[t] = C(e,t) c[e-t] + sum_{u=t+1}^{e} C(e,u) G[e-u][u-t-1];  inputs: c[d] (R+1), G[d][w] (Rw x (R+1), w major)
        Kin = (R + 1) + Rw * (R + 1)
        A = np.zeros(((R + 1) * (R + 1), Kin), dtype=object)
        for e in range(R + 1):
            for t in range(e + 1):
                A[e * (R + 1) + t, e - t] += comb(e, t)
                for u in range(t + 1, e + 1):
                    A[e * (R + 1) + t, (R + 1) + (u - t - 1) * (R + 1) + (e - u)] += comb(e, u)
        self.Apoly = _dyadic([int(x) for x in A.reshape(-1)], L, A.shape)
        # numerator matrix per spin: columns Pi(x) (x + ub)^t (t <= R), then Pi_i(x); in y = 2x the roots
        # y_i = 2 x_i are integers (<= 0), so everything is exact integer arithmetic, scaled at the end
        Dout = self.Dout
        vals = [0] * (Jo * Dout * (R + 1 + NP))
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
                cols.append((q, npj + t))
                q = _polymul_lin(q, two_ub)
            cols += [(_syndiv(Pi, y0), npj - 1) for y0 in roots]
            for c, (poly, sc) in enumerate(cols):
                cc = c if c <= R else R + 1 + (c - R - 1)
                for k, v in enumerate(poly):                          # x^k coefficient: v 2^(k - sc)
                    if v:
                        vals[(jo * Dout + k) * (R + 1 + NP) + cc] = Fraction(v) * Fraction(2) ** (k - sc)
        self.Anum = _dyadic(vals, L, (Jo, Dout, R + 1 + NP))

    def start(self, B):
        """zero moment accumulators for a batch of B right-hand sides"""
        self.N = self.plan.Smax ** 2 * B
        self.B = B
        self.mom = mpf.zeros((2 * self.Jo * self.NP, self.R + 1, self.N), self.L)

    def accumulate(self, m, res):
        """recursion hook: add level m's residues of output spins to the moments"""
        lev = self.levels[m]
        if lev is None:
            return
        sl, rw, nn = lev
        T = sl.numel()
        X = _take(res, sl).reshape(T, self.N)
        Y = mpf.mul(X, _expand(_take(self.rcpow, nn.long())[:, None], (T, self.N)))
        th = 128
        _kernel(self.L)(((T * self.N + th - 1) // th,), (th,),
                        (cp.from_dlpack(Y.m.contiguous()), cp.from_dlpack(Y.e.contiguous()), cp.from_dlpack(nn),
                         cp.from_dlpack(rw), cp.from_dlpack(self.mom.m), cp.from_dlpack(self.mom.e),
                         np.int64(T), np.int64(self.N), np.int64(self.mom.shape[0]), np.int32(self.R + 1)))

    def offload(self):
        """move the finished moments to pinned host memory (finish_iter streams them back per spin chunk):
        frees GPU memory for the final GEMMs on small cards"""
        self.mom = mpf.MPF(self.mom.m.cpu().pin_memory(), self.mom.e.cpu().pin_memory())

    def finish(self, h_inf, chunk=None):
        """h_inf: MPF (2, order + 1, Jh, Smax^2, B).  Returns numerators MPF (Jo, Dout, R+1, 2, Smax^2, B):
        [spin, x^k coefficient, log r derivative e, parity, structure a*Smax+b, rhs]"""
        return _cat([num for _, num in self.finish_iter(h_inf, chunk)], 0)

    def finish_iter(self, h_inf, chunk=None, prof=None):
        """as finish, by chunks of output spins (bounded temporaries): yields (first spin index, numerators).
        prof: dict accumulating GPU seconds per stage"""
        tick = (lambda k: _tick(prof, k)) if prof is not None else (lambda k: None)
        tick(None)
        R, Jo, NP, order = self.R, self.Jo, self.NP, self.plan.order
        N, Rw = self.N, max(R, 1)
        S2 = self.plan.Smax ** 2
        Nc = 2 * N
        jidx = torch.tensor([(tj - self.plan.tj12 % 2) // 2 for tj in self.tjs], device=dev)
        H = mpf.MPF(h_inf.m[:, :, :order + 1][:, :, :, jidx], h_inf.e[:, :order + 1][:, :, jidx])  # (2, o+1, Jo, S2, B)
        H = _permute(H, (1, 2, 0, 3, 4)).reshape(order + 1, Jo * Nc)
        c = mpf.gemm(self.Wh, H).reshape(R + 1, Jo, Nc)                                  # c[d]
        c = _permute(c, (1, 0, 2))                                                       # (Jo, R+1, Nc)
        tick("h_inf part")
        mom = self.mom.reshape(2, Jo, NP, R + 1, N)
        chunk = chunk or Jo
        for j0 in range(0, Jo, chunk):
            j1 = min(Jo, j0 + chunk)
            J = j1 - j0
            v = mom[:, j0:j1]
            if v.m.device != dev:
                v = mpf.MPF(v.m.to(dev), v.e.to(dev))
            v = _permute(v, (1, 2, 3, 0, 4)).reshape(J, NP, R + 1, Nc)                   # v_i[d]
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
            X = _permute(_cat([c[j0:j1], G], 1), (1, 0, 2)).reshape(-1, J * Nc)
            del G
            P = mpf.gemm(self.Apoly, X).reshape(R + 1, R + 1, J, Nc)                    # (e, t, jo, col)
            P = _permute(P, (2, 1, 0, 3)).reshape(J, R + 1, (R + 1) * Nc)               # (jo, t, e col)
            tick("polynomial part")
            Bn = _cat([P, a.reshape(J, NP, (R + 1) * Nc)], 1)                            # (jo, t | i, e col)
            del P, a, X
            tick("layout")
            num = bgemm(self.Anum[j0:j1], Bn)                                            # (J, Dout, (R+1) Nc)
            tick("numerators")
            yield j0, num.reshape(J, self.Dout, R + 1, 2, S2, self.B)
