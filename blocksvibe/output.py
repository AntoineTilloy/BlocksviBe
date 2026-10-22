"""blocks_3d's write_output on the GPU: numerator polynomials (Log_r_Derivatives, for all lambda orders n of
a task) -> derivatives in the xt or xt_radial coordinates, and a blocks_3d-compatible JSON writer.

blocks_3d (src/write_output), for one (spin, parity, structure) and per coefficient of the numerators
(every step is linear with scalar coefficients, so it acts on each x^k coefficient separately), order O =
Lambda, sigma = (-1)^parity, n = parity + 2 idx:

  lambda_lr_to_lrho_lrhob: F(i,j) = 2^-(i+j) sum_{k<=i, l<=j} (-1)^(j-l) C(i,k) C(j,l) num[n = i+j-k-l][e = k+l],
                           j <= min(i, O-i), F(j,i) = sigma F(i,j);
  congruence_transform with stirling S, rho_z R_z and (Delta_1+2 != 0) power_prefactor P(Delta/2):
                           Z = M F M^T with M = P R_z S (all lower triangular, so blocks_3d's truncations
                           to the stored triangle i + j <= O never drop a term);
  zzb_to_xt:               xt(n, m) = ((m-parity)/2)!/m! sum_{k<=n, l<=m} (-1)^l C(n,k) C(m,l) Z(n+m-k-l, k+l),
                           m = parity, parity+2, ... <= O - n;
  lrho_to_xt_radial:       xt_radial(i) = sum_j T(i,j) num[n = parity][j],  T = radial prefactor R_z S
                           (order O = Lambda - parity).

The composite W (outputs x inputs (idx, e)) is built once per (Lambda, parity, Delta_1+2) in mpmath, using
that an input (idx, e) only feeds the antidiagonal i + j = n + e of F, and applied to all spins,
structures, coefficients and tasks as one RNS GEMM.  Coordinates xt and xt_radial only (what the
stress-tensor pipeline uses).
"""
import json
import math
import mpmath as mp
import numpy as np
import torch

from . import structure as S
from .recursion_gpu import gemm as RG_gemm, _permute, _take
from .. import mpf
from ..rns import dev


def _poch(a, n):
    r = mp.mpf(1)
    for k in range(n):
        r *= a + k
    return r


def _factorial_power(x, n):
    return 1 / _poch(x + 1, -n) if n < 0 else _poch(x + 1 - n, n)


def _binom(k, j):
    """blocks_3d's binomial_coefficient(int k, int j) (generalised to negative arguments), exact"""
    def poch(a, n):
        r = 1
        for q in range(n):
            r *= a + q
        return r
    if k < j:
        return 0 if j < 0 else mp.mpf(poch(k - j + 1, j)) / math.factorial(j)
    return mp.mpf(poch(j + 1, k - j)) / math.factorial(k - j)


def stirling(O):
    S1 = [[mp.mpf(0)] * (O + 1) for _ in range(O + 1)]
    S1[0][0] = mp.mpf(1)
    for n in range(O):
        for k in range(n + 1):
            S1[n + 1][k + 1] = -n * S1[n][k + 1] + S1[n][k]
    rc = 3 - 2 * mp.sqrt(2)
    return mp.matrix([[rc ** -n * S1[n][k] for k in range(O + 1)] for n in range(O + 1)])


def rho_z(O):
    z = mp.matrix(O + 1, O + 1)
    s2 = mp.sqrt(2)
    for p in range(O + 1):
        for n in range(p + 1):
            acc = mp.mpf(0)
            for k in range(n + 1):
                acc += (_binom(n, k) * mp.mpf(0.5) ** n * (-1) ** k * (4 * s2) ** (n - k) * _binom(-2 * n, p - k - n)
                        * (4 - 2 * s2) ** (-n - p + k))
            z[p, n] = mp.factorial(p) / mp.factorial(n) * acc
    return mp.inverse(z)


def power_prefactor(O, D):
    P = mp.matrix(O + 1, O + 1)
    for n in range(O + 1):
        for k in range(n + 1):
            P[n, k] = _factorial_power(-D, n - k) * mp.mpf(2) ** (n - k + D) * _binom(n, k)
    return P


def one_minus_power_prefactor(O):
    P = mp.matrix(O + 1, O + 1)
    h = mp.mpf(0.5)
    for n in range(O + 1):
        for k in range(n + 1):
            P[n, k] = (-1) ** (n - k) * _factorial_power(-h, n - k) * mp.mpf(2) ** (n - k + h) * _binom(n, k)
    return P


def _obj(A):
    return np.array([[A[i, j] for j in range(A.cols)] for i in range(A.rows)], dtype=object)


class Output:
    """write_output for one coordinate ("xt" | "xt_radial"), Lambda, parity (= (1 - four_pt_sign)/2) and
    Delta_1+2 (decimal string).  Inputs: numerators of the lambda orders n = parity, parity+2, ... <= Lambda
    (xt) or n = parity only (xt_radial), derivatives e <= Lambda - n."""

    def __init__(self, coordinate, Lam, parity, delta_1_plus_2, L):
        self.coord, self.Lam, self.parity, self.L = coordinate, Lam, parity, L
        mp.mp.prec = 32 * L + 128
        D = mp.mpf(delta_1_plus_2)
        p = parity
        if coordinate == "xt_radial":
            O = Lam - p
            pre = power_prefactor(O, D) if p == 0 else one_minus_power_prefactor(O) * power_prefactor(O, D + 1)
            T = pre * rho_z(O) * stirling(O)
            self.inputs = [(0, e) for e in range(O + 1)]
            self.outputs = [(i, 0) for i in range(O + 1)]
            W = _obj(T)
        elif coordinate == "xt":
            W = self._xt(Lam, p, D)
        else:
            raise ValueError(coordinate)
        self.nL = (Lam - p) // 2 + 1                                   # lambda orders of a task
        self.W = mpf.from_mp(list(W.reshape(-1)), L, W.shape)

    def _xt(self, O, p, D):
        sig = -1 if p % 2 else 1
        M = rho_z(O) * stirling(O)
        if D != 0:
            M = power_prefactor(O, D / 2) * M
        M = _obj(M)
        self.inputs = [(idx, e) for idx in range((O - p) // 2 + 1) for e in range(O - (p + 2 * idx) + 1)]
        self.outputs = [(n, b) for n in range(O - p + 1) for b in range((O - n - p) // 2 + 1)]
        K = len(self.inputs)
        # Z_u(i, j) for i + j <= O, by antidiagonal s = n + e of the input
        Z = np.zeros((O + 1, O + 1, K), dtype=object)
        Z[:] = mp.mpf(0)
        for s in range(O + 1):
            us = [u for u, (idx, e) in enumerate(self.inputs) if p + 2 * idx + e == s]
            if not us:
                continue
            f = np.zeros((s + 1, len(us)), dtype=object)               # F_u(k, s-k), k = 0..s (full, sigma-symmetric)
            f[:] = mp.mpf(0)
            for c, u in enumerate(us):
                e = self.inputs[u][1]
                for k in range(s + 1):
                    i, j = (k, s - k) if k >= s - k else (s - k, k)
                    if j > min(i, O - i):
                        continue
                    acc = 0
                    for a in range(max(0, e - j), min(i, e) + 1):
                        b = e - a
                        acc += (-1) ** (j - b) * math.comb(i, a) * math.comb(j, b)
                    v = mp.mpf(acc) * mp.mpf(2) ** -s
                    f[k, c] = v if k >= s - k else sig * v
            # Z(i, j) = sum_k M(i, k) M(j, s-k) f(k)
            for i in range(O + 1):
                ks = np.arange(max(0, s - (O - i)), min(i, s) + 1)
                if not len(ks):
                    continue
                for j in range(max(0, s - i), O - i + 1):
                    kk = ks[(s - ks) <= j]
                    if not len(kk):
                        continue
                    coef = np.array([M[i, k] * M[j, s - k] for k in kk], dtype=object)
                    Z[i, j, us] += coef.dot(f[kk])
        W = np.zeros((len(self.outputs), K), dtype=object)
        W[:] = mp.mpf(0)
        for o, (n, b) in enumerate(self.outputs):
            m = p + 2 * b
            fac = mp.factorial((m - p) // 2) / mp.factorial(m)
            for jj in range(n + m + 1):                                # Z(n+m-jj, jj)
                c = sum((-1) ** l * math.comb(n, jj - l) * math.comb(m, l)
                        for l in range(max(0, jj - n), min(m, jj) + 1))
                if c:
                    W[o] += (fac * c) * Z[n + m - jj, jj]
        return W

    def apply(self, num, nL=None):
        """num: MPF (Jo, Dout, R+1, 2, S2, nq * nL) from Derivs (rhs = task q major, lambda order idx minor,
        nL lambda orders per task: all of them for xt, >= 1 for xt_radial, which reads idx 0;
        R >= Lambda - parity).  Returns MPF (Jo, 2, S2, nq, outputs, Dout)."""
        Jo, Dout, R1, _, S2, B = num.shape
        nL = nL or self.nL
        nq = B // nL
        X = _permute(num.reshape(Jo, Dout, R1, 2, S2, nq, nL), (2, 6, 0, 3, 4, 5, 1))        # (e, idx, jo, p, s, q, d)
        uidx = torch.tensor([e * nL + i for i, e in self.inputs], device=dev)
        X = _take(X.reshape(R1 * nL, -1), uidx)                                              # (inputs, cols)
        out = RG_gemm(self.W, X).reshape(len(self.outputs), Jo, 2, S2, nq, Dout)
        return _permute(out, (1, 2, 3, 4, 0, 5))


def apply_per_order(out, nums):
    """Output.apply for numerators given per lambda order (each from its own Derivs with R = Lambda - n):
    nums[idx]: MPF (Jo, Dout_n, R_n+1, 2, S2, nq), idx over the task's lambda orders (all of them for xt,
    the first for xt_radial).  Returns MPF (Jo, 2, S2, nq, outputs, max Dout)."""
    Dmax = max(x.shape[1] for x in nums)
    rows = []
    for idx, x in enumerate(nums):
        e_needed = [e for i, e in out.inputs if i == idx]
        if not e_needed:
            continue
        Jo, D, R1, _, S2, nq = x.shape
        X = _permute(x, (2, 0, 3, 4, 5, 1))                                   # (e, jo, p, s, q, d)
        if D < Dmax:
            X = mpf.MPF(torch.cat([X.m, torch.zeros((X.L, R1, Jo, 2, S2, nq, Dmax - D), dtype=X.m.dtype, device=dev)], -1),
                        torch.cat([X.e, torch.full((R1, Jo, 2, S2, nq, Dmax - D), mpf.EZ, dtype=X.e.dtype, device=dev)], -1))
        rows.append(_take(X.reshape(R1, -1), torch.tensor(e_needed, device=dev)))
    X = mpf.MPF(torch.cat([r.m for r in rows], 1), torch.cat([r.e for r in rows], 0))
    Jo, _, _, _, S2, nq = nums[0].shape
    res = RG_gemm(out.W, X).reshape(len(out.outputs), Jo, 2, S2, nq, Dmax)
    return _permute(res, (1, 2, 3, 4, 0, 5))


def _fmt(x, digits):
    if x == 0:
        return "0"
    return mp.nstr(x, digits, strip_zeros=True, min_fixed=-mp.inf, max_fixed=mp.inf)


def write_json(path, outs, vals, tj, tj12, tj43, two_js, q4, sign, delta_12, delta_43, delta_1_plus_2, order, Lam,
               kept_pole_order, pp, bits, two_js_int, spin_index, task=0):
    """one blocks_3d-style JSON file (one spin, one task) from Output.apply results already on the host:
    outs: list of Output (coordinates, in blocks_3d's order), vals: matching numpy object arrays of mpf
    (Jo, 2, S2, nq, outputs, Dout)"""
    digits = int(bits * math.log10(2))
    eps = mp.mpf(0.5) * mp.mpf(2) ** -int(0.9 * math.log2(10) * digits)
    ub = tj / 2 + (0.5 if tj <= 1 else 1)
    poles = sorted({mp.mpf(S.delta_2(P, tj)) / 2 - mp.mpf(ub) for P in S.poles(tj, tj12, tj43, kept_pole_order)})
    h = {"j_external": [x / 2 for x in two_js], "j_internal": tj / 2, "delta_minus_x": ub, "j_12": tj12 / 2,
         "j_43": tj43 / 2, "delta_12": delta_12, "delta_43": delta_43, "delta_1_plus_2": delta_1_plus_2,
         "four_pt_struct": [x / 2 for x in q4], "four_pt_sign": sign, "order": order, "lambda": Lam,
         "kept_pole_order": kept_pole_order, "coordinates": [o.coord for o in outs],
         "pole_list_x": [float(x) for x in poles], "precision_input": bits,
         "j_internal_input": [x / 2 for x in two_js_int], "generator": "sdpvibe.blocks (GPU)"}
    Sm = int(round(vals[0].shape[2] ** 0.5))
    derivs = []
    for p in range(2):
        nl, nr = len(S.so3(pp[p][0], tj12, tj)), len(S.so3(pp[p][1], tj43, tj))
        dp = []
        for a in range(nl):
            da = []
            for b in range(nr):
                db = []
                for o, v in zip(outs, vals):
                    rows = {}
                    for k, (n, bb) in enumerate(o.outputs):
                        poly = list(v[spin_index, p, a * Sm + b, task, k])
                        size, mx = len(poly), mp.mpf(0)                 # blocks_3d's trailing truncation
                        for d in range(len(poly) - 1, -1, -1):
                            c = abs(poly[d])
                            if mx < eps * c:
                                size = d + 1
                            mx = max(mx, c)
                        if all(c == 0 for c in poly):
                            size = 1
                        rows.setdefault(n, []).append([_fmt(c, digits) for c in poly[:size]])
                    db.append([rows[n] for n in sorted(rows)])
                da.append(db)
            dp.append(da)
        derivs.append(dp)
    h["derivs"] = derivs
    with open(path, "w") as f:
        json.dump(h, f)
