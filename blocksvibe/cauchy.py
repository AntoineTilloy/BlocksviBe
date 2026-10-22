"""Cauchy step of the blocks recursion on CUDA cores: H[g, r, n] = sum_i X[g, i, n] * 2 / d[g, r, i]
with small integer d (|d| < 2^31; d = 0 means "no term").  Every coefficient is 2/d, so each term is a
multiprecision division by a one-limb integer -- O(L) instead of the O(L^2) of a full product or the
per-output CRT of the RNS GEMM.  X is aligned to a common exponent per column (g, n) (normwise error,
like the GEMM), each term is truncated 32 bits below the aligned mantissa, the sum is kept exact in
L + 3 limbs and rounded once.
"""
import cupy as cp
import numpy as np
import torch

from .. import mpf
from ..rns import dev

_SRC = r"""
typedef unsigned int u32; typedef unsigned long long u64; typedef long long i64;
#define L __L__
#define LW (L + 3)
extern "C" __global__ void cauchy(const int* X, const int* D, int* O, i64 nb, i64 Mr, i64 Kc, i64 N, const u64* RINV, int dmax) {
    i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    i64 tot = nb * Mr * N; if (t >= tot) return;
    i64 n = t % N, r = (t / N) % Mr, g = t / (N * Mr);
    i64 plane = nb * Kc * N;                          // limb-plane stride of X
    u32 acc[LW];
    for (int k = 0; k < LW; k++) acc[k] = 0;
    const int* drow = D + (g * Mr + r) * Kc;
    for (i64 i = 0; i < Kc; i++) {
        int d = drow[i]; if (d == 0) continue;
        const int* x = X + (g * Kc + i) * N + n;
        u32 a[L];
        for (int k = 0; k < L; k++) a[k] = (u32)x[k * plane];
        bool neg = (a[L - 1] >> 31) != 0;
        if (neg) { u64 c = 1; for (int k = 0; k < L; k++) { u64 v = (u64)(~a[k]) + c; a[k] = (u32)v; c = v >> 32; } }
        if (d < 0) { neg = !neg; d = -d; }
        u64 ud = (u64)d;
        u64 inv = d <= dmax ? RINV[d] : 0xFFFFFFFFFFFFFFFFull / ud;   // reciprocal table: no 64-bit division
        // q = floor(|x| * 2^32 / d): limbs q[1..L] from a[0..L-1], q[0] from the appended zero limb
        u32 q[L + 1]; u64 rem = 0;
        for (int k = L; k >= 0; k--) {
            u64 cur = (rem << 32) | (k > 0 ? a[k - 1] : 0u);   // cur < d 2^32
            u64 qq = __umul64hi(cur, inv);
            u64 rr = cur - qq * ud;
            while (rr >= ud) { qq++; rr -= ud; }
            q[k] = (u32)qq; rem = rr;
        }
        if (!neg) { u64 c = 0; for (int k = 0; k < L + 1; k++) { u64 v = (u64)acc[k] + q[k] + c; acc[k] = (u32)v; c = v >> 32; }
                    for (int k = L + 1; k < LW; k++) { u64 v = (u64)acc[k] + c; acc[k] = (u32)v; c = v >> 32; } }
        else { u64 b = 0; for (int k = 0; k < L + 1; k++) { u64 v = (u64)acc[k] - q[k] - b; acc[k] = (u32)v; b = (v >> 32) & 1; }
               for (int k = L + 1; k < LW; k++) { u64 v = (u64)acc[k] - b; acc[k] = (u32)v; b = (v >> 32) & 1; } }
    }
    for (int k = 0; k < LW; k++) O[k * tot + t] = (int)acc[k];
}
"""
_mods = {}
_rinv = {}
DMAX = 4096                                           # reciprocal table size (larger |D' - D_i|: one division)


def _reciprocals():
    """floor((2^64 - 1) / d) for d = 0..DMAX (entry 0 unused), on the GPU"""
    if DMAX not in _rinv:
        v = np.array([0] + [(2 ** 64 - 1) // d for d in range(1, DMAX + 1)], dtype=np.uint64)
        _rinv[DMAX] = cp.asarray(v)
    return _rinv[DMAX]


def _kernel(L):
    if L not in _mods:
        _mods[L] = cp.RawModule(code=_SRC.replace("__L__", str(L)), options=("-std=c++17",)).get_function("cauchy")
    return _mods[L]


def cauchy(X, D):
    """X: MPF (nb, Kc, N); D: int32 tensor (nb, Mr, Kc) -> MPF (nb, Mr, N) = sum_i X[:, i] * 2 / D[:, :, i]"""
    nb, Kc, N = X.shape
    Mr = D.shape[1]
    L = X.L
    Xm, E = mpf._align_rows(X.T)                       # (L, nb, N, Kc), exps (nb, N)
    Xm = Xm.permute(0, 1, 3, 2).contiguous()           # (L, nb, Kc, N)
    tot = nb * Mr * N
    W = torch.empty((L + 3, tot), dtype=torch.int32, device=dev)
    T = 128
    _kernel(L)(((tot + T - 1) // T,), (T,),
               (cp.from_dlpack(Xm), cp.from_dlpack(D.to(torch.int32).contiguous()), cp.from_dlpack(W),
                np.int64(nb), np.int64(Mr), np.int64(Kc), np.int64(N), _reciprocals(), np.int32(DMAX)))
    ein = (E[:, None, :].expand(nb, Mr, N) - 31).reshape(-1)       # value = acc * 2^(E - 32 + 1)
    return mpf.normalize(W, ein.contiguous(), L).reshape(nb, Mr, N)
