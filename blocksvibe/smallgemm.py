"""Batched small multiprecision GEMM on CUDA cores: C[t] = A[t] B[t] for many tiny matrices (inner
dimension K of a few units, as in the pole step of the blocks recursion, K <= 5).  Same semantics as
mpf.gemm (A aligned per row, B per column, exact products, one rounding per entry) but with schoolbook
limb products and exact accumulation in 2L + 2 limbs instead of RNS + CRT, whose fixed per-entry cost
dominates at such small K.
"""
import cupy as cp
import numpy as np
import torch

from .. import mpf
from ..rns import dev

_SRC = r"""
typedef unsigned int u32; typedef unsigned long long u64; typedef long long i64;
#define L __L__
#define LW (2 * L + 2)
extern "C" __global__ void smallgemm(const int* A, const int* B, int* O, i64 T, i64 M, i64 K, i64 N) {
    i64 idx = blockIdx.x * (i64)blockDim.x + threadIdx.x;
    i64 tot = T * M * N; if (idx >= tot) return;
    i64 j = idx % N, i = (idx / N) % M, t = idx / (N * M);
    i64 pa = T * M * K, pb = T * K * N;              // limb-plane strides
    u32 acc[LW];
    for (int k = 0; k < LW; k++) acc[k] = 0;
    for (i64 k = 0; k < K; k++) {
        u32 a[L], b[L];
        const int* ap = A + (t * M + i) * K + k;
        const int* bp = B + (t * K + k) * N + j;
        for (int l = 0; l < L; l++) { a[l] = (u32)ap[l * pa]; b[l] = (u32)bp[l * pb]; }
        bool na = a[L - 1] >> 31, nb = b[L - 1] >> 31;
        if (na) { u64 c = 1; for (int l = 0; l < L; l++) { u64 v = (u64)(~a[l]) + c; a[l] = (u32)v; c = v >> 32; } }
        if (nb) { u64 c = 1; for (int l = 0; l < L; l++) { u64 v = (u64)(~b[l]) + c; b[l] = (u32)v; c = v >> 32; } }
        // acc +-= a b row by row (mpn_addmul_1 / mpn_submul_1), no separate product buffer
        #pragma unroll
        for (int x = 0; x < L; x++) {
            u32 ax = a[x];
            u64 c = 0;
            if (na == nb) {
                #pragma unroll
                for (int y = 0; y < L; y++) { u64 v = (u64)ax * b[y] + acc[x + y] + c; acc[x + y] = (u32)v; c = v >> 32; }
                #pragma unroll
                for (int l = x + L; l < LW; l++) { u64 v = (u64)acc[l] + c; acc[l] = (u32)v; c = v >> 32; }
            } else {
                #pragma unroll
                for (int y = 0; y < L; y++) {
                    u64 pr = (u64)ax * b[y] + c;          // subtract lo(pr), carry hi(pr) + borrow
                    u32 lo = (u32)pr; u32 cur = acc[x + y];
                    acc[x + y] = cur - lo; c = (pr >> 32) + (cur < lo);
                }
                #pragma unroll
                for (int l = x + L; l < LW; l++) { u32 cur = acc[l]; acc[l] = cur - (u32)c; c = (cur < (u32)c); }
            }
        }
    }
    for (int l = 0; l < LW; l++) O[l * tot + idx] = (int)acc[l];
}
"""
_mods = {}


def _kernel(L):
    if L not in _mods:
        _mods[L] = cp.RawModule(code=_SRC.replace("__L__", str(L)), options=("-std=c++17",)).get_function("smallgemm")
    return _mods[L]


def smallgemm(A, B):
    """A: MPF (T, M, K), B: MPF (T, K, N) -> MPF (T, M, N)"""
    T, M, K = A.shape
    N = B.shape[2]
    L = A.L
    Am, rA = mpf._align_rows(A)                                    # (L, T, M, K), (T, M)
    Bm, cB = mpf._align_rows(B.T)                                  # (L, T, N, K), (T, N)
    Bm = Bm.permute(0, 1, 3, 2).contiguous()                       # (L, T, K, N)
    tot = T * M * N
    W = torch.empty((2 * L + 2, tot), dtype=torch.int32, device=dev)
    th = 128
    _kernel(L)(((tot + th - 1) // th,), (th,),
               (cp.from_dlpack(Am.contiguous()), cp.from_dlpack(Bm), cp.from_dlpack(W),
                np.int64(T), np.int64(M), np.int64(K), np.int64(N)))
    ein = (rA[:, :, None] + cB[:, None, :]).reshape(-1)
    return mpf.normalize(W, ein.contiguous(), L).reshape(T, M, N)
