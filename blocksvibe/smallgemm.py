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
        u32 p[2 * L];
        for (int l = 0; l < 2 * L; l++) p[l] = 0;
        for (int x = 0; x < L; x++) {
            u32 ax = a[x]; if (!ax) continue;
            u64 c = 0;
            for (int y = 0; y < L; y++) { u64 v = (u64)ax * b[y] + p[x + y] + c; p[x + y] = (u32)v; c = v >> 32; }
            p[x + L] = (u32)c;
        }
        if (na == nb) { u64 c = 0; for (int l = 0; l < 2 * L; l++) { u64 v = (u64)acc[l] + p[l] + c; acc[l] = (u32)v; c = v >> 32; }
                        for (int l = 2 * L; l < LW; l++) { u64 v = (u64)acc[l] + c; acc[l] = (u32)v; c = v >> 32; } }
        else { u64 bw = 0; for (int l = 0; l < 2 * L; l++) { u64 v = (u64)acc[l] - p[l] - bw; acc[l] = (u32)v; bw = (v >> 32) & 1; }
               for (int l = 2 * L; l < LW; l++) { u64 v = (u64)acc[l] - bw; acc[l] = (u32)v; bw = (v >> 32) & 1; } }
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
