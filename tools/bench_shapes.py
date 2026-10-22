"""Throughput of sdpvibe.mpf.gemm at the shapes of the batched blocks recursion.
   python tools/bench_shapes.py [L]   (L=29 limbs -> 926-bit mantissas)"""
import sys, time
import torch
from sdpvibe import mpf
from blocksvibe.recursion_gpu import bgemm
from sdpvibe.rns import dev

L = int(sys.argv[1]) if len(sys.argv) > 1 else 29

def rnd(shape):
    n = 1
    for s in shape: n *= s
    m = torch.randint(-2**31, 2**31 - 1, (L, n), dtype=torch.int32, device=dev)
    e = torch.randint(-40, 40, (n,), dtype=torch.int64, device=dev) - 32 * L
    return mpf.normalize(m, e, L).reshape(*shape)

def bench(nb, M, K, N, label):
    A, B = rnd((nb, M, K)), rnd((nb, K, N))
    bgemm(A, B); torch.cuda.synchronize()
    t = time.time(); reps = 3
    for _ in range(reps): C = bgemm(A, B)
    torch.cuda.synchronize(); dt = (time.time() - t) / reps
    out = nb * M * N
    print(f"{label:8s} nb={nb:5d} M={M:4d} K={K:4d} N={N:5d}: {dt*1e3:8.1f} ms  {dt/out*1e9:7.1f} ns/entry  "
          f"{nb*M*K*N/dt/1e9:6.2f} G MAC/s", flush=True)

for N in (16, 64, 256):
    bench(1024 if N < 256 else 256, 25, 25, N, "pole")
