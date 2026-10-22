"""Cauchy step alone: CUDA-core kernel (2/integer coefficients) vs RNS GEMM, per output entry.
   python tools/bench_cauchy.py [L]"""
import sys, time
import torch
from sdpvibe import mpf
from blocksvibe.cauchy import cauchy
from blocksvibe.recursion_gpu import bgemm, random_mpf
from sdpvibe.rns import dev

L = int(sys.argv[1]) if len(sys.argv) > 1 else 29
for nb, M, K, N in ((4, 128, 32, 1024), (4, 128, 128, 1024), (4, 128, 384, 1024), (2, 256, 256, 1024)):
    X = random_mpf((nb, K, N), L)
    D = torch.randint(1, 600, (nb, M, K), device=dev, dtype=torch.int32) * (torch.randint(0, 2, (nb, M, K), device=dev) * 2 - 1).int()
    C = mpf.from_mp([0], L, (1,))
    import mpmath as mp
    mp.mp.prec = 32 * L + 64
    Dc = D.cpu().flatten().tolist()
    C = mpf.from_mp([mp.mpf(2) / d for d in Dc], L, (nb, M, K))
    out = {}
    for name, f in (("cuda", lambda: cauchy(X, D)), ("rns", lambda: bgemm(C, X))):
        f(); torch.cuda.synchronize(); t = time.time()
        for _ in range(3): r = f()
        torch.cuda.synchronize(); dt = (time.time() - t) / 3; out[name] = r
        print(f"{name:4s} nb={nb} M={M} K={K:3d} N={N}: {dt*1e3:7.1f} ms {dt/(nb*M*N)*1e9:6.1f} ns/entry", flush=True)
    d = mpf.add(out["cuda"], out["rns"], -1)
    rel = (d.e.float() + 32 * L - (out["rns"].e.float() + 32 * L)).max().item()
    print(f"     max |cuda - rns| / |entry| ~ 2^{rel:.0f}")
