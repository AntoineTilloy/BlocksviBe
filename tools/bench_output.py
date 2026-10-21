"""Timing of write_output on the GPU (sdpvibe/blocks/output.py, xt) for one task: composite-map construction
(once per Lambda, parity, Delta_1+2) and its application to synthetic numerators of every lambda order for a
few spins (scaled to all spins).  Shapes as Derivs would produce them (poles of spin j_max at full order).
   python tools/blocks/bench_output.py two_j12 two_j43 j_max order Lambda L spins"""
import sys, time
import torch
from sdpvibe.blocks import recursion_gpu as RG, output as OUT, structure as S

tj12, tj43, jmax, order, Lam, L, J = map(int, sys.argv[1:8])
NP = len(S.poles(2 * jmax, tj12, tj43, order))
Sm = 5 if tj12 == 8 else (min(tj12, tj43) + 1)
t0 = time.time()
out = OUT.Output("xt", Lam, 0, "6", L)
t1 = time.time()
ns = list(range(0, Lam + 1, 2))
nums = [RG.random_mpf((J, NP + Lam - n + 1, Lam - n + 1, 2, Sm * Sm, 1), L) for n in ns]
for rep in range(2):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t2 = time.time()
    for j in range(J):
        res = OUT.apply_per_order(out, [RG.mpf.MPF(x.m[:, j:j + 1].contiguous(), x.e[j:j + 1].contiguous()) for x in nums])
    torch.cuda.synchronize(); t3 = time.time()
Jo = jmax + 1
print(f"write_output xt, Lambda {Lam}, {32 * L - 2} bits: map {tuple(out.W.shape)} built in {t1 - t0:.1f} s; apply "
      f"{(t3 - t2) / J:.3f} s per spin ({NP} poles), i.e. {(t3 - t2) / J * Jo:.2f} s per task ({Jo} spins, "
      f"{len(ns)} lambda orders) = {(t3 - t2) / J * Jo / len(ns):.3f} s per right-hand side; peak GPU memory "
      f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
