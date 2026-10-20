"""Check the batched GPU recursion against instrumented blocks_3d dumps (several n as the batch).
   python tools/blocks/check_gpu.py <dump prefix> two_j12 two_j43 two_j_max order L n1 [n2 ...]
   Without residues in the dumps (B3D_DUMP_NORES) only timings are reported."""
import sys, time
import numpy as np
import mpmath as mp
import torch
from sdpvibe import mpf
from sdpvibe.blocks import dump, recursion_gpu as RG

pre = sys.argv[1]
tj12, tj43, tjmax, order, L = map(int, sys.argv[2:7])
ns = list(map(int, sys.argv[7:]))
M = dump.read_M(pre + "_M.txt")
dumps = [dump.read_n(f"{pre}_n{n}.txt") for n in ns]
pp = dumps[0][0]
B = len(ns)
t0 = time.time()
plan = RG.Plan(tj12, tj43, tjmax, order, pp)
t1 = time.time()
rec = RG.Recursion(plan, M, L)
Sm, J = plan.Smax, plan.Jh
strs = ["0"] * (2 * order * J * Sm * Sm * B)
for k, d in enumerate(dumps):
    for (s, m, j, a, b), v in d[1].items():
        if m < order:
            strs[((((s * order + m) * J + j) * Sm + a) * Sm + b) * B + k] = v
h = mpf.from_strings(strs, L, (2, order, J, Sm * Sm, B))
torch.cuda.synchronize(); t2 = time.time()
res = rec.run(h); torch.cuda.synchronize()
t3 = time.time()
hc, rc = plan.counts()
print(f"plan {t1 - t0:.1f} s, constants+h_inf {t2 - t1:.1f} s, GPU recursion {t3 - t2:.2f} s; slots {plan.nslots}, "
      f"Smax {Sm}, B {B}, entries/rhs: Cauchy {hc:.3g} pole {rc:.3g}")
if any(d[2] for d in dumps):
    mp.mp.prec = 32 * L
    vals = res.to_mp()
    worst = big = 0; rel = []
    blockmax = {}
    for k, d in enumerate(dumps):
        for (p, m, j, q, a, b), v in d[2].items():
            blockmax[(k, p, m, j, q)] = max(blockmax.get((k, p, m, j, q), 0), abs(mp.mpf(v)))
    for k, d in enumerate(dumps):
        for (p, m, j, q, a, b), v in d[2].items():
            v = mp.mpf(v); big = max(big, abs(v))
            if abs(v) < blockmax[(k, p, m, j, q)] * mp.mpf(2) ** -200:
                continue                    # cancellation noise in blocks_3d itself (a true zero)
            g = vals[rec.slot(p, m, j, q), a * Sm + b, k]
            worst = max(worst, abs(g - v))
            if v != 0: rel.append(float(mp.log(abs(g - v) / abs(v), 2)) if g != v else -9999)
    rel = np.array(rel)
    print(f"{len(rel)} residues compared; max abs difference 2^{mp.nstr(mp.log(worst / big, 2), 4)} relative to "
          f"max |res| {mp.nstr(big, 3)}; per-entry relative error > 2^-200: {(rel > -200).sum()} entries "
          f"(entries below 2^-200 x the largest of their (pole, spin) block skipped: blocks_3d noise)")
    if __import__("os").environ.get("SHOWBAD"):
        for k, d in enumerate(dumps):
            for (p, m, j, q, a, b), v in d[2].items():
                v = mp.mpf(v); g = vals[rec.slot(p, m, j, q), a * Sm + b, k]
                bm = blockmax[(k, p, m, j, q)]
                if abs(v) >= bm * mp.mpf(2) ** -200 and v != 0 and abs(g - v) > abs(v) * mp.mpf(2) ** -200:
                    print("  ", (k, p, m, j, q, a, b), mp.nstr(v, 6), mp.nstr(g, 6), "block max", mp.nstr(bm, 3),
                          "rel err 2^%s" % mp.nstr(mp.log(abs(g - v) / abs(v), 2), 4))
