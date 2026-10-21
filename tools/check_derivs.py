"""GPU recursion + post-recursion stages (sdpvibe/blocks/derivatives.py) vs the Log_r_Derivatives numerators
dumped by an instrumented blocks_3d (tools/blocks/b3d_dump.patch: <prefix>_N<n>.txt), several lambda orders
n as the batch, with the recycled slot buffer (Plan(reuse=True)).
   python tools/blocks/check_derivs.py <dump prefix> two_j12 two_j43 two_j_max order Lambda L n1 [n2 ...]"""
import sys, time
from collections import defaultdict
import mpmath as mp
import torch
from sdpvibe import mpf
from sdpvibe.blocks import dump, recursion_gpu as RG, derivatives as DV

pre = sys.argv[1]
tj12, tj43, tjmax, order, Lam, L = map(int, sys.argv[2:8])
ns = list(map(int, sys.argv[8:]))
M = dump.read_M(pre + "_M.txt")
dumps = [dump.read_n(f"{pre}_n{n}.txt") for n in ns]
pp = dumps[0][0]
B, R = len(ns), Lam - min(ns)
t0 = time.time()
plan = RG.Plan(tj12, tj43, tjmax, order, pp, reuse=True)
rec = RG.Recursion(plan, M, L)
der = DV.Derivs(plan, R, L)
t1 = time.time()
Sm, J = plan.Smax, plan.Jh
strs = ["0"] * (2 * (order + 1) * J * Sm * Sm * B)
for k, d in enumerate(dumps):
    for (s, m, j, a, b), v in d[1].items():
        strs[((((s * (order + 1) + m) * J + j) * Sm + a) * Sm + b) * B + k] = v
h = mpf.from_strings(strs, L, (2, order + 1, J, Sm * Sm, B))
torch.cuda.synchronize(); t2 = time.time()
der.start(B)
prof = {}
rec.run(h, prof=prof, hook=der.accumulate)
t3 = time.time()
num = der.finish(h, chunk=int(__import__("os").environ.get("CHUNK", "0")) or None); torch.cuda.synchronize()
t4 = time.time()
print(f"plan+constants {t1 - t0:.1f} s, h_inf {t2 - t1:.1f} s; GPU recursion+moments {t3 - t2:.2f} s "
      f"(cauchy {prof['cauchy']:.2f} pole {prof['pole']:.2f} moments {prof.get('moments', 0):.2f}), "
      f"finish {t4 - t3:.2f} s; physical slots {plan.nphys} of {2 * plan.nslots}, spins {der.Jo}, poles {der.NP}, R {R}")

mp.mp.prec = 32 * L
vals = num.to_mp()
jo_of = {tj: jo for jo, tj in enumerate(der.tjs)}
worst, cnt, coef_worst = mp.mpf(0), 0, -9999
hist = defaultdict(int)
for k, n in enumerate(ns):
    for line in open(f"{pre}_N{n}.txt"):
        w = line.split()
        tj, p, a, b, e, nc = map(int, w[1:7])
        ref = [mp.mpf(x) for x in w[7:7 + nc]]
        g = [vals[jo_of[tj], i, e, p, a * Sm + b, k] for i in range(der.Dout)]
        ref += [mp.mpf(0)] * (der.Dout - nc)
        big = max(abs(x) for x in ref)
        if big == 0:
            continue
        err = max(abs(x - y) for x, y in zip(g, ref)) / big       # normwise (relative to the largest coefficient)
        worst = max(worst, err); cnt += 1
        hist[int(mp.floor(mp.log(err, 2))) if err else -9999] += 1
        for x, y in zip(g, ref):                                    # per coefficient, where not tiny
            if abs(y) > big * mp.mpf(2) ** -64:
                coef_worst = max(coef_worst, float(mp.log(abs(x - y) / abs(y), 2)) if x != y else -9999)
print(f"{cnt} numerator polynomials (spins x parities x structures x e x {B} rhs) compared: max error relative to "
      f"the largest coefficient 2^{mp.nstr(mp.log(worst, 2), 4) if worst else '-inf'}; per coefficient (those > "
      f"2^-64 x the largest) 2^{coef_worst:.1f}")
print("histogram of log2 normwise errors:", dict(sorted(hist.items())))
