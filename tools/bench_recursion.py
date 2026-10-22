"""Time the batched GPU recursion on a synthetic operator (random M, h_inf) of production shape.
   python tools/blocks/bench_recursion.py two_j12 two_j43 two_j_max order L B1 [B2 ...]
   env: CAUCHY=auto|cuda|rns  POLE=cuda|gemm|elementwise  CHUNK=<tasks per pole-step chunk>"""
import os, sys, time
import torch
from sdpvibe.blocks import recursion_gpu as RG

tj12, tj43, tjmax, order, L = map(int, sys.argv[1:6])
Bs = list(map(int, sys.argv[6:]))
t0 = time.time()
plan = RG.Plan(tj12, tj43, tjmax, order, ((0, 0), (1, 1)), reuse=os.environ.get("REUSE") == "1")
t1 = time.time()
opts = dict(cauchy=os.environ.get("CAUCHY", "auto"), pole=os.environ.get("POLE", "cuda"),
            task_chunk=int(os.environ.get("CHUNK", 8192)))
rec = RG.Recursion(plan, None, L, **opts)
torch.cuda.synchronize(); t2 = time.time()
hc, rc = plan.counts()
print(f"{opts} 2j12={tj12} 2j43={tj43} 2jmax={tjmax} order={order} L={L} ({32*L-2} bits): plan {t1-t0:.1f} s, "
      f"constants {t2-t1:.1f} s; slots {2*plan.nslots}, Smax {plan.Smax}; padded entries per rhs: "
      f"Cauchy {hc:.3g}, pole {rc:.3g}", flush=True)
for B in Bs:
    h = RG.random_mpf((2, order, plan.Jh, plan.Smax ** 2, B), L)
    torch.cuda.synchronize(); t = time.time()
    prof = {}; res = rec.run(h, prof=prof); torch.cuda.synchronize(); dt = time.time() - t
    print(f"  B={B:4d}: {dt:8.2f} s  = {dt/B:7.3f} s per rhs, {dt/B/(hc+rc)*1e9:6.1f} ns per padded entry, "
          f"peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB; Cauchy {prof['cauchy']:.2f} s, pole {prof['pole']:.2f} s", flush=True)
    del res, h; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
