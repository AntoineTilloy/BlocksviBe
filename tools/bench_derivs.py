"""Timing of the GPU post-recursion stages (derivative moments accumulated during the recursion, then
residues / r^Delta / Delta -> x / numerators, sdpvibe/blocks/derivatives.py) on a synthetic operator of the
shape of a real task (random M and h_inf: the cost does not depend on the values), recycled slot buffer.
Default: no recursion, the moments are taken level by level from a compact random slot buffer (fits a
6 GB laptop GPU at order 60); OFFLOAD=1 moves the moments to the host before the final GEMMs; RECURSION=1 runs the real recursion with the moments as its hook.
SPINS=k: only the k largest output spins (the most poles) -- per-spin costs of large shapes on a small GPU.
   python tools/blocks/bench_derivs.py two_j12 two_j43 j_max order R L B [spin chunks to try ...]"""
import os, sys, time
import torch
from sdpvibe.blocks import recursion_gpu as RG, derivatives as DV

tj12, tj43, jmax, order, R, L, B = map(int, sys.argv[1:8])
chunks = [int(c) for c in sys.argv[8:]] or [None]
t0 = time.time()
plan = RG.Plan(tj12, tj43, 2 * jmax, order, ((0, 0), (1, 1)), reuse=True)
t1 = time.time()
full = os.environ.get("RECURSION") == "1"
rec = RG.Recursion(plan, None, L) if full else None
nsp = int(os.environ.get("SPINS", "0"))
der = DV.Derivs(plan, R, L, two_js=list(range(2 * jmax - 2 * (nsp - 1), 2 * jmax + 1, 2)) if nsp else None)
t2 = time.time()
Sm = plan.Smax
print(f"plan {t1 - t0:.1f} s (physical slots {plan.nphys} of {2 * plan.nslots}), constants {t2 - t1:.1f} s; "
      f"spins {der.Jo}, poles {der.NP}, R {R}, L {L} ({32 * L - 2} bits), B {B}", flush=True)
h = RG.random_mpf((2, order + 1, plan.Jh, Sm * Sm, B), L)
for chunk in chunks:
    try:
        for rep in range(2):                                 # first pass compiles the kernels
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            der.start(B, window=int(os.environ.get("WINDOW", 8)))
            prof = {"cauchy": 0.0, "pole": 0.0}
            if full:
                rec.run(h, prof=prof, hook=der.accumulate)
            else:
                # the gather only touches the slots born at the level: a compact buffer (slot indices mod
                # its size, still scattered) does the same work in a fraction of the memory
                cap = max(lev[0].numel() for lev in der.levels if lev is not None)
                res = RG.random_mpf((cap, Sm * Sm, B), L)
                levels = der.levels
                der.levels = [None if lev is None else (lev[0] % cap,) + lev[1:] for lev in levels]
                torch.cuda.synchronize(); ta = time.time()
                for m in range(order):
                    der.accumulate(m, res)
                torch.cuda.synchronize(); prof["moments"] = time.time() - ta
                der.levels = levels
                del res
            tb = time.time()
            if os.environ.get("OFFLOAD") == "1":
                der.offload()
                torch.cuda.synchronize()
            tb2 = time.time()
            n = 0
            fprof = {}
            for j0, num in der.finish_iter(h, chunk, prof=fprof):
                n += num.e.numel()
                del num
            torch.cuda.synchronize()
            tc = time.time()
    except torch.OutOfMemoryError:
        print(f"spin chunk {chunk}: out of GPU memory", flush=True)
        der.mom = None
        torch.cuda.empty_cache()
        continue
    print(f"spin chunk {chunk}, per right-hand side: recursion {(prof['cauchy'] + prof['pole']) / B:.3f} s (cauchy "
          f"{prof['cauchy'] / B:.3f}, pole {prof['pole'] / B:.3f}); post-recursion {(prof['moments'] + tc - tb2) / B:.3f} s "
          f"(moments during the recursion {prof['moments'] / B:.3f}, finish {(tc - tb2) / B:.3f}; offload of the "
          f"moments to the host {(tb2 - tb) / B:.3f}, not counted); {n / B:.3g} numerator "
          f"coefficients; peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n  finish: "
          + ", ".join(f"{k} {v / B:.3f}" for k, v in fprof.items() if k != "_t"), flush=True)
