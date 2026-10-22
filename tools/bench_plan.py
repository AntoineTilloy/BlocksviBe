"""Build time and memory of the vectorised recursion plan (recursion_gpu.Plan, slot reuse) at production
shapes.   python tools/bench_plan.py two_j12 two_j43 j_max order"""
import resource, sys, time
import torch
from blocksvibe import recursion_gpu as RG

tj12, tj43, jmax, order = map(int, sys.argv[1:5])
t0 = time.time()
plan = RG.Plan(tj12, tj43, 2 * jmax, order, ((0, 0), (1, 1)), reuse=True)
t1 = time.time()
nbytes = sum(v.numel() * v.element_size() for lev in plan.levels if lev for v in lev.values() if torch.is_tensor(v))
tasks = sum(len(lev["t_slot"]) for lev in plan.levels if lev)
dense = sum(lev["nb"] * lev["Mr"] * lev["Kc"] for lev in plan.levels if lev)
print(f"2j12={tj12} 2j43={tj43} j_max={jmax} order={order}: plan {t1 - t0:.1f} s, peak RSS "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB, index tensors {nbytes / 1e9:.2f} GB; "
      f"{tasks:.3g} tasks, {2 * plan.nslots:.3g} logical / {plan.nphys:.3g} physical slots, {len(plan.Mkeys)} M keys, "
      f"dense Cauchy tables would be {dense:.3g} entries ({dense * 12 / 1e9:.1f} GB as cdiff+cidx)")
if len(sys.argv) > 6:                                   # also the derivative-stage constants: R L
    from blocksvibe import derivatives as DV
    R, L = int(sys.argv[5]), int(sys.argv[6])
    t2 = time.time()
    der = DV.Derivs(plan, R, L)
    torch.cuda.synchronize()
    t3 = time.time()
    gb = sum(x.m.numel() * 4 + x.e.numel() * 8 for x in (der.Anum, der.Ttab)) / 1e9
    print(f"Derivs (R={R}, L={L}): {t3 - t2:.1f} s; spins {der.Jo}, poles {der.NP}; constants on GPU {gb:.2f} GB "
          f"(numerator matrix {der.Anum.shape})")
