"""Full GPU block chain on a synthetic operator of a real task's shape (random M and h_inf: the cost does not
depend on the values): for each lambda order n of one task (four-point structure), recursion + derivative
moments + numerators (Derivs with R = Lambda - n), numerators staged per spin chunk on disk, then write_output
(xt) per spin chunk over all n.  Per right-hand side = per (task, n).
   python tools/blocks/bench_chain.py two_j12 two_j43 j_max order Lambda L spin_chunk stage_dir
   env: OFFLOAD=1 (moments to the host before the final GEMMs), TASK_CHUNK (pole-step tasks per launch),
        ORDERS=n1,n2,... (time only these lambda orders, no staging / write_output: bench_output.py)"""
import os, shutil, sys, time
import numpy as np
import torch
from sdpvibe import mpf
from sdpvibe.blocks import recursion_gpu as RG, derivatives as DV, output as OUT

tj12, tj43, jmax, order, Lam, L, chunk = map(int, sys.argv[1:8])
stage = sys.argv[8]
shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
p0 = 0
ns = list(range(p0, Lam + 1, 2))
only = [int(x) for x in os.environ["ORDERS"].split(",")] if os.environ.get("ORDERS") else None
run_ns = only or ns
T = {}
def tick(k, t0):
    torch.cuda.synchronize(); T[k] = T.get(k, 0.0) + time.time() - t0; return time.time()

t = time.time()
plan = RG.Plan(tj12, tj43, 2 * jmax, order, ((0, 0), (1, 1)), reuse=True); t = tick("setup: plan", t)
rec = RG.Recursion(plan, None, L, task_chunk=int(os.environ.get("TASK_CHUNK", 8192))); t = tick("setup: recursion constants", t)
der0 = DV.Derivs(plan, Lam - p0, L)
ders = {n: der0.for_order(Lam - n) for n in run_ns}; t = tick("setup: derivative constants", t)
out = None if only else OUT.Output("xt", Lam, p0, "6", L)
t = tick("setup: write_output map", t)
print(f"2j12={tj12} 2j43={tj43} j_max={jmax} order={order} Lambda={Lam} L={L} ({32 * L - 2} bits): physical slots "
      f"{plan.nphys}, spins {der0.Jo}, poles {der0.NP}, write_output map {tuple(out.W.shape) if out else '-'}; "
      + ", ".join(f"{k} {v:.1f} s" for k, v in T.items()), flush=True)
Sm = plan.Smax
h = RG.random_mpf((2, order + 1, plan.Jh, Sm * Sm, 1), L)
peak = 0
for rep in range(2):                                   # pass 0: compile the kernels (Triton specialises per GEMM shape)
    for n in run_ns:
        Tn = dict(T)
        der = ders[n]
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        der.start(1)
        prof = {}
        rec.run(h, prof=prof, hook=der.accumulate)
        if rep:
            for k in ("cauchy", "pole", "moments"):
                T[k] = T.get(k, 0.0) + prof[k]
        t = time.time()
        if os.environ.get("OFFLOAD") == "1":
            der.offload()
        for j0, num in der.finish_iter(h, chunk):
            torch.cuda.synchronize()
            if rep:
                T["numerators"] = T.get("numerators", 0.0) + time.time() - t
            t = time.time()
            if rep and not only:
                np.save(f"{stage}/n{n}_j{j0}_m.npy", num.m.cpu().numpy()); np.save(f"{stage}/n{n}_j{j0}_e.npy", num.e.cpu().numpy())
                T["staging (disk)"] = T.get("staging (disk)", 0.0) + time.time() - t
            del num
            t = time.time()
        if rep:
            T["moments offload"] = T.get("moments offload", 0.0)
        peak = max(peak, torch.cuda.max_memory_allocated())
        der.mom = None
        torch.cuda.empty_cache()
        if rep:
            print(f"  n={n} (R={Lam - n}): " + ", ".join(f"{k} {T.get(k, 0) - Tn.get(k, 0):.3f} s" for k in
                  ("cauchy", "pole", "moments", "numerators")) + f"; peak GPU memory so far {peak / 1e9:.2f} GB", flush=True)
if only:
    sys.exit(0)
Jo = der0.Jo
del rec, ders, der0, der                               # write_output needs none of the recursion's state
torch.cuda.empty_cache()
load = lambda j0: [(np.load(f"{stage}/n{n}_j{j0}_m.npy"), np.load(f"{stage}/n{n}_j{j0}_e.npy")) for n in ns]
spin = lambda nums, j: [mpf.MPF(torch.from_numpy(m[:, j:j + 1]).to(RG.dev), torch.from_numpy(e[j:j + 1]).to(RG.dev))
                        for m, e in nums]
OUT.apply_per_order(out, spin(load(0), 0))            # warm-up: compile the write_output GEMM shapes
nout = 0
for j0 in range(0, Jo, chunk):
    t = time.time()
    nums = load(j0)
    t = tick("staging (disk)", t)
    for j in range(nums[0][1].shape[0]):               # one spin at a time (GEMM working set)
        t = time.time()
        x = spin(nums, j)
        t = tick("staging (disk)", t)
        torch.cuda.reset_peak_memory_stats()
        res = OUT.apply_per_order(out, x)
        t = tick("write_output", t)
        nout += res.e.numel()
        peak = max(peak, torch.cuda.max_memory_allocated())
        del x, res
shutil.rmtree(stage, ignore_errors=True)
nr = len(ns)
comp = ["cauchy", "pole", "moments", "numerators", "write_output"]
print(f"per right-hand side ({nr} lambda orders, one task): " + ", ".join(f"{k} {T[k] / nr:.3f} s" for k in comp)
      + f"; total {sum(T[k] for k in comp) / nr:.3f} s (+ staging {T['staging (disk)'] / nr:.3f} s, not counted); "
      f"{nout:.3g} output coefficients per task; peak GPU memory {peak / 1e9:.2f} GB", flush=True)
