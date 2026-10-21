"""End-to-end check of the GPU block chain against blocks_3d's JSON output: recursion (M matrices and h_inf
from an instrumented blocks_3d run, tools/blocks/b3d_dump.patch) -> derivatives/numerators -> write_output
(sdpvibe/blocks/output.py) -> blocks_3d-style JSON files, compared per spin, structure, coordinate and
derivative with the files blocks_3d wrote (same task, same output template).
   python tools/blocks/check_output.py <blocks_3d out template, e.g. dir/out_{}.json> <dump prefix>
          two_j12 two_j43 two_j_max order Lambda four_pt_sign delta_1_plus_2 coords(xt,xt_radial) L bits
   (TTTT, q4 and Delta_12 = Delta_43 = 0 are read back from blocks_3d's JSON header)
   env PER_ORDER=1: one lambda order per recursion run (Derivs.for_order, output.apply_per_order)"""
import glob, json, os, sys, time
from collections import defaultdict
import mpmath as mp
import torch
from sdpvibe import mpf
from sdpvibe.blocks import dump, recursion_gpu as RG, derivatives as DV, output as OUT

tmpl, pre = sys.argv[1], sys.argv[2]
tj12, tj43, tjmax, order, Lam, sign = map(int, sys.argv[3:9])
d12, coords, L, bits = sys.argv[9], sys.argv[10].split(","), int(sys.argv[11]), int(sys.argv[12])
p0 = (1 - sign) // 2
radial = all(c.endswith("_radial") for c in coords)
ns = [p0] if radial else list(range(p0, Lam + 1, 2))
M = dump.read_M(pre + "_M.txt")
dumps = [dump.read_n(f"{pre}_n{n}.txt") for n in ns]
pp = dumps[0][0]
B, R = len(ns), Lam - p0
t0 = time.time()
plan = RG.Plan(tj12, tj43, tjmax, order, pp, reuse=True)
rec = RG.Recursion(plan, M, L)
der = DV.Derivs(plan, R, L)
outs = [OUT.Output(c, Lam, p0, d12, L) for c in coords]
t1 = time.time()
Sm, J = plan.Smax, plan.Jh
strs = ["0"] * (2 * (order + 1) * J * Sm * Sm * B)
for k, d in enumerate(dumps):
    for (s, m, j, a, b), v in d[1].items():
        strs[((((s * (order + 1) + m) * J + j) * Sm + a) * Sm + b) * B + k] = v
h = mpf.from_strings(strs, L, (2, order + 1, J, Sm * Sm, B))
torch.cuda.synchronize(); t2 = time.time()
if os.environ.get("PER_ORDER") == "1":                  # one lambda order at a time (Derivs.for_order), as bench_chain
    nums = []
    for k, n in enumerate(ns):
        dn = der.for_order(Lam - n)
        hk = mpf.MPF(h.m[..., k:k + 1].contiguous(), h.e[..., k:k + 1].contiguous())
        dn.start(1)
        rec.run(hk, hook=dn.accumulate)
        nums.append(dn.finish(hk, chunk=3))
    res = [OUT.apply_per_order(o, nums) for o in outs]
else:
    der.start(B)
    rec.run(h, hook=der.accumulate)
    num = der.finish(h)
    res = [o.apply(num, nL=B) for o in outs]
torch.cuda.synchronize(); t3 = time.time()
print(f"constants {t1 - t0:.1f} s (write_output maps {[tuple(o.W.shape) for o in outs]}), GPU chain {t3 - t2:.2f} s "
      f"for {B} lambda order(s)")

mp.mp.prec = 32 * L
vals = [r.to_mp() for r in res]
ref0 = json.load(open(tmpl.format(tjmax // 2)))
outdir = os.path.join(os.path.dirname(tmpl) or ".", "gpu")
os.makedirs(outdir, exist_ok=True)
worst, cnt, hist = mp.mpf(0), 0, defaultdict(int)
two_js_int = list(range(tj12 % 2, tjmax + 1, 2))
for jo, tj in enumerate(der.tjs):
    path = os.path.join(outdir, f"out_{tj / 2:g}.json")
    OUT.write_json(path, outs, vals, tj, tj12, tj43, [round(2 * x) for x in ref0["j_external"]],
                   [round(2 * x) for x in ref0["four_pt_struct"]], sign, ref0["delta_12"], ref0["delta_43"], d12,
                   order, Lam, ref0["kept_pole_order"], pp, bits, two_js_int, jo)
    ours = json.load(open(path))
    ref = json.load(open(tmpl.format(f"{tj / 2:g}")))
    assert [len(x) for x in ours["derivs"]] == [len(x) for x in ref["derivs"]]
    for p, (dp, rp) in enumerate(zip(ours["derivs"], ref["derivs"])):
        for da, ra in zip(dp, rp):
            for db, rb in zip(da, ra):
                for dc, rc in zip(db, rb):                         # coordinate
                    assert [len(r) for r in dc] == [len(r) for r in rc], (tj, p)
                    for d0, r0 in zip(dc, rc):
                        for g, r in zip(d0, r0):                  # one polynomial
                            g = [mp.mpf(x) for x in g]; r = [mp.mpf(x) for x in r]
                            n = max(len(g), len(r))
                            g += [mp.mpf(0)] * (n - len(g)); r += [mp.mpf(0)] * (n - len(r))
                            big = max(abs(x) for x in r)
                            if big == 0:
                                continue
                            err = max(abs(x - y) for x, y in zip(g, r)) / big
                            worst = max(worst, err); cnt += 1
                            hist[int(mp.floor(mp.log(err, 2))) if err else -9999] += 1
print(f"{coords}, sign {sign}, Delta_1+2 {d12}: {cnt} output polynomials in {len(der.tjs)} JSON files compared with "
      f"blocks_3d: max error relative to the largest coefficient 2^{mp.nstr(mp.log(worst, 2), 4) if worst else '-inf'}")
print("histogram of log2 errors:", dict(sorted(hist.items())))
