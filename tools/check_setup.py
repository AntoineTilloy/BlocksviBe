"""b3d_setup (csrc/b3d_setup.cxx) vs the dumps of an instrumented blocks_3d run (tools/b3d_dump.patch,
B3D_DUMP=<prefix>) of the same task: M matrices and h_inf must agree to the dump's printed digits.
   python tools/check_setup.py <dump prefix> <setup file> L"""
import sys
import mpmath as mp
import numpy as np
from blocksvibe import dump, setup, recursion_gpu as RG

pre, path, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
mp.mp.prec = 32 * L + 64
Mref = dump.read_M(pre + "_M.txt")
Smax = max(max(len(m), len(m[0]) if m else 0) for m in Mref.values())
M, tasks = setup.read(path, L)
assert len(tasks) == 1
t = tasks[0]
vals = M.padded(Smax).to_mp()


def rel(a, b):
    a, b = mp.mpf(a), mp.mpf(b)
    return abs(a - b) / max(abs(b), mp.mpf(2) ** -10000)


worst = mp.mpf(0); n = 0
assert set(Mref) == set(M.index), (len(Mref), len(M.index))
for key, mat in Mref.items():
    i = M.index[key]
    for a, row in enumerate(mat):
        for b, x in enumerate(row):
            if mp.mpf(x) != 0:
                worst = max(worst, rel(vals[i, a, b], x)); n += 1
print(f"M: {len(Mref)} matrices, {n} nonzero entries, max relative difference 2^{mp.nstr(mp.log(worst, 2), 4) if worst else '-inf'}")

for nn, (idx, v) in sorted(t.h.items()):
    ppref, H, _ = dump.read_n(f"{pre}_n{nn}.txt")
    assert ppref == t.pp, (ppref, t.pp)
    hv = v.to_mp()
    assert len(H) == len(idx), (len(H), len(idx))
    worst = mp.mpf(0); big = max(abs(mp.mpf(x)) for x in H.values())
    for k, key in enumerate(map(tuple, idx)):
        worst = max(worst, abs(hv[k] - mp.mpf(H[key])) / big)
    print(f"h_inf n={nn}: {len(H)} entries, max difference relative to the largest 2^{mp.nstr(mp.log(worst, 2), 4) if worst else '-inf'}")
