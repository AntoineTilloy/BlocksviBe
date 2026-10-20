"""Check the reference recursion (sdpvibe/blocks/recursion_ref.py) against an instrumented blocks_3d dump.
   python tools/blocks/check_ref.py <dump prefix> <n> two_j12 two_j43 two_j_max order prec_bits"""
import sys, time
import mpmath as mp
from fractions import Fraction
from sdpvibe.blocks import dump, recursion_ref

pre, n, tj12, tj43, tjmax, order, prec = sys.argv[1], *map(int, sys.argv[2:])
mp.mp.prec = prec + 64
num = lambda x: mp.mpf(x.numerator) / x.denominator if isinstance(x, Fraction) else mp.mpf(x)
M = {k: [[num(x) for x in row] for row in v] for k, v in dump.read_M(pre + "_M.txt").items()}
pp, H, R = dump.read_n(f"{pre}_n{n}.txt")
H = {k: num(v) for k, v in H.items()}
t = time.time()
res = recursion_ref.recursion(H, M, tj12, tj43, tjmax, order, pp, num)
print(f"{len(res)} residues computed in {time.time() - t:.1f} s; blocks_3d dumped {len(R)}")
worst, scale = 0, 0
for k, v in R.items():
    v = mp.mpf(v); scale = max(scale, abs(v))
    worst = max(worst, abs(res.get(k, 0) - v) / max(abs(v), mp.mpf(2) ** -prec))
missing = [k for k, v in res.items() if k not in R and v != 0]
print(f"max relative difference {mp.nstr(worst, 3)} (2^{mp.nstr(mp.log(worst, 2) if worst else -9999, 4)}), "
      f"max |residue| {mp.nstr(scale, 3)}, nonzero residues not in dump: {len(missing)}")
