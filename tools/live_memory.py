"""Memory of the batched recursion per right-hand side: all residue slots (what blocks_3d keeps) vs the
peak of residues still needed as Cauchy columns (slot (m, j, P) lives until level m + shift(P)), plus
derivative moments sum_m m^k r_c^m res for internal spins (k <= R).
   python tools/blocks/live_memory.py two_j12 two_j43 j_max order bits R"""
import sys
import numpy as np
from sdpvibe.blocks import structure as S

tj12, tj43, jmax, order, bits, R = map(int, sys.argv[1:])
tjmax = 2 * jmax
born = np.zeros(order + 2); die = np.zeros(order + 2)
total = internal = 0
for m in range(order):
    for tj in S.two_j_values(m, order, tj12, tjmax):
        for P in S.poles(tj, tj12, tj43, order - m - S.j_excess(tj, tjmax)):
            total += 1
            born[m] += 1; die[min(order, m + S.shift(P)) + 1] -= 1
live = np.cumsum(born + die)
mom = sum(len(S.poles(tj, tj12, tj43, order)) for tj in range(tj12 % 2, tjmax + 1, 2)) * (R + 1)
B = (bits + 64) / 8 * 25 * 2                      # bytes per slot: 25 structures, 2 parities, mantissa+exp
print(f"2j12={tj12} 2j43={tj43} jmax={jmax} order={order} {bits} bits: all slots {total:.3g} "
      f"({total*B/1e9:.1f} GB/rhs), peak live Cauchy slots {live.max():.3g} ({live.max()*B/1e9:.1f} GB/rhs, at m="
      f"{int(live.argmax())}), moments (R={R}) {mom*B/1e9:.1f} GB/rhs")
