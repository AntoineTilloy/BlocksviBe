"""Batched GPU blocks recursion (blocksvibe/recursion_gpu.py) vs the exact CPU port of blocks_3d's
fill_residues (recursion_ref.py) on a random operator and random initial data (several right-hand sides).
(The CPU port itself is checked against an instrumented blocks_3d with tools/check_ref.py.)
   python tests/test_blocks_recursion.py"""
import random
import mpmath as mp
import torch
from sdpvibe import mpf
from blocksvibe import recursion_gpu as RG, recursion_ref, structure as S


def test(tj12=4, tj43=2, tjmax=6, order=7, L=10, B=3, seed=1, **opts):
    rng = random.Random(seed)
    pp = ((0, 1), (1, 0))
    plan = RG.Plan(tj12, tj43, tjmax, order, pp)
    mp.mp.prec = 32 * L + 64
    rstr = lambda: mp.nstr(mp.mpf(rng.uniform(-2, 2)) + mp.mpf(rng.random()) * mp.mpf(2) ** -150, 120)
    M = {}
    for (P, d, tjNN, ppp, tj) in plan.Mkeys:
        rows = len(S.so3((ppp + S.parity(P)) % 2, tjNN, tj))
        cols = len(S.so3(ppp, tjNN, S.pole_two_j(P, tj)))
        M[(P, d, tjNN, ppp, tj)] = [[rstr() for _ in range(cols)] for _ in range(rows)]
    Sm, J = plan.Smax, plan.Jh
    H = {}
    strs = ["0"] * (2 * order * J * Sm * Sm * B)
    for s in range(2):
        for m in range(order):
            for j in range(J):
                tj = tj12 % 2 + 2 * j
                for a in range(len(S.so3(pp[s][0], tj12, tj))):
                    for b in range(len(S.so3(pp[s][1], tj43, tj))):
                        for k in range(B):
                            v = rstr()
                            H[(k, s, m, j, a, b)] = v
                            strs[((((s * order + m) * J + j) * Sm + a) * Sm + b) * B + k] = v
    res = RG.Recursion(plan, M, L, **opts).run(mpf.from_strings(strs, L, (2, order, J, Sm * Sm, B))).to_mp()
    num = lambda x: mp.mpf(x.numerator) / x.denominator if hasattr(x, "numerator") else mp.mpf(x)
    Mn = {k: [[mp.mpf(x) for x in row] for row in v] for k, v in M.items()}
    worst = 0
    for k in range(B):
        hk = {key[1:]: mp.mpf(v) for key, v in H.items() if key[0] == k}
        ref = recursion_ref.recursion(hk, Mn, tj12, tj43, tjmax, order, pp, num)
        blk = {}
        for (p, m, j, q, a, b), v in ref.items():
            blk[(p, m, j, q)] = max(blk.get((p, m, j, q), 0), abs(v))
        for (p, m, j, q, a, b), v in ref.items():
            g = res[plan.phys_slot(p * plan.nslots + plan.offset[(m, j)] + q), a * Sm + b, k]
            if blk[(p, m, j, q)]:
                worst = max(worst, abs(g - v) / blk[(p, m, j, q)])
    bits = -float(mp.log(worst, 2)) if worst else float("inf")
    print(f"{opts or 'default'}: {len(ref)} residues x {B} rhs: max error relative to the (pole, spin) block: 2^-{bits:.1f}")
    assert bits > 32 * L - 64, bits


if __name__ == "__main__":
    for opts in ({"cauchy": "auto", "pole": "cuda", "kcut": 3}, {"cauchy": "cuda", "pole": "cuda"}, {"cauchy": "cuda", "pole": "gemm"},
                 {"cauchy": "rns", "pole": "elementwise"}):
        test(**opts)
        test(tj12=8, tj43=8, tjmax=4, order=6, B=2, seed=2, **opts)
