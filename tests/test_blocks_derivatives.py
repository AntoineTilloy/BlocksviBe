"""GPU post-recursion stages (sdpvibe/blocks/derivatives.py, moments accumulated during a recursion with a
recycled slot buffer) vs a line-by-line mpmath port of blocks_3d's residues_to_derivatives, restore_r_delta
(polynomial division by (Delta - D_i)), shift_fraction and together_with_factors, applied to the residues of
the same recursion run without slot reuse.  Random operator and initial data, several right-hand sides.
(Checked against an instrumented blocks_3d with tools/blocks/check_derivs.py.)
   python tests/test_blocks_derivatives.py"""
from math import comb
import mpmath as mp
from sdpvibe.blocks import recursion_gpu as RG, derivatives as DV, structure as S
from sdpvibe.blocks.recursion_gpu import random_mpf


def pmul(p, q):
    out = [mp.mpf(0)] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            out[i + j] += x * y
    return out


def padd(p, q):
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)]


def pdiv_lin(p, r):
    """p(x) = (x - r) q(x) + rem (boost quotient_remainder by a monic linear divisor)"""
    q = [mp.mpf(0)] * max(len(p) - 1, 0)
    acc = mp.mpf(0)
    for k in range(len(p) - 1, 0, -1):
        acc = p[k] + r * acc
        q[k - 1] = acc
    return q, (p[0] + r * acc if p else mp.mpf(0))


def reference(h, res, tj, p, a, b, order, R, tj12, tj43):
    rc = 3 - 2 * mp.sqrt(2)
    W = [[(mp.mpf(n) ** k if (n or k) else mp.mpf(1)) * rc ** n for n in range(order + 1)] for k in range(R + 1)]
    ji = (tj - tj12 % 2) // 2
    c = [sum(W[k][m] * h(p, m, ji, a, b) for m in range(order + 1)) for k in range(R + 1)]
    poles = []
    for P in S.poles(tj, tj12, tj43, order):
        s = S.shift(P)
        v = [sum(W[k][ii + s] * res(p, ii, ji, S.position(tj, tj12, tj43, order - ii, P), a, b)
                 for ii in range(order - s + 1)) for k in range(R + 1)]
        poles.append((mp.mpf(S.delta_2(P, tj)) / 2, v))
    rdelta = lambda e, d: [mp.mpf(0)] * (e - d) + [mp.mpf(comb(e, d))] if d <= e else [mp.mpf(0)]
    ub = mp.mpf(tj) / 2 + (mp.mpf(1) / 2 if tj <= 1 else 1)
    xs = [D - ub for D, _ in poles]
    prod = [mp.mpf(1)]
    for x in xs:
        prod = pmul(prod, [-x, mp.mpf(1)])
    comp = []
    for i in range(len(xs)):
        q = [mp.mpf(1)]
        for k, x in enumerate(xs):
            if k != i:
                q = pmul(q, [-x, mp.mpf(1)])
        comp.append(q)
    out = []
    for e in range(R + 1):
        poly, resid = [mp.mpf(0)], []
        for d in range(R + 1):
            poly = padd(poly, [t * c[d] for t in rdelta(e, d)])
        for D, v in poles:
            rdp = [mp.mpf(0)]
            for d in range(R + 1):
                rdp = padd(rdp, [t * v[d] for t in rdelta(e, d)])
            q, rem = pdiv_lin(rdp, D)
            poly = padd(poly, q)
            resid.append(rem)
        shifted, power = [mp.mpf(0)], [mp.mpf(1)]                  # shift_fraction: P(x + ub)
        for t in poly:
            shifted = padd(shifted, [t * y for y in power])
            power = pmul(power, [ub, mp.mpf(1)])
        num = pmul(shifted, prod)
        for r, q in zip(resid, comp):
            num = padd(num, [r * y for y in q])
        out.append(num)
    return out


def test(tj12=4, tj43=2, tjmax=6, order=7, R=3, L=10, B=2, seed=1):
    pp = ((0, 1), (1, 0))
    plans = [RG.Plan(tj12, tj43, tjmax, order, pp, reuse=r) for r in (False, True)]
    recs = [RG.Recursion(pl, None, L) for pl in plans]
    recs[1].Mtab = recs[0].Mtab                                       # same random operator
    Sm = plans[0].Smax
    import torch
    torch.manual_seed(seed)
    h = random_mpf((2, order + 1, plans[0].Jh, Sm * Sm, B), L, spread=2)
    res = recs[0].run(h).to_mp()
    der = DV.Derivs(plans[1], R, L)
    der.start(B)
    recs[1].run(h, hook=der.accumulate)
    num = der.finish(h, chunk=2).to_mp()
    hv = h.to_mp()
    mp.mp.prec = 32 * L + 64
    worst, cnt = 0, 0
    for k in range(B):
        hf = lambda p, m, j, a, b: hv[p, m, j, a * Sm + b, k]
        rf = lambda p, m, j, q, a, b: res[recs[0].slot(p, m, j, q), a * Sm + b, k]
        for jo, tj in enumerate(der.tjs):
            for p in range(2):
                for a in range(len(S.so3(pp[p][0], tj12, tj))):
                    for b in range(len(S.so3(pp[p][1], tj43, tj))):
                        ref = reference(hf, rf, tj, p, a, b, order, R, tj12, tj43)
                        for e, poly in enumerate(ref):
                            big = max(abs(x) for x in poly)
                            g = [num[jo, i, e, p, a * Sm + b, k] for i in range(der.Dout)]
                            poly = poly + [0] * (der.Dout - len(poly))
                            assert all(x == 0 for x in g[len(poly):])
                            worst = max(worst, max(abs(x - y) for x, y in zip(g, poly)) / big)
                            cnt += 1
    bits = -float(mp.log(worst, 2)) if worst else float("inf")
    print(f"{cnt} numerators (R={R}, {B} rhs): max error relative to the largest coefficient 2^-{bits:.1f}")
    assert bits > 32 * L - 40, bits


if __name__ == "__main__":
    test()
    test(tj12=8, tj43=8, tjmax=4, order=6, R=5, B=1, seed=2)
    test(tj12=3, tj43=1, tjmax=5, order=5, R=0, B=2, seed=3)
