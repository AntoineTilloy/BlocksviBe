"""Reference (CPU, mpmath) radial recursion: a line-by-line port of blocks_3d's fill_residues /
set_residue.  Used to pin down the semantics that the batched GPU version (recursion_gpu.py) must
reproduce, and checked against residues dumped by an instrumented blocks_3d.

Inputs: h_inf[(pp, m, jidx, a, b)] and M[((t,k), delta, two_jNN, ppp, two_j)] -> matrix, as numbers
(any type supporting + and *, e.g. mpmath.mpf).  Output: res[(pp, m, jidx, pole_index, a, b)].
"""
from . import structure as S


def recursion(h_inf, M, two_j12, two_j43, two_j_max, order, pseudo_parities, num):
    """num: converter from Fraction/str to the number type"""
    res = {}
    zero = num(0)
    for m in range(order):
        for jidx, tj in enumerate(S.two_j_values(m, order, two_j12, two_j_max)):
            pl = S.poles(tj, two_j12, two_j43, order - m - S.j_excess(tj, two_j_max))
            for pidx, P in enumerate(pl):
                tjp = S.pole_two_j(P, tj)
                dprime = S.delta_2(P, tj) + 2 * S.shift(P)
                prime = S.poles(tjp, two_j12, two_j43, m)
                jpex = S.j_excess(tjp, two_j_max)
                terms = [(m - S.shift(i),
                          S.position(tjp, two_j12, two_j43, order - m + S.shift(i) - jpex, i),
                          num(S.cauchy_coeff(dprime, S.delta_2(i, tjp)))) for i in prime]
                Q = num(S.q_value(P, tj))
                sign = -1 if (tjp - tj) % 4 else 1
                for p in range(2):
                    src = (p + S.parity(P)) % 2
                    LM = M.get((P, 0, two_j12, pseudo_parities[src][0], tj))
                    RM = M.get((P, 1, two_j43, pseudo_parities[src][1], tj))
                    if not LM or not LM[0] or not RM or not RM[0]:
                        continue
                    nl, nr = len(LM[0]), len(RM[0])
                    h = [[h_inf.get((src, m, tjp // 2, a, b), zero)
                          + sum((res.get((src, mm, tjp // 2, q, a, b), zero) * c for mm, q, c in terms), zero)
                          for b in range(nr)] for a in range(nl)]
                    inter = [[sum((h[a][b] * RM[b0][b] for b in range(nr)), zero) * sign
                              for b0 in range(len(RM))] for a in range(nl)]
                    for a0 in range(len(LM)):
                        for b0 in range(len(RM)):
                            res[(p, m, jidx, pidx, a0, b0)] = Q * sum(
                                (LM[a0][a] * inter[a][b0] for a in range(nl)), zero)
    return res
