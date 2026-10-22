"""The vectorised plan builder (recursion_gpu.Plan) vs the original per-task builder (plan_ref.PlanRef):
identical index tensors, keys, slot layout and physical slot mapping, with and without slot reuse.
   python tests/test_blocks_plan.py"""
import time
import numpy as np
import torch
from blocksvibe import recursion_gpu as RG
from blocksvibe.plan_ref import PlanRef
from blocksvibe import structure as S, derivatives as DV

CASES = [(8, 8, 30, 30, ((0, 0), (1, 1))), (4, 2, 20, 24, ((0, 1), (1, 0))), (3, 1, 9, 12, ((0, 1), (1, 1))),
         (5, 5, 11, 10, ((1, 0), (0, 1))), (4, 4, 20, 16, ((0, 0), (1, 1))), (0, 0, 6, 8, ((0, 0), (1, 1))),
         (8, 4, 4, 7, ((0, 1), (1, 0)))]


def same(a, b, what):
    a = a.cpu() if torch.is_tensor(a) else torch.as_tensor(a)
    b = b.cpu() if torch.is_tensor(b) else torch.as_tensor(b)
    assert a.shape == b.shape and torch.equal(a.long(), b.long()), what


def test(tj12, tj43, tjmax, order, pp, reuse):
    t0 = time.time(); ref = PlanRef(tj12, tj43, tjmax, order, pp, reuse=reuse)
    t1 = time.time(); new = RG.Plan(tj12, tj43, tjmax, order, pp, reuse=reuse)
    t2 = time.time()
    for a in ("Smax", "Jh", "nslots", "nphys", "zero_slot", "dmax", "Mkeys", "Qkeys"):
        assert getattr(ref, a) == getattr(new, a), a
    for key, v in ref.offset.items():
        assert new.offset[key] == v
    for t in range(2 * ref.nslots + 1):
        assert ref.phys_slot(t) == new.phys_slot(t), t
    assert len(ref.levels) == len(new.levels)
    for m, (a, b) in enumerate(zip(ref.levels, new.levels)):
        assert (a is None) == (b is None), m
        if a is None:
            continue
        for f in ("m", "nb", "Mr", "Kc"):
            assert a[f] == b[f], (m, f)
        for f in ("gsrc", "gj", "col", "t_slot", "t_hrow", "t_L", "t_R", "t_q"):
            same(a[f], b[f], (m, f))
        cdiff, cidx = RG.cauchy_tables(b)
        same(a["cdiff"], cdiff, (m, "cdiff")); same(a["cidx"], cidx, (m, "cidx"))
    # derivative-moment bookkeeping (Derivs.levels) vs the original per-entry loop
    der = DV.Derivs(new, 2, 6)
    jo_of = {tj: jo for jo, tj in enumerate(der.tjs)}
    for m in range(order):
        sl, rw, nn = [], [], []
        for jidx, tj in enumerate(S.two_j_values(m, order, tj12, tjmax)):
            jo = jo_of.get(tj)
            if jo is None:
                continue
            for pidx, P in enumerate(S.poles(tj, tj12, tj43, order - m - S.j_excess(tj, tjmax))):
                i = S.position(tj, tj12, tj43, order, P)
                for p in range(2):
                    ph = ref.phys_slot(p * ref.nslots + ref.offset[(m, jidx)] + pidx)
                    if ph != ref.zero_slot:
                        sl.append(ph); rw.append((p * der.Jo + jo) * der.NP + i); nn.append(m + S.shift(P))
        lev = der.levels[m]
        assert (lev is None) == (not sl), m
        if sl:
            same(lev[0], sl, (m, "moment slots")); same(lev[1], rw, (m, "rows")); same(lev[2], nn, (m, "n"))
    print(f"{(tj12, tj43, tjmax, order, pp)} reuse={reuse}: identical ({len(new.Mkeys)} M keys, "
          f"{new.nphys} physical slots); reference {t1 - t0:.2f} s, vectorised {t2 - t1:.2f} s")


if __name__ == "__main__":
    for c in CASES:
        for reuse in (False, True):
            test(*c, reuse)
