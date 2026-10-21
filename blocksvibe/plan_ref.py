"""Reference plan builder of the batched blocks recursion: the first, per-task Python implementation of
recursion_gpu.Plan (dense per-level Cauchy tables cidx/cdiff, dict bookkeeping).  Only used to check the
vectorised builder."""
import torch

from . import structure as S


class PlanRef:
    """The original (pure Python, per-task) plan builder, kept as the reference that the vectorised
    recursion_gpu.Plan must reproduce exactly (tests/test_blocks_plan.py).  Slow: ~265 s at order 60."""

    def __init__(self, two_j12, two_j43, two_j_max, order, pseudo_parities, reuse=False):
        self.tj12, self.tj43, self.tjmax, self.order, self.pp = two_j12, two_j43, two_j_max, order, pseudo_parities
        self.SL = lambda p, tj: len(S.so3(pseudo_parities[p][0], two_j12, tj))
        self.SR = lambda p, tj: len(S.so3(pseudo_parities[p][1], two_j43, tj))
        self.Smax = max(max(self.SL(p, tj), self.SR(p, tj)) for p in range(2)
                        for tj in range(two_j12 % 2, two_j_max + 2 * order + 1, 2))
        self.Jh = len(S.two_j_values(0, order + 1, two_j12, two_j_max))   # spins of h_inf (as blocks_3d)
        self.offset, n = {}, 0
        for m in range(order):
            for jidx, tj in enumerate(S.two_j_values(m, order, two_j12, two_j_max)):
                self.offset[(m, jidx)] = n
                n += len(S.poles(tj, two_j12, two_j43, order - m - S.j_excess(tj, two_j_max)))
        self.nslots = n
        self.zero_slot = 2 * n
        self.Mkeys, self.Qkeys = {}, {}           # (P, delta, two_jNN, ppp, two_j) / (P, tj, sign) -> index
        self.dmax = 1
        self.levels = [self._level(m) for m in range(order)]
        self._physical(reuse)

    def _mkey(self, key):
        return self.Mkeys.setdefault(key, len(self.Mkeys))

    def _level(self, m):
        tj12, tj43, tjmax, order = self.tj12, self.tj43, self.tjmax, self.order
        groups = {}
        tasks = []
        for jidx, tj in enumerate(S.two_j_values(m, order, tj12, tjmax)):
            pl = S.poles(tj, tj12, tj43, order - m - S.j_excess(tj, tjmax))
            for pidx, P in enumerate(pl):
                tjp = S.pole_two_j(P, tj)
                dprime = S.delta_2(P, tj) + 2 * S.shift(P)
                for p in range(2):
                    src = (p + S.parity(P)) % 2
                    if not (self.SL(p, tj) and self.SR(p, tj) and self.SL(src, tjp) and self.SR(src, tjp)):
                        continue
                    g = groups.get((src, tjp))
                    if g is None:
                        jpex = S.j_excess(tjp, tjmax)
                        cols = [(src * self.nslots + self.offset[(m - S.shift(i), tjp // 2)]
                                 + S.position(tjp, tj12, tj43, order - m + S.shift(i) - jpex, i),
                                 S.delta_2(i, tjp)) for i in S.poles(tjp, tj12, tj43, m)]
                        g = groups[(src, tjp)] = {"rows": {}, "cols": cols, "gi": len(groups)}
                    r = g["rows"].setdefault(dprime, len(g["rows"]))
                    sign = -1 if (tjp - tj) % 4 else 1
                    tasks.append((p * self.nslots + self.offset[(m, jidx)] + pidx, g["gi"], r,
                                  self._mkey((P, 0, tj12, self.pp[src][0], tj)),
                                  self._mkey((P, 1, tj43, self.pp[src][1], tj)),
                                  self.Qkeys.setdefault((P, tj, sign), len(self.Qkeys))))
        if not groups:
            return None
        gl = sorted(groups.items(), key=lambda kv: kv[1]["gi"])
        nb = len(gl)
        Mr = max(len(g["rows"]) for _, g in gl)
        Kc = max(1, max(len(g["cols"]) for _, g in gl))
        col = torch.full((nb, Kc), self.zero_slot, dtype=torch.long)
        cidx = torch.zeros((nb, Mr, Kc), dtype=torch.long)       # coefficient table index (0 -> zero)
        cdiff = torch.zeros((nb, Mr, Kc), dtype=torch.int32)     # D' - D_i (coefficient 2/diff; 0 -> zero)
        for gi, ((src, tjp), g) in enumerate(gl):
            for c, (slot, di) in enumerate(g["cols"]):
                col[gi, c] = slot
            for d, r in g["rows"].items():
                for c, (_, di) in enumerate(g["cols"]):
                    diff = d - di
                    self.dmax = max(self.dmax, abs(diff))
                    cidx[gi, r, c] = 0 if diff == 0 else (2 * abs(diff) - (1 if diff > 0 else 0))
                    cdiff[gi, r, c] = diff
        t = torch.tensor(tasks, dtype=torch.long)
        return {"m": m, "nb": nb, "Mr": Mr, "Kc": Kc,
                "gsrc": torch.tensor([k[0] for k, _ in gl]), "gj": torch.tensor([k[1] // 2 for k, _ in gl]),
                "col": col, "cidx": cidx, "cdiff": cdiff, "t_slot": t[:, 0], "t_hrow": t[:, 1] * Mr + t[:, 2],
                "t_L": t[:, 3], "t_R": t[:, 4], "t_q": t[:, 5]}

    def _physical(self, reuse):
        """logical -> physical slots (self.phys, born slots only; unborn slots are zero and read as the
        zero slot).  reuse: greedy recycling of a slot after the level that reads it as a Cauchy column
        (the gather copies the columns before the pole step writes, so that level may reuse it), or after
        its own level if it is never read."""
        z = self.zero_slot
        born, last = {}, {}
        for lev in self.levels:
            if lev is None:
                continue
            m = lev["m"]
            for t in lev["t_slot"].tolist():
                born[t] = m
            for c in lev["col"].reshape(-1).tolist():
                if c != z:
                    last[c] = max(last.get(c, -1), m)
        if not reuse:
            self.phys, self.nphys = {t: t for t in born}, 2 * self.nslots
        else:
            self.phys, free, release, n = {}, [], {}, 0
            for t in sorted(born, key=lambda t: (born[t], t)):
                m = born[t]
                for mm in [k for k in release if k <= m]:
                    free += release.pop(mm)
                ph = free.pop() if free else n
                n += ph == n
                self.phys[t] = ph
                d = last.get(t, -1)
                release.setdefault(d if d > m else m + 1, []).append(ph)
            self.nphys = n
        self.zero_slot = self.nphys
        for lev in self.levels:
            if lev is None:
                continue
            lev["t_slot"] = torch.tensor([self.phys[t] for t in lev["t_slot"].tolist()], dtype=torch.long)
            lev["col"] = torch.tensor([self.phys.get(c, self.nphys) for c in lev["col"].reshape(-1).tolist()],
                                      dtype=torch.long).reshape(lev["col"].shape)

    def phys_slot(self, logical):
        """physical slot of a logical one (the zero slot if it is never written)"""
        return self.phys.get(logical, self.zero_slot)

    def counts(self):
        """output entries of the Cauchy and pole steps per right-hand side (rows padded per level,
        structures padded to Smax^2) -- what run() computes"""
        h = sum(lev["nb"] * lev["Mr"] for lev in self.levels if lev) * self.Smax ** 2
        r = sum(len(lev["t_slot"]) for lev in self.levels if lev) * self.Smax ** 2
        return h, r
