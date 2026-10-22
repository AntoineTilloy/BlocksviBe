"""write_output on the GPU (blocksvibe/output.py: one composite matrix per (Lambda, parity, Delta_1+2))
vs a line-by-line mpmath port of blocks_3d's lambda_lr_to_lrho_lrhob, congruence_transform (stirling, rho_z,
power_prefactor), zzb_to_xt and lrho_to_xt_radial, on random numerator coefficients.
(Checked against blocks_3d's JSON output with tools/check_output.py.)
   python tests/test_blocks_output.py"""
import random
import mpmath as mp
import torch
from sdpvibe import mpf
from blocksvibe import output as OUT


def congruence(parity, T, inp):
    O = len(inp) - 1
    inter = [[mp.mpf(0)] * (O + 1) for _ in range(O + 1)]
    for i in range(O + 1):
        for k in range(O + 1):
            for l in range(O + 1 - k):
                v = inp[k][l] if l <= k else (inp[l][k] if parity % 2 == 0 else -inp[l][k])
                inter[i][l] += T[i, k] * v
    return [[sum((T[j, l] * inter[i][l] for l in range(O)), mp.mpf(0)) for j in range(min(i, O - i) + 1)]
            for i in range(O + 1)]


def reference_xt(num, Lam, parity, D):
    """num[idx][e] (scalars) -> xt[n][b]"""
    O = Lam
    F = [[sum(((-1) ** (j - l) * mp.mpf(2) ** (-i - j) * mp.binomial(i, k) * mp.binomial(j, l)
               * num[(i + j - k - l - parity) // 2][k + l]
               for k in range(i + 1) for l in range((i + j - k - parity + 2) % 2, j + 1, 2)), mp.mpf(0))
          for j in range(min(i, O - i) + 1)] for i in range(O + 1)]
    Z = congruence(parity, OUT.rho_z(O), congruence(parity, OUT.stirling(O), F))
    if D != 0:
        Z = congruence(parity, OUT.power_prefactor(O, D / 2), Z)
    out = []
    for n in range(O - parity + 1):
        row = []
        for m in range(parity, O - n + 1, 2):
            r = mp.mpf(0)
            for k in range(n + 1):
                for l in range(m + 1):
                    i, j = n + m - k - l, k + l
                    z = Z[i][j] if i >= j else (1 if parity % 2 == 0 else -1) * Z[j][i]
                    r += (-1) ** l * mp.factorial((m - parity) // 2) / mp.factorial(m) * mp.binomial(n, k) * mp.binomial(m, l) * z
            row.append(r)
        out.append(row)
    return out


def test(Lam, parity, D, L=10, seed=0):
    rng = random.Random(seed)
    o_xt, o_rad = OUT.Output("xt", Lam, parity, D, L), OUT.Output("xt_radial", Lam, parity, D, L)
    mp.mp.prec = 32 * L + 128
    nL, R = (Lam - parity) // 2 + 1, Lam - parity
    num = [[mp.mpf(rng.uniform(-1, 1)) * mp.mpf(2) ** rng.randint(-20, 20) if e <= Lam - parity - 2 * idx else mp.mpf(0)
            for e in range(R + 1)] for idx in range(nL)]
    # as a Derivs output: (Jo=1, Dout=1, R+1, 2, S2=1, nL), parity slot 0 only
    vals = [num[idx][e] if p == 0 else 0 for e in range(R + 1) for p in range(2) for idx in range(nL)]
    X = mpf.from_mp(vals, L, (1, 1, R + 1, 2, 1, nL))
    worst = 0
    for o, ref in ((o_xt, reference_xt(num, Lam, parity, mp.mpf(D))), (o_rad, None)):
        g = o.apply(X, nL=nL).to_mp()[0, 0, 0, 0, :, 0]
        if ref is None:                                    # lrho_to_xt_radial: one matrix product
            O = Lam - parity
            pre = (OUT.power_prefactor(O, mp.mpf(D)) if parity == 0 else
                   OUT.one_minus_power_prefactor(O) * OUT.power_prefactor(O, mp.mpf(D) + 1))
            T = pre * OUT.rho_z(O) * OUT.stirling(O)
            ref = [[sum((T[i, j] * num[0][j] for j in range(O + 1)), mp.mpf(0))] for i in range(O + 1)]
        flat = [x for row in ref for x in row]
        big = max(abs(x) for x in flat)
        worst = max(worst, max(abs(a - b) for a, b in zip(g, flat)) / big)
    bits = -float(mp.log(worst, 2))
    print(f"Lambda {Lam}, parity {parity}, Delta_1+2 {D}: xt {len(o_xt.outputs)} and xt_radial {len(o_rad.outputs)} "
          f"outputs, max error relative to the largest 2^-{bits:.1f}")
    assert bits > 32 * L - 40, bits


if __name__ == "__main__":
    test(7, 0, "2.3"); test(7, 1, "2.3"); test(6, 0, "0"); test(9, 1, "6", seed=2)
