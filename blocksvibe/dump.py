"""Reader for the debug dumps of an instrumented blocks_3d (tools/blocks/b3d_dump.patch, env B3D_DUMP).

<prefix>_M.txt   : M type k delta(0=12,1=43) two_jNN prime_pseudo_parity two_j rows cols entries...
<prefix>_n<n>.txt: P pp00 pp01 pp10 pp11 / H pp m j a b value / R pp m j pole a b value
<prefix>_N<n>.txt: N two_j pp a b e ncoeffs coeffs... (Log_r_Derivatives numerators in x, read by
                  tools/blocks/check_derivs.py)
Values are kept as decimal strings (converted by the consumer at its own precision).
"""
from collections import defaultdict


def read_M(path):
    M = {}
    with open(path) as f:
        for line in f:
            w = line.split()
            t, k, d, tjNN, ppp, tj, r, c = map(int, w[1:9])
            v = w[9:]
            M[((t, k), d, tjNN, ppp, tj)] = [v[i * c:(i + 1) * c] for i in range(r)]
    return M


def read_n(path):
    """-> pseudo_parities, H[(pp, m, j, a, b)] = str, R[(pp, m, j, pole, a, b)] = str"""
    H, R, pp = {}, {}, None
    with open(path) as f:
        for line in f:
            w = line.split()
            if w[0] == "P":
                a = list(map(int, w[1:]))
                pp = ((a[0], a[1]), (a[2], a[3]))
            elif w[0] == "H":
                H[tuple(map(int, w[1:6]))] = w[6]
            else:
                R[tuple(map(int, w[1:7]))] = w[7]
    return pp, H, R
