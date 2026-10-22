"""blocks_3d's setup stages for the GPU chain: M matrices and h_infinity, from the C++ helper b3d_setup
(csrc/b3d_setup.cxx, blocks_3d's own functions at its working precision), read into GPU MPF tensors.

Why a helper and not a port: M (Clebsch-Gordan/SO(3) algebra per pole) and h_infinity (Wigner-matrix
derivative caches per four-point structure) are ~800 lines of blocks_3d with many conventions, and cost
~1-3% of blocks_3d's time (M once per (j12, j43, Delta_12, Delta_43) group and four-point parity; h_inf
~2 core-s per (q4, lambda order) at order 52).  Linking blocks_3d's own code gives bit-identical inputs
for the GPU chain at no porting risk, runs on the CPU (multithreaded, concurrently with GPU work of other
tasks), and is computed once per group for all its q4 when tasks are batched (cli.py batch).
"""
import os
import shutil
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def helper_path():
    """b3d_setup: $BLOCKSVIBE_B3D_SETUP, <repo>/build/b3d_setup (csrc/Makefile), or on PATH"""
    for p in (os.environ.get("BLOCKSVIBE_B3D_SETUP"), os.path.join(HERE, "..", "build", "b3d_setup"),
              shutil.which("b3d_setup")):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return os.path.abspath(p)
    return None


def run_helper(out, two_js, two_j12, two_j43, delta_12, delta_43, order, Lam, two_j_max, radial, q4s,
               precision, threads):
    """q4s: list of (q4 (4 two_j-style ints), sign).  Writes the binary setup file `out`."""
    exe = helper_path()
    if exe is None:
        raise FileNotFoundError("b3d_setup not found (build it: make -C csrc, or set BLOCKSVIBE_B3D_SETUP)")
    args = [exe, f"out={out}", f"precision={precision}", f"threads={threads}",
            "two_js=" + ",".join(map(str, two_js)), f"two_j12={two_j12}", f"two_j43={two_j43}",
            f"delta_12={delta_12}", f"delta_43={delta_43}", f"order={order}", f"lambda={Lam}",
            f"two_j_max={two_j_max}", f"radial={int(radial)}"]
    args += ["q4=" + ",".join(map(str, q)) + f",{s}" for q, s in q4s]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"b3d_setup failed ({r.returncode}): {r.stderr.strip()[-2000:]}")


def numbers_to_mpf(rec, K, L):
    """rec: (n, 2 + K) int64 records [size, exp, limbs (top K, little endian)] -> MPF (n,) (rounded to L)"""
    import torch
    from sdpvibe import mpf
    from sdpvibe.rns import dev
    n = rec.shape[0]
    if n == 0:
        return mpf.zeros((0,), L)
    size, ex = rec[:, 0], rec[:, 1]
    w = np.zeros((n, 2 * K + 1), dtype=np.uint32)
    w[:, :2 * K] = rec[:, 2:].view(np.uint32).reshape(n, 2 * K)
    neg = size < 0
    if neg.any():                                     # two's complement of the negative magnitudes
        x = w[neg]
        nz = x != 0
        low = np.argmax(nz, axis=1)                   # lowest nonzero word (nonzero values only)
        col = np.arange(x.shape[1])[None, :]
        y = np.where(col > low[:, None], ~x, x)
        y[np.arange(len(x)), low] = (-x[np.arange(len(x)), low].astype(np.int64)).astype(np.uint32)
        w[neg] = y
    ein = np.where(size == 0, 0, 64 * (ex - K)).astype(np.int64)
    Xw = torch.from_numpy(np.ascontiguousarray(w.T).view(np.int32)).to(dev)
    return mpf.normalize(Xw, torch.from_numpy(ein).to(dev), L)


class MTable:
    """M matrices from b3d_setup.  index[key] -> record; key as in dump.read_M: ((pole type, pole value),
    delta (0 = 12, 1 = 43), two_jNN, prime_pseudo_parity, two_j).  padded(Smax): MPF (nrec + 1, Smax, Smax)
    (last row: zero matrix for keys without a record)."""

    def __init__(self, hdr, vals):
        self.hdr, self.vals = hdr, vals
        self.index = {((int(h[0]), int(h[1])), int(h[2]), int(h[3]), int(h[4]), int(h[5])): i
                      for i, h in enumerate(hdr)}
        self._padded = {}

    def padded(self, Smax):
        if Smax not in self._padded:
            import torch
            from sdpvibe import mpf
            from sdpvibe.rns import dev
            hdr, L = self.hdr, self.vals.L
            nM, rows, cols = len(hdr), hdr[:, 6], hdr[:, 7]
            assert nM == 0 or (rows.max() <= Smax and cols.max() <= Smax)
            cnt = rows * cols
            rec = np.repeat(np.arange(nM), cnt)
            k = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            c = np.maximum(np.repeat(cols, cnt), 1)
            flat = (rec * Smax + k // c) * Smax + k % c
            mats = mpf.zeros(((nM + 1) * Smax * Smax,), L)
            fi = torch.from_numpy(flat).to(dev)
            mats.m[:, fi] = self.vals.m
            mats.e[fi] = self.vals.e
            self._padded[Smax] = mats.reshape(nM + 1, Smax, Smax)
        return self._padded[Smax]


class Task:
    """one four-point structure: q4 (two_j-style), sign, pseudo-parity types, lambda orders,
    h[n] = list of (pp, m, j, a, b) index array, MPF values)"""

    def __init__(self, q4, sign, pp, h):
        self.q4, self.sign, self.pp, self.h = q4, sign, pp, h


def read(path, L):
    """-> (MTable, [Task]) with numbers on the GPU at L limbs"""
    raw = np.fromfile(path, dtype=np.int64)
    assert raw[:1].view("S8")[0] == b"B3DSETUP", "not a b3d_setup file"
    K = int(raw[1]); pos = 2
    rlen = 2 + K

    def words(k):
        nonlocal pos
        out = raw[pos:pos + k]; pos += k
        return out

    nM = int(words(1)[0])
    hdr = words(8 * nM).reshape(nM, 8).copy()
    nvals = int((hdr[:, 6] * hdr[:, 7]).sum())
    M = MTable(hdr, numbers_to_mpf(words(rlen * nvals).reshape(nvals, rlen), K, L))

    tasks = []
    for _ in range(int(words(1)[0])):
        t = words(10)
        q4, sign = tuple(int(x) for x in t[:4]), int(t[4])
        pp = ((int(t[5]), int(t[6])), (int(t[7]), int(t[8])))
        h = {}
        for _ in range(int(t[9])):
            n, cnt = (int(x) for x in words(2))
            idx = words(5 * cnt).reshape(cnt, 5).copy()
            h[n] = (idx, numbers_to_mpf(words(rlen * cnt).reshape(cnt, rlen), K, L))
        tasks.append(Task(q4, sign, pp, h))
    assert pos == len(raw), "trailing data in b3d_setup file"
    return M, tasks


def h_inf_tensor(task, ns, order, Jh, Smax, L):
    """h_inf MPF (2, order + 1, Jh, Smax^2, len(ns)) for the lambda orders ns of a task (zeros elsewhere),
    the layout of Recursion.run"""
    import torch
    from sdpvibe import mpf
    from sdpvibe.rns import dev
    B = len(ns)
    out = mpf.zeros((2 * (order + 1) * Jh * Smax * Smax * B,), L)
    for k, n in enumerate(ns):
        idx, v = task.h[n]
        s, m, j, a, b = idx.T
        ok = (m <= order) & (j < Jh)
        flat = ((((s * (order + 1) + m) * Jh + j) * Smax + a) * Smax + b) * B + k
        fi = torch.from_numpy(flat[ok]).to(dev)
        okt = torch.from_numpy(ok).to(dev)
        out.m[:, fi] = v.m[:, okt]; out.e[fi] = v.e[okt]
    return out.reshape(2, order + 1, Jh, Smax * Smax, B)
