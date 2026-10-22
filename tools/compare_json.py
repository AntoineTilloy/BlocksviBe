"""Compare blocks_3d-style JSON block tables (e.g. official blocks_3d vs blocksvibe) file by file:
header fields (all but args / output_template) must be identical, the derivs nesting identical, every
polynomial's coefficient count identical once trailing coefficients below 2^-(prec - 8) of its largest are
dropped (blocks_3d leaves rounding noise where exact cancellations are expected, e.g. at 384 bits: those
polynomials come out longer, with ~1e-120 relative top coefficients), and coefficients equal to the
precision (max error relative to the largest coefficient of the polynomial).  The text layout is compared
line by line with numbers masked and each polynomial's coefficient lines collapsed.
   python tools/compare_json.py <dir or glob A> <dir or glob B> [--prec bits]
Files are paired by name.  Exit status 1 on any structural difference."""
import glob
import json
import os
import re
import sys
from collections import Counter
import mpmath as mp


def files(spec):
    fs = glob.glob(os.path.join(spec, "*.json")) if os.path.isdir(spec) else glob.glob(spec)
    return {os.path.basename(f): f for f in fs}


NUM = re.compile(r'"-?[0-9][0-9.e+-]*"|-?[0-9][0-9.e+-]*')


def skeleton(path):
    out = []
    for l in open(path):
        if '"args"' in l or '"output_template"' in l:
            continue
        m = NUM.sub("#", l).rstrip("\n")
        if m in ("#,", "#"):                     # one coefficient line: collapse runs
            if out and out[-1] == "<coeffs>":
                continue
            m = "<coeffs>"
        out.append(m)
    return out


def trim(xs, rel):
    big = max((abs(v) for v in xs), default=0)
    n = len(xs)
    while n > 1 and abs(xs[n - 1]) <= rel * big:
        n -= 1
    return xs[:n]


def main():
    a, b = files(sys.argv[1]), files(sys.argv[2])
    prec = int(sys.argv[sys.argv.index("--prec") + 1]) if "--prec" in sys.argv else 1100
    mp.mp.prec = prec + 64
    if set(a) != set(b):
        print("file sets differ:", sorted(set(a) ^ set(b))[:10]); sys.exit(1)
    worst, worst_at, npoly, bad, hist = mp.mpf(0), None, 0, 0, Counter()
    layout_diff, nraw = 0, [0]
    noise = mp.mpf(2) ** -(prec - 8)
    for name in sorted(a):
        ja, jb = json.load(open(a[name])), json.load(open(b[name]))
        for k in set(ja) | set(jb):
            if k in ("derivs", "args", "output_template"):
                continue
            if ja.get(k) != jb.get(k):
                print(f"{name}: header '{k}' differs: {ja.get(k)} vs {jb.get(k)}"); bad += 1
        la, lb = skeleton(a[name]), skeleton(b[name])
        if la != lb:
            layout_diff += 1
            if layout_diff <= 3:
                i = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
                print(f"{name}: text layout differs at line {i}: {la[i:i+1]} vs {lb[i:i+1]} ({len(la)} vs {len(lb)} lines)")

        def walk(x, y, path):
            nonlocal worst, worst_at, npoly, bad
            if isinstance(x, list) and x and isinstance(x[0], str):
                if not (isinstance(y, list) and y and isinstance(y[0], str)):
                    print(f"{name} {path}: polynomial vs {y}"); bad += 1; return
                xs, ys = [mp.mpf(v) for v in x], [mp.mpf(v) for v in y]
                if len(xs) != len(ys):
                    nraw[0] += 1
                    xs, ys = trim(xs, noise), trim(ys, noise)
                    if len(xs) != len(ys):
                        print(f"{name} {path}: polynomial length {len(xs)} vs {len(ys)} after dropping noise")
                        bad += 1; return
                big = max(abs(v) for v in xs)
                npoly += 1
                if big == 0:
                    if any(ys):
                        bad += 1; print(f"{name} {path}: zero vs nonzero")
                    return
                err = max(abs(u - v) for u, v in zip(xs, ys)) / big
                hist[int(mp.floor(mp.log(err, 2))) if err else -9999] += 1
                if err > worst:
                    worst, worst_at = err, f"{name} {path}"
                return
            if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y):
                print(f"{name} {path}: nesting differs"); bad += 1; return
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, path + [i])
        walk(ja["derivs"], jb["derivs"], [])
    print(f"{len(a)} files, {npoly} polynomials: max error relative to the largest coefficient "
          f"2^{mp.nstr(mp.log(worst, 2), 4) if worst else '-inf'} ({worst_at}); structural differences: {bad}; "
          f"files with a different text layout: {layout_diff}; polynomials of different raw length (noise tails "
          f"below 2^-{prec - 8}): {nraw[0]}")
    print("log2 error histogram:", dict(sorted(hist.items())))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
