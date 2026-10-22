"""Compare two polynomial matrix programs (SDPB PMP JSON, as written by stress-tensors-3d's `part` stage)
built from different block tables: same JSON structure, every number equal to the precision.
   python tools/compare_pmp.py <nsv or dir A> <nsv or dir B> [--prec bits]
An .nsv is the null-separated list of PMP files pmp2sdp reads; files are paired by name.  Numbers are
compared per innermost list (a polynomial's coefficients, a sample vector, ...) relative to its largest
entry, scalars relative to themselves; innermost lists of different length are compared after dropping
trailing entries below 2^-(prec - 8) of the largest (blocks_3d's sub-precision noise tails can lengthen
polynomials, tools/compare_json.py).  Prints the worst error and where; exit 1 on structural differences."""
import glob
import json
import os
import sys
from collections import Counter
import mpmath as mp


def pmp_files(spec):
    if os.path.isdir(spec):
        fs = glob.glob(os.path.join(spec, "**", "*.json"), recursive=True)
    elif spec.endswith(".nsv"):
        fs = [f for f in open(spec, "rb").read().decode().split("\0") if f.strip()]
        base = os.path.dirname(spec)
        fs = [f if os.path.isabs(f) else os.path.join(base, f) for f in fs]
    else:
        fs = glob.glob(spec)
    return {os.path.basename(f): f for f in fs}


def num(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return mp.mpf(x)
    if isinstance(x, str):
        try:
            return mp.mpf(x)
        except (ValueError, TypeError):
            return None
    return None


def main():
    a, b = pmp_files(sys.argv[1]), pmp_files(sys.argv[2])
    prec = int(sys.argv[sys.argv.index("--prec") + 1]) if "--prec" in sys.argv else 1024
    mp.mp.prec = prec + 64
    noise = mp.mpf(2) ** -(prec - 8)
    if set(a) != set(b):
        print("file sets differ:", sorted(set(a) ^ set(b))[:10]); sys.exit(1)
    st = {"worst": mp.mpf(0), "at": None, "lists": 0, "nums": 0, "bad": 0, "trimmed": 0}
    hist = Counter()

    def trim(v):
        big = max((abs(x) for x in v), default=0)
        n = len(v)
        while n > 1 and abs(v[n - 1]) <= noise * big:
            n -= 1
        return v[:n]

    def walk(x, y, path, name):
        if isinstance(x, dict):
            if not isinstance(y, dict) or set(x) != set(y):
                print(f"{name} {path}: keys differ"); st["bad"] += 1; return
            for k in x:
                walk(x[k], y[k], path + [k], name)
            return
        if isinstance(x, list):
            if not isinstance(y, list):
                print(f"{name} {path}: list vs {type(y).__name__}"); st["bad"] += 1; return
            xs = [num(v) for v in x] if x and not isinstance(x[0], (list, dict)) else None
            ys = [num(v) for v in y] if y and not isinstance(y[0], (list, dict)) else None
            if xs is not None and ys is not None and None not in xs and None not in ys:
                if len(xs) != len(ys):
                    st["trimmed"] += 1
                    xs, ys = trim(xs), trim(ys)
                    if len(xs) != len(ys):
                        print(f"{name} {path}: length {len(xs)} vs {len(ys)} after dropping noise"); st["bad"] += 1
                        return
                big = max((abs(v) for v in xs), default=0)
                st["lists"] += 1; st["nums"] += len(xs)
                if big == 0:
                    if any(ys):
                        print(f"{name} {path}: zero vs nonzero"); st["bad"] += 1
                    return
                err = max(abs(u - v) for u, v in zip(xs, ys)) / big
                hist[int(mp.floor(mp.log(err, 2))) if err else -9999] += 1
                if err > st["worst"]:
                    st["worst"], st["at"] = err, f"{name} {path}"
                return
            if len(x) != len(y):
                print(f"{name} {path}: length {len(x)} vs {len(y)}"); st["bad"] += 1; return
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, path + [i], name)
            return
        u, v = num(x), num(y)
        if u is None or v is None:
            if x != y:
                print(f"{name} {path}: {x!r} vs {y!r}"); st["bad"] += 1
            return
        st["nums"] += 1
        if u != v:
            err = abs(u - v) / max(abs(u), abs(v))
            hist[int(mp.floor(mp.log(err, 2)))] += 1
            if err > st["worst"]:
                st["worst"], st["at"] = err, f"{name} {path}"

    for name in sorted(a):
        walk(json.load(open(a[name])), json.load(open(b[name])), [], name)
    w = st["worst"]
    print(f"{len(a)} PMP files, {st['nums']} numbers in {st['lists']} vectors: max relative error "
          f"2^{mp.nstr(mp.log(w, 2), 4) if w else '-inf'} ({st['at']}); structural differences: {st['bad']}; "
          f"vectors compared after dropping noise tails: {st['trimmed']}")
    print("log2 error histogram:", dict(sorted(hist.items())))
    sys.exit(1 if st["bad"] else 0)


if __name__ == "__main__":
    main()
