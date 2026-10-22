"""The drop-in CLI (bin/blocks_3d -> blocksvibe.cli) against a reference blocks_3d binary on small tasks of
the stress-tensor families: TTTT, TTTe (Delta_43 != 0), T-sigma (Delta_12, Delta_43 != 0); xt, xt_radial,
both; signs +-1; single tasks and one batch.  Same files, same text layout (numbers masked), same
polynomial lengths (up to blocks_3d's sub-precision noise tails, tools/compare_json.py), coefficients to
the precision.  (TTTe case at 384 bits vs blocks_3d at 768 bits: GPU 2^-374.6, blocks_3d itself 2^-350.7.)
   B3D_REF=<blocks_3d binary> [B3D_REF_PRELOAD=<libjemalloc.so>] python tests/test_dropin.py [workdir]
Needs csrc/ built (build/b3d_setup)."""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = os.environ.get("B3D_REF", os.path.expanduser("~/Projects/sdpb_gpu/ext/src_b3d/blocks_3d/build_native/blocks_3d"))
PRE = os.environ.get("B3D_REF_PRELOAD", os.path.expanduser("~/Projects/sdpb_gpu/ext/mamba/b3denv/lib/libjemalloc.so.2"))

TTTT = "--j-external 2,2,2,2 --j-12 2 --j-43 2 --delta-12 0 --delta-43 0"
CASES = {   # name: (blocks_3d options without -o, precision for the comparison)
    "tttt_xt": (TTTT + " --j-internal 0-10 --delta-1-plus-2 2.3 --four-pt-struct 1,2,1,0 --four-pt-sign 1"
                " --order 16 --lambda 7 --coordinates xt --kept-pole-order 16 --precision 256", 256),
    "tttt_radial_minus": (TTTT + " --j-internal 0-12 --delta-1-plus-2 6 --four-pt-struct 2,1,0,1 --four-pt-sign -1"
                          " --order 20 --lambda 11 --coordinates xt_radial --kept-pole-order 20 --precision 512", 512),
    "ttte_xt": ("--j-external 2,2,2,0 --j-12 2 --j-43 2 --delta-12 0 --delta-43 -1.5874 --j-internal 0-9,12"
                " --delta-1-plus-2 6 --four-pt-struct 1,0,2,0 --four-pt-sign -1 --order 18 --lambda 9"
                " --coordinates xt --kept-pole-order 18 --precision 384", 384),
    "tttt_shift": (TTTT + " --j-internal 0-12 --delta-1-plus-2 6 --four-pt-struct 1,2,1,0 --four-pt-sign 1"
                   " --order 30 --lambda 9 --coordinates xt --kept-pole-order 8 --precision 512", 512),
    "ttte_shift_radial": ("--j-external 2,2,2,0 --j-12 4 --j-43 2 --delta-12 0 --delta-43 -1.5874 --j-internal 0-10"
                          " --delta-1-plus-2 6 --four-pt-struct 2,0,2,0 --four-pt-sign -1 --order 24 --lambda 11"
                          " --coordinates xt_radial --kept-pole-order 6 --precision 512", 512),
    "tsts_both": ("--j-external 2,0,0,2 --j-12 2 --j-43 2 --delta-12 2.4819 --delta-43 2.4819 --j-internal 0-8"
                  " --delta-1-plus-2 3.5181 --four-pt-struct 1,0,0,-1 --four-pt-sign 1 --order 14 --lambda 9"
                  " --coordinates xt,xt_radial --kept-pole-order 14 --precision 256", 256),
}
BATCH = [TTTT + f" --j-internal 0-8 --delta-1-plus-2 6 --four-pt-struct {q} --four-pt-sign {s} --order 14 --lambda 9"
         " --coordinates xt --kept-pole-order 14 --precision 256" for q, s in
         (("2,2,2,2", 1), ("1,2,1,0", 1), ("1,2,1,0", -1), ("0,1,-1,2", -1))]


def run(cmd, env=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    assert r.returncode == 0, (cmd, r.stdout[-2000:], r.stderr[-2000:])


def compare(a, b, prec):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "compare_json.py"), a, b, "--prec", str(prec)],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    assert r.returncode == 0 and "different text layout: 0" in r.stdout, r.stdout + r.stderr
    worst = float(r.stdout.split("coefficient 2^")[1].split()[0])
    assert worst < -(prec - 40), worst          # blocks_3d's own error reaches 2^-350 at 384 bits (TTTe case)


def main():
    w = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="dropin_")
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("BLOCKSVIBE_PYTHON", sys.executable)
    pre = f"LD_PRELOAD={PRE} " if os.path.exists(PRE) else ""
    for name, (opts, prec) in CASES.items():
        run(f"{pre}{REF} {opts} --num-threads 4 -o {w}/{name}/ref/b_{{}}.json > /dev/null")
        run(f"{ROOT}/bin/blocks_3d {opts} --num-threads 4 -o {w}/{name}/gpu/b_{{}}.json", env)
        print(name, end=": ")
        compare(f"{w}/{name}/ref", f"{w}/{name}/gpu", prec)
    lines = os.path.join(w, "batch.txt")
    with open(lines, "w") as f:
        for i, opts in enumerate(BATCH):
            run(f"{pre}{REF} {opts} --num-threads 4 -o {w}/batch{i}/ref/b_{{}}.json > /dev/null")
            f.write(f"{opts} --num-threads 4 -o {w}/batch{i}/gpu/b_{{}}.json\n")
    run(f"{ROOT}/bin/blocks_3d batch --lines {lines}", env)
    for i in range(len(BATCH)):
        print(f"batch task {i}", end=": ")
        compare(f"{w}/batch{i}/ref", f"{w}/batch{i}/gpu", 256)
    print("all drop-in cases agree with blocks_3d")


if __name__ == "__main__":
    main()
