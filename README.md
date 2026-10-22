# BlocksviBe

GPU replacement for [blocks_3d](https://gitlab.com/bootstrapcollaboration/blocks_3d)'s computation of
3d conformal blocks (derivatives of spinning conformal blocks for the numerical bootstrap). It is meant to
be used the way blocks_3d is used: a separate block generator next to the SDP solver that writes
blocks_3d's output files. Downstream pipelines (stress-tensors-3d / hyperion, pmp2sdp) stay unchanged,
and either [SDPB](https://github.com/davidsd/sdpb) or [SDPviBe](https://github.com/AntoineTilloy/SDPviBe)
(drop-in `sdpb` on GPUs) solves the resulting SDPs.

Like SDPviBe, this code is vibe coded (written by Claude Code), and validated against blocks_3d.

## What runs where

blocks_3d, per task (one four-point structure q4, one (j12, j43, Delta_12, Delta_43) group, all spins
and lambda orders):

| stage | here |
|---|---|
| M matrices, h_infinity (setup) | blocks_3d's own C++ code, linked into the small helper `b3d_setup` (`csrc/`) |
| radial recursion | GPU: batched multiprecision recursion, all (q4, lambda order) right-hand sides of a group at once |
| residues -> log r derivatives, r^Delta, Delta -> x, pole shifting, `together` | GPU: fixed linear maps as RNS/INT8 tensor-core GEMMs |
| write_output (xt, xt_radial) | GPU GEMM with the composite coordinate map; JSON text on the CPU (gmpy2, parallel) |

The multiprecision GPU arithmetic (`sdpvibe.mpf`, `sdpvibe.rns`) comes from the `sdpvibe` package. Nothing
here depends on its solver.

Why the helper? M and h_inf are ~800 lines of blocks_3d (Clebsch-Gordan and Wigner-matrix caches, many
conventions) and cost ~1-3% of blocks_3d's time. Linking blocks_3d's own code gives the GPU chain
bit-identical inputs with no porting risk. M is computed once per group and cached on disk, so it is
shared by the 22 q4 tasks of a group. h_inf runs on the CPU, multithreaded, while other tasks use the GPU.

## Install

    pip install -e ~/Projects/SDPviBe -e ~/Projects/BlocksviBe     # or PYTHONPATH=<SDPviBe>:<BlocksviBe>
    pip install gmpy2 cupy-cuda12x                                # cupy matching your CUDA
    make -C csrc B3D=<blocks_3d checkout> ENV=<prefix with boost, gmp, eigen3 headers/libs>
                                                                  # e.g. ENV=/usr after
                                                                  # apt install libboost-dev libgmp-dev libeigen3-dev

## Drop-in `blocks_3d`

    bin/blocks_3d [blocks_3d options]          # same options, -p param file, same output files

It writes the same per-spin JSON files as blocks_3d, including the exact text layout and the number format
(boost's mpf output at blocks_3d's precision), plus the `.profile` file. On the GPU: coordinates `xt` and
`xt_radial`, with or without pole shifting (`--kept-pole-order`). Other coordinates, or a missing GPU or
helper, or a failed GPU run, go to the official binary:

    export BLOCKSVIBE_FALLBACK=/path/to/official/blocks_3d
    export BLOCKSVIBE_JEMALLOC=/path/to/libjemalloc.so.2   # official images are musl builds: 20-40x slower without

Concurrent invocations are batched. Each call drops its arguments in a spool directory per group, and the
first one waits for a quiet period (3 s) and then runs every pending task of the group as one GPU batch.
A pipeline that calls blocks_3d once per four-point structure (hyperion) therefore gets batched
recursions without changes. One process at a time uses a GPU (lock file per GPU). Explicit batches:

    bin/blocks_3d batch --lines tasks.txt      # one blocks_3d command line per line
    bin/blocks_3d batch params1 params2 ...    # blocks_3d param files

Environment variables are documented in `blocksvibe/cli.py`.

## Validation

* `tests/test_dropin.py`: the drop-in against a reference blocks_3d. Cases are TTTT, TTTe
  (Delta_43 != 0) and T-sigma-T-sigma (Delta_12, Delta_43 != 0); xt, xt_radial and both; signs +-1; with
  and without pole shifting; and a 4-task batch. Results: same files, same layout, same polynomial lengths,
  coefficients to blocks_3d's precision (2^-252.5 at 256 bits, 2^-509 at 512 bits). At 384 bits blocks_3d's
  own output carries 2^-350 rounding errors and sub-precision noise tails that the GPU result does not
  have (checked against a 768-bit blocks_3d run).
* `tests/test_blocks_*.py`: kernels and stages against exact ports of blocks_3d's algorithms.
* `vast/st3d_ab.sh`: end-to-end check on a real stress-tensor point. It runs the (T, sigma, epsilon) nmax=6
  hyperion program twice, with the official blocks_3d and with the drop-in, then compares the PMPs and
  solves both (docs/BLOCKS_PLAN.md, section 9).

## Layout

    blocksvibe/   the package: recursion_gpu, derivatives (incl. pole shifting), output, setup (helper I/O),
                  b3djson (blocks_3d's file formats), cli (drop-in), structure (pole bookkeeping)
    csrc/         b3d_setup: blocks_3d's M-matrix and h_infinity stages
    tests/        python tests/test_*.py (scripts; GPU needed)
    tools/        checks against an instrumented blocks_3d (b3d_dump.patch), JSON/PMP comparison, benchmarks
    vast/         scripts for rented GPUs (benchmarks, the end-to-end A/B run)
    docs/         BLOCKS_PLAN.md: design, validation, measurements

History: split out of SDPviBe (`sdpvibe/blocks/` on branch claude/reverent-morse-c96101) with git
filter-repo.

## Credits

blocks_3d (R. Erramilli, L. Iliesiu, P. Kravchuk, W. Landry, D. Poland, D. Simmons-Duffin, *blocks_3d:
Software for general 3d conformal blocks*, [arXiv:2011.01959](https://arxiv.org/abs/2011.01959)): the
algorithms ported here, and the source of the setup stages linked by `b3d_setup`.
