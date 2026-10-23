# BlocksviBe

[blocks_3d](https://gitlab.com/bootstrapcollaboration/blocks_3d) on the GPU: a drop-in `blocks_3d` that
writes the same JSON files, so pipelines (stress-tensors-3d / hyperion) run unchanged. Vibe coded (Claude
Code), validated against blocks_3d. Uses [SDPviBe](https://github.com/AntoineTilloy/SDPviBe)'s
multiprecision GPU arithmetic.

**Validation** (one RTX 3090, nmax=6 stress-tensor point, all 644 blocks_3d tasks): blocks 1.86x faster than
blocks_3d on 64 cores, PMP identical to 2^-757, same solver verdict.

## Install

    pip install -e <SDPviBe> -e . gmpy2 cupy-cuda12x
    make -C csrc B3D=<blocks_3d source> ENV=<prefix with boost, gmp, eigen3>

## Use

    bin/blocks_3d [blocks_3d options]          # same CLI and -p param file as blocks_3d
    bin/blocks_3d batch --lines tasks.txt      # explicit batch, one command line per line

GPU: `xt`, `xt_radial`, with or without pole shifting. Anything else runs `$BLOCKSVIBE_FALLBACK` (the
official binary, with `$BLOCKSVIBE_JEMALLOC` preloaded). Concurrent calls of the same group are batched
automatically. Options: `blocksvibe/cli.py`.

## Layout

`blocksvibe/` package, `csrc/` helper linking blocks_3d's setup code (M matrices, h_infinity), `tests/`
(`python tests/test_*.py`), `tools/` comparison and benchmarks, `vast/` rented-GPU scripts,
`docs/BLOCKS_PLAN.md` design and measurements.

Credits: blocks_3d (Erramilli, Iliesiu, Kravchuk, Landry, Poland, Simmons-Duffin,
[arXiv:2011.01959](https://arxiv.org/abs/2011.01959)).
