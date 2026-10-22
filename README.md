# BlocksviBe

GPU replacement for [blocks_3d](https://gitlab.com/bootstrapcollaboration/blocks_3d)'s computation of
3d conformal blocks (derivatives of spinning conformal blocks for the numerical bootstrap), written to be
used the way blocks_3d is used: a separate block generator next to the SDP solver, writing
blocks_3d-compatible output, so downstream pipelines (stress-tensors-3d / hyperion, pmp2sdp) are unchanged
and either [SDPB](https://github.com/davidsd/sdpb) or [SDPviBe](https://github.com/AntoineTilloy/SDPviBe)
solves the resulting SDPs.

The multiprecision GPU arithmetic (`sdpvibe.mpf`, `sdpvibe.rns`: exact RNS/INT8 tensor-core GEMMs and
CUDA-core multiprecision kernels) comes from the `sdpvibe` package; nothing here depends on its solver.

Layout: `blocksvibe/` (the package), `tests/` (run as scripts: `python tests/test_blocks_*.py`),
`tools/` (benchmarks and checks against an instrumented blocks_3d, `tools/b3d_dump.patch`), `vast/`
(benchmark scripts for rented GPUs), `docs/BLOCKS_PLAN.md` (design, validation, measurements).

Install next to SDPviBe:

    pip install -e ~/Projects/SDPviBe -e ~/Projects/BlocksviBe      # or PYTHONPATH=<SDPviBe>:<BlocksviBe>

History: split out of SDPviBe (branch claude/reverent-morse-c96101, `sdpvibe/blocks/`) with git filter-repo.
