# Blocks and PMP generation for the stress-tensor bootstrap: why it is slow, how to make it fast

2026-10-20.  Scope: the problem-generation side of the (T, sigma, epsilon) pipeline of
<https://gitlab.com/davidsd/stress-tensors-3d> (blocks_3d + Haskell/hyperion PMP assembly), now that the
SDP solve runs on GPUs (SOTA_PLAN.md).  "Measured" = run here (laptop: i7-11850H 8C/16T + RTX A3000 6 GB;
vast.ai: Threadripper 3960X + RTX 3090, 2026-10-20); everything else is labelled as an estimate.

## 0. Summary

1. **The 20-40x we were losing was the allocator, not the algorithm.**  The official blocks_3d docker
   image is Alpine/musl; musl's malloc serialises GMP's many temporaries across threads (2/3 of the CPU
   time is spent in the kernel).  Preloading Alpine's jemalloc (or mimalloc) into the same binary gives
   bit-identical output and:

   | task (8 threads unless noted) | musl (image as is) | + mimalloc | + jemalloc | native glibc build |
   |---|---|---|---|---|
   | TTTT j12=j43=4, order 30, Lambda 11, xt, 29 spins (laptop) | 1022 s | 32 s | | |
   | production nmax=10 task: TTTT j12=j43=4, order 60, Lambda 19, xt_radial, spins 0..60 | 25-50 min (EPYC, our vast run) | 115 s (laptop) | **74 s** (laptop) | 84 s (78 s + jemalloc) |
   | order-30 task on the 3090 box (3960X, 12 cores, 48 threads) | >1500 s (stopped) | | **37 s** | |
   | order-60 task on the 3090 box | | | 108 s (48 thr), 96 s (8 thr) | |

   Our nmax=10 attempt (550 tasks, ~7 h of blocks on 64 cores) would have taken ~15-30 min.  The
   collaboration's task statistics are *not* affected (their nmax=6 TTTe task takes 269 s on Expanse vs
   235 s on this laptop with jemalloc, same 10 threads): they use a native build.  Fixed in
   `vast/st3d_setup.sh` (blocks_3d and scalar_blocks wrappers preload jemalloc).  Side finding: the
   official **SDPB** image is musl too; jemalloc makes it 1.37x faster on an nmax=6 SDP (one rank, same
   iterates), so our GPU-vs-SDPB comparisons were mildly unfair to SDPB.

2. **Where the time goes at production size** (their statistics, core-hours on Expanse):

   | | nmax=22 (Lambda 43, order 130, 832 bits) | nmax=26 (Lambda 51, order 152, 896 bits) |
   |---|---|---|
   | one-time per island: TTTT blocks_3d (550 tasks) | ~7.4k (est. from nmax=18/26) | 18.4k |
   | one-time: ttoBlock (TTO x TTO composites, Haskell) | ~0.6k (est.) | 1.0k |
   | **per point: TTTe blocks_3d (45 tasks, depend on Delta_eps)** | **980** | **2,420** |
   | per point: blocks with sigma (Delta_12/43 != 0) | 432 | 951 |
   | per point: TTss/TTee blocks_3d (point-independent, recomputed) | 73 | 188 |
   | per point: Haskell (tsMixedBlock + partChunk + part) | 314 | 725 |
   | **per point total** | **~1.8k** | **~4.3k** |

   blocks_3d is ~80% of the per-point work; within blocks_3d the radial recursion dominates at high order
   (70% of a nmax=6 TTTe task at order 80, 44% of a nmax=14 TTTT task, where the post-recursion stages
   and 19 GB of JSON output per task take the rest; section 2).

3. **The block computation is linear algebra with a huge batch dimension.**  For a given
   (j12, j43, Delta_12, Delta_43) the recursion operator (pole positions, Cauchy coefficients
   2/(Delta'-Delta_i), the L/R "M matrices", Q) is the same for all four-point structures q4 and all
   lambda-derivative orders n: e.g. the 550 TTTT tasks are 25 operators x 22 q4, each applied to 26
   values of n, i.e. **572 right-hand sides per operator**.  blocks_3d runs them as 550 separate
   processes and recomputes the operator (and its M-matrix cache, single-threaded) in each.  Batched,
   each recursion level is a batched multiprecision GEMM (Cauchy step) plus a batched 25x25 GEMM (pole
   step).  All later stages (residues -> log r derivatives, r^Delta restoration, Delta -> x shift,
   "together" into numerator polynomials, coordinate transforms) are fixed linear maps too.

4. **GPU prototype** (`blocksvibe/`): the batched recursion (CUDA-core kernels for the Cauchy and
   pole steps, RNS/INT8 GEMMs as an alternative), **validated against an instrumented blocks_3d**
   (residues agree to 2^-262 relative at 256 bits, two n batched; the only differing entries are
   blocks_3d's own cancellation noise, which the exact CPU port reproduces).  **Measured on an RTX 3090
   at 926 bits: 0.99 s per right-hand side at order 60 (nmax=10 shape) vs 142 CPU core-seconds for
   blocks_3d's recursion at 896 bits, i.e. one 3090 ~ 140 cores**, improving with the order (~74 at
   order 30, ~1.6x per doubling): projected ~300-600 cores at nmax=22-26 with these kernels (blocks_3d itself measured at
   ~1,700 core-s per right-hand side at order 130), ~1000+ with the next kernel work (section 6).  Memory, not arithmetic, is the production constraint: 9-19 GB of live state per
   right-hand side at nmax=22-26 (section 5).
   **Post-recursion stages ported too (2026-10-21)**: derivative moments accumulated during the recursion,
   r^Delta restoration, Delta -> x shift and `together_with_factors` as batched RNS GEMMs, producing
   blocks_3d's `Log_r_Derivatives` numerators; **validated against blocks_3d to its working precision**
   (2^-261 at 256 bits, 2^-506 at 512 bits, relative to the largest coefficient, TTTT j12=j43=2, orders
   16-20).  **4.3 s per right-hand side on the laptop A3000 at the nmax=10 shape vs 211 core-s for the
   same blocks_3d stages** (A3000 ~ 50 cores, ~40-55 at order 30; a 3090 should be ~3x faster,
   estimate), with residue slots recycled after their last use (42% of the slots at order 60).
   **And write_output (xt, xt_radial) plus a vectorised plan builder (2026-10-21)**: the whole blocks_3d
   chain after h_inf now runs on the GPU and matches blocks_3d's JSON to its working precision (2^-508.7
   at 512 bits).  **Measured end to end per right-hand side: order 52 / Lambda 19: 266 core-s
   (blocks_3d, 1 thread) vs 4.28 s on the A3000 (62 cores) and 1.66 s on a 3090 (160 cores); nmax=10
   shape 2.06 s, nmax=22 shape ~17.5 s on a 3090 (~200 cores)**; plan builder 0.4 s at order 60 (was 265
   s), 6.3 s at order 152 (section 6).  **Optimised 2026-10-22 (aligned RNS GEMMs, B=1 Cauchy rule,
   tabulated pole operands, windowed moments): 1.05 s at order 52 (254 cores), 1.39 s at nmax=10, ~8.8 s
   at nmax=22 on a 3090, i.e. ~400 cores and ~3.6 h (~$0.75) of one 3090 per nmax=22 point.**

5. **Haskell side** (section 7): ttoBlock/tsMixedBlock are JSON-parse bound (~50 us per 240-byte
   number, files re-read 3x), partChunk is dominated by an O(D^2) Horner polynomial shift x -> x+1e-6 per
   term, and several point-independent tables/chunks are recomputed for every point.  Estimated
   725 -> ~100 core-h per point at nmax=26 with modest changes.

6. **Delta_eps interpolation of the TTTe family is not a free win**: blocks are polynomials in
   Delta_43 of degree ~2 x order; a Taylor/Chebyshev representation accurate to 2^-830 over an island of
   width ~1e-4 needs degree ~60-80, i.e. as many nodes as an island has points.  It pays only if the
   block precision can be lowered (open question, cheap to test, section 8).

## 1. The pipeline and its task statistics

Per point hyperion runs (task names from `task_stats/`): `block3d` (blocks_3d, one task = one four-point
structure q4 x (j12, j43) for all spins and all lambda orders), `tsMixedBlock`/`ttoBlock` (composite
blocks: sums of blocks_3d tables with three-point-structure coefficients, Haskell), `partChunk`/`part`
(PMP pieces, Haskell), then pmp2sdp and SDPB.  TTTT tables and the TTO x TTO composites are stored in a
central directory and computed once per (nmax, order, precision); everything else is per point.

blocks_3d families (j_external): TTTT (2,2,2,2) one-time; TTTe (2,2,2,0) with Delta_43 = Delta_eps - 3
(45 tasks, the per-point bulk); (2,2,0,0) TTss/TTee with Delta_12 = Delta_43 = 0 (point-independent, but
stored per job); sigma-containing families with Delta_12, Delta_43 != 0.  Within a family the cost varies
~10x between q4 structures (e.g. nmax=26 TTTe j12=4: 0.23 h vs 2.0 h at 128 threads); memory reaches
200 GB for TTTT j12=j43=4 tasks at nmax=26 (all residues of one n plus all outputs are held in RAM).

## 2. blocks_3d: algorithm and cost

`lambda_log_r_tables` loops over lambda orders n = parity, parity+2, ..., Lambda.  For each n:

* `h_infinity` (cheap): initial data h_inf[pp][m][j][a][b], m <= order, spins up to j_max + order.
* **`radial_recursion`** (`fill_residues`): for m = 0..order-1, for every spin j and pole P of the block
  (types I-IV, ~2.5 (order - m) poles), with j' = P.two_j(j):
  `H = h_inf[src][m][j'] + sum_i 2/(D'_P - D_i) res[src][m - shift_i][j'][i]` (i over the ~2.5 m poles
  of the j' block at order m), then `res[p][m][j][P] = Q L_P H R_P^T` (S x S structure matrices,
  S <= j12 + 1 <= 5 per pseudo-parity).  Cost ~ J order^3 S^2; e.g. per n (CPU multiprecision MACs, `opcount.py` model
  that reproduces blocks_3d's per-level task counts exactly):

  | (2 j12, 2 j43), j_max, order | tasks (both parities) | Cauchy MACs | L/R MACs |
  |---|---|---|---|
  | (8,8), 30, 30 | 9.2e4 | 4.5e7 | 1.6e7 |
  | (8,8), 60, 60 (nmax=10) | 7.0e5 | 6.7e8 | 1.3e8 |
  | (8,8), 50, 105 (nmax=18) | 2.1e6 | 3.1e9 | 3.8e8 |
  | (8,8), 90, 130 (nmax=22) | 5.1e6 | 1.0e10 | 9.4e8 |
  | (8,8), 100, 152 (nmax=26) | 7.8e6 | 1.8e10 | 1.4e9 |
  | (4,4), 100, 152 | 7.7e6 | 5.4e9 | 2.7e8 |

  Measured per-task time is ~450 us at order 30 independent of m under musl (allocation-bound) and
  ~110-170 us/task with jemalloc (orders 30-60, 512 bits).
* `residues_to_derivatives`: sum_m m^k r_c^m res[m] for k <= Lambda - n (a GEMM with K = order).
* `restore_r_delta` (log r derivatives of r^Delta h: O(R^3) per pole as written, O(R^2) suffices),
  `shift` (Delta -> x, and `together_with_factors`: numerator = poly prod(x - x_i) + sum_i r_i
  prod_{k!=i}(x - x_k), O(P^2) per derivative with P ~ 2.5 order poles), `write_output` (polynomial
  coordinate transforms (lambda, log r) -> (z, zb) -> (x, t), ~Lambda^4 polynomial operations per
  element, JSON at full precision).
* Output volume: ~P + Lambda coefficients per (spin, structure, derivative): for TTTT at nmax=26 ~20 TB
  of JSON in total (360 MB per (table, spin) file, inferred from the ttoBlock parse statistics).

Measured stage splits (jemalloc/mimalloc unless noted; seconds, all lambda orders):

| task | threads | total | init | recursion | derivs | shift | output |
|---|---|---|---|---|---|---|---|
| TTTT j12=j43=4, order 30, Lambda 11, xt, musl | 8 | 1022 | 5 | 20 | 176 | 377 | 431 |
| same, mimalloc | 8 | 32 | 4.5 | 7.5 | 2.9 | 4.9 | 10.9 |
| TTTT j12=j43=4, order 60, Lambda 19, xt_radial | 8 | 74 | 28 | 14.5 | 10.6 | 14.3 | 5.1 |
| TTTe j12=4, j43=2, order 80, Lambda 11, xt, 1024 bits | 10 | 235 | 24 | 163 | 6 | 15 | 17 |
| TTTT j12=j43=4, order 90, Lambda 27, xt, 832 bits, spins 0..50 (nmax=14 shape; 3960X, 12 cores) | 48 | 2828 | 96 | 1236 (44%) | 225 | 460 | 731 |

The nmax=14-shape task wrote 19 GB of JSON (one four-point structure).  The post-recursion stages
(derivatives, shift/together, output) are already half of the time at nmax=14; their cost grows like
Lambda^2 x poles per spin, the recursion's like order^3, so their share should reach ~2/3 at nmax=22-26
(Lambda^2/order: 8 at nmax=14, 14 at nmax=22, 17 at nmax=26): a GPU port of the recursion alone would
gain at most ~1.5-2x; the whole chain must move.
`init` (M matrices, pole shifting) is single-threaded and identical for the 22 q4 tasks of a (j12, j43)
group: computing it once per group is a free CPU-side saving.

## 3. Reuse across tasks and points

* **Same operator, many right-hand sides** (section 0.3): batch q4 x n.  TTTT: 25 operators x (22 q4 x
  26 n).  TTTe: 5 operators (j12 = 0..4, j43 = 2) x (9 q4 x ~26 n).
* **TTTT once per island** (already done by hyperion via the central store), likewise TTO x TTO
  composites.  Not done but point-independent: (2,2,0,0) tables (73-188 core-h per point), TTTT
  partChunks (~28% of partChunk), TTO x SO3 composites keyed by an irrelevant external Delta.
* **Delta_eps (TTTe)**: the operator depends on Delta_43 through the R matrices (polynomials of degree
  growing with the pole order) and h_inf (binomial series).  Taylor in delta = Delta_43 - Delta_43^0:
  the Cauchy step is linear (D+1 times the right-hand sides) but the pole step becomes a truncated
  polynomial convolution ((D+1)^2/2); with |delta| ~ 5e-5 and pole orders ~150, terms fall like
  (2 order |delta|)^D / D!, so D ~ 70-80 for 2^-830.  Chebyshev nodes over the island need as many
  evaluations.  Only worth it at reduced block precision (section 8).

## 4. GPU blocks: design and prototype

`blocksvibe/structure.py` (pole/spin bookkeeping, 1:1 port), `recursion_ref.py` (exact CPU port of
`fill_residues`, mpmath), `recursion_gpu.py` (batched GPU recursion), `cauchy.py` and `smallgemm.py`
(CUDA-core kernels), `dump.py` + `tools/b3d_dump.patch` (instrumented blocks_3d: dumps M matrices,
h_inf and residues when `B3D_DUMP=<prefix>`).  Checks: `tools/check_ref.py`, `check_gpu.py` (vs
dumps), `tests/test_blocks_recursion.py` (GPU vs exact port, all kernel variants).  Benchmarks:
`tools/bench_recursion.py`, `bench_cauchy.py`, `bench_shapes.py`, `opcount.py`, `live_memory.py`.
Post-recursion stages: `derivatives.py`; checks `tools/check_derivs.py` (vs the numerators dumped
by the patched blocks_3d), `tests/test_blocks_derivatives.py` (vs an mpmath port of blocks_3d's code);
benchmark `tools/bench_derivs.py`.

* Plan (CPU, once per operator): slot layout of all residues (slot = parity, level, spin, pole), per level
  the (src, j') groups with their Cauchy rows D' and columns (prime poles), and per task the indices of
  its H row, L/R matrices and Q.  **Vectorised (numpy) since 2026-10-21**, identical tensors to the first
  per-task Python builder (kept as `plan_ref.PlanRef`; `tests/test_blocks_plan.py`, 7 cases incl. odd
  two_j12, with and without slot reuse); the per-level Cauchy tables are kept as row/column pole positions
  and expanded on the GPU per level (dense they would be 13.9 GB at order 152).  2j12 = 2j43 = 8, slot
  reuse (`bench_plan.py`, laptop / 3090 box):

  | order, j_max | build | peak RSS | index tensors | tasks | physical slots | derivative constants (R) |
  |---|---|---|---|---|---|---|
  | 60, 60 | 0.4 s (was ~265 s) | 0.8 GB | 0.05 GB | 0.70 M | 0.29 M | 1.8 s (19) |
  | 130, 90 | 4.5 s | 1.8 GB | 0.35 GB | 5.1 M | 2.1 M | 8.9 s (43) |
  | 152, 100 | 6.3 s | 2.4 GB | 0.53 GB | 7.8 M | 3.3 M | 13 s (51), 1.9 GB on the GPU at 896 bits |
* Constants (GPU, once per operator): table of all M matrices, Q values (exact rationals, sign folded
  in), table of 2/k.
* Per level: (1) gather the prime-pole residues X (nb x K x S^2 B); **Cauchy step** H = C X + h_inf with
  C = 2/(D' - D_i): either a CUDA-core kernel that uses that the coefficients are 2/integer (each term is a
  multiprecision division by a one-limb integer via a precomputed reciprocal: O(L) per term), or the batched
  RNS/INT8 GEMM (cost dominated by the per-entry CRT); `auto` picks per level (kernel for K <= 128 poles).
  (2) **pole step** res = Q L H R^T: two batched S x S GEMMs on CUDA cores (schoolbook limb products,
  exact accumulation in 2L+2 limbs, one rounding), or q (L (x) R) as a batched 25x25 RNS GEMM, or
  elementwise.  Errors are normwise per column, like blocks_3d's floating-point sums.
* Validation: TTTe-like task (2j12 = 2j43 = 4, Delta_43 = -1.5874, q4 = (1,2,1,0), order 12, j_max 10,
  256 bits) with n = 0 and n = 2 as a batch of two: all 48,685 non-noise residues agree with blocks_3d to
  2^-262 relative to the largest residue (all kernel variants); the exact port agrees with blocks_3d entry
  by entry except entries that are cancellation noise (~1e-97) in both.  Random operators (S up to 5,
  three right-hand sides): GPU = exact port to 2^-310 at 318-bit mantissas.
* Slot recycling (`Plan(reuse=True)`): a residue is read once as a Cauchy column, at level m + shift, so
  its physical slot is recycled after that level (or after its own level if it is never read; the
  derivative moments are taken right after the level that produces it).  Physical slots: 4,062 of
  10,134 at order 16, 294,860 of 699,214 (42%) at order 60, j_max 60, 2j12 = 2j43 = 8.

### write_output (`output.py`)

blocks_3d's coordinate conversion of the numerators for xt and xt_radial (what the stress-tensor programs
use): lambda_lr_to_lrho_lrhob, congruence transforms with stirling, rho_z and (Delta_1+2 != 0)
power_prefactor, zzb_to_xt; lrho_to_xt_radial.  All are scalar linear maps acting on each x^k coefficient,
and the three transforms are lower triangular, so blocks_3d's truncations to the stored triangle i + j <= O
never drop a term: zzb = M F M^T with M = P R_z S, and an input (lambda order n, log r derivative e) only
feeds the antidiagonal i + j = n + e of F.  The composite (outputs x inputs, e.g. 110 x 110 at Lambda 19,
506 x 506 at Lambda 43) is built once per (Lambda, parity, Delta_1+2) in mpmath (2 s at Lambda 19, 49-69
s at Lambda 43; shared by all tasks of a run) and applied to all spins, structures, coefficients and
tasks as one RNS GEMM.  `write_json` writes blocks_3d's per-spin JSON layout (incl. its trailing-
coefficient truncation).  A task's lambda orders can run one at a time (`Derivs.for_order` shares the
R_max constants as slices; `output.apply_per_order`).

Validation, end to end (`check_output.py`: GPU recursion from blocks_3d's dumped M and h_inf ->
numerators -> write_output -> JSON, compared per spin, structure, coordinate and derivative with blocks_3d's
JSON), TTTT j12 = j43 = 2:

| task | blocks_3d precision | output polynomials | max error / largest coefficient |
|---|---|---|---|
| q4 (1,2,1,0), sign +1, order 16, Lambda 7, Delta_1+2 = 2.3, xt + xt_radial | 256 bits | 3,444 | 2^-252.5 |
| same, sign -1 (parity 1), xt / xt_radial (separate runs) | 256 bits | 1,968 / 861 | 2^-252.5 / 2^-252.5 |
| q4 (2,1,0,1), sign +1, order 20, Lambda 11, Delta_1+2 = 6, xt + xt_radial (also one lambda order at a time) | 512 bits | 8,046 | **2^-508.7** |

i.e. blocks_3d's own precision (the 512-bit case shows the 256-bit residual is blocks_3d's rounding).
`tests/test_blocks_output.py`: vs a literal mpmath port of blocks_3d's functions, 2^-315 at 318 bits.
(Side finding: blocks_3d fails with "Unable to find transform for order=Lambda-1" when asked for xt and
xt_radial together with four_pt_sign = -1: the radial transform's order is never inserted.)

### Post-recursion stages (`derivatives.py`)

blocks_3d per (spin j, parity, structure), w[k][n] = n^k r_c^n, R = Lambda - n_lambda, poles i of spin j
at full order (positions D_i, shifts s_i), x = Delta - ub_j:

* `residues_to_derivatives`: c[k] = sum_m w[k][m] h_inf[m], v_i[k] = sum_m w[k][m + s_i] res[m][i].
  On the GPU the moments v are **accumulated level by level** during the recursion (hook after each
  level): y = res r_c^n (one product), then v[k] += n^k y for k <= R by one-limb multiplications in a
  CUDA-core kernel.  No residue has to survive its last Cauchy use.  c is one GEMM.
* `restore_r_delta`: g_e = sum_{d<=e} C(e,d) Delta^(e-d) (c[d] + sum_i v_i[d]/(Delta - D_i)) =
  P_e(Delta) + sum_i a_i[e]/(Delta - D_i), with a_i[e] = sum_d C(e,d) D_i^(e-d) v_i[d] (batched
  (R+1)^2 GEMMs over (spin, pole): O(R^2) per pole instead of blocks_3d's O(R^3) polynomial divisions)
  and P_e[t] = C(e,t) c[e-t] + sum_{u>t} C(e,u) G[e-u][u-t-1], G[d][w] = sum_i D_i^w v_i[d] (one
  batched GEMM with K = poles, then one fixed binomial matrix).
* `shift_fraction` + `together_with_factors`: numerator_e(x) = P_e(x + ub) Pi(x) + sum_i a_i[e] Pi_i(x)
  is **one batched GEMM per spin chunk** with A_j = [Pi(x)(x + ub)^t, t <= R | Pi_i(x)] (K = R + 1 +
  poles), built exactly in integers (in y = 2x the roots are integers <= 0: products and exact synthetic
  division) and rounded once.  Pole shifting (kept_pole_order < order) is not ported: the stress-tensor
  programs use kept_pole_order = order.
* Validation (`check_derivs.py`, recycled slots, h_inf from the dump): TTTT j12 = j43 = 2, q4 =
  (1,2,1,0), order 16, Lambda 7, j 0..10, blocks_3d at 256 bits, GPU 318 bits, n = 0, 2, 4, 6 as one
  batch: all 2,460 numerator polynomials agree to **2^-261.3 relative to their largest coefficient**
  (2^-253 per coefficient above 2^-64 of the largest); q4 = (2,1,0,1), order 20, Lambda 11, j 0..12,
  512 bits (GPU 574), n = 0, 4, 10: 3,278 polynomials to **2^-506.5** (2^-500 per coefficient), i.e.
  blocks_3d's own precision.  Random operators vs the mpmath port: 2^-299 at 318 bits.

Kernel rates (RTX A3000 laptop, 926-bit mantissas, per output entry):

| Cauchy step, K prime poles | K=32 | K=128 | K=256 | K=384 |
|---|---|---|---|---|
| CUDA-core kernel (2/integer) | 15.7 ns | 51.5 ns | 110 ns | 183 ns |
| RNS/INT8 GEMM (N = 1024) | 41.0 ns | 48.9 ns | 52.5 ns | 70.9 ns |

| pole step (25 x 25 per task) | RNS GEMM, N = B = 16 / 64 / 256 |
|---|---|
| A3000 | 90 / 71 / 64 ns (at 574-926 bits) |
| RTX 3090 | 21 / 17 / 16 ns |

## 5. Memory: the real constraint at production size

Per right-hand side (one q4, one n), 896 bits, 2j12 = 2j43 = 8, from `live_memory.py`:

| | all residues (what blocks_3d keeps) | peak live Cauchy columns | derivative moments (R+1 per pole) |
|---|---|---|---|
| order 60, j_max 60 (nmax=10) | 1.3 GB | 0.6 GB | 0.5 GB |
| order 130, j_max 90 (nmax=22) | 14.3 GB | 6.2 GB | 5.5 GB |
| order 152, j_max 100 (nmax=26) | 23.5 GB | 10.0 GB | 8.9 GB |

A residue is needed only until the level that consumes it as a Cauchy column (m + shift); internal-spin
residues can be reduced on the fly (both implemented since 2026-10-21, section 4; measured at order 60:
294,860 physical slots, 0.9 GB at 926 bits, and 1.17 GB of moments for R = 19, per right-hand side) to the Lambda-n+1 moments sum_m m^k r_c^m res[m] that
`residues_to_derivatives` needs.  Even so, a 24 GB card holds one or two right-hand sides at nmax=22-26:
production needs (a) kernels that are efficient at small batch -- the CUDA-core kernels are (no
per-call conversion of the constant Cauchy matrix, no CRT), the RNS GEMM is not (at N = 25 B = 50-100
converting C costs more than the products; a fix is a precomputed residue table of the ~1100 distinct
values 2/k per alignment shift) -- and/or (b) host-RAM streaming of the slot buffer (each slot is written
once and read once as a Cauchy column: ~2 x 24 GB per right-hand side at nmax=26, i.e. ~2 s at PCIe 4
speed, overlappable).

## 6. Measured GPU performance and projection

Batched GPU recursion on a synthetic operator of the shape of a real task (random M/h_inf: the cost does
not depend on the values), 926-bit mantissas, vs the recursion stage of blocks_3d (jemalloc) on the same
problem, per right-hand side (one q4, one n), 2j12 = 2j43 = 8:

| | blocks_3d recursion (CPU core-seconds) | RTX A3000 laptop | **RTX 3090** | 3090 = CPU cores |
|---|---|---|---|---|
| order 30, j_max 30 | 10 (laptop, 512 bits) | 0.21 s (B=6, 574 bits) | **0.135 s** (B=24) | ~74 (CPU at 512 bits) |
| order 60, j_max 60 (nmax=10) | 116 at 512 bits, **142 at 896 bits** (laptop); 137 at 512 bits (3960X) | -- (6 GB) | **0.99 s** (B=6) | **~140** |

3090 kernel variants at order 30, B=24: CUDA-core Cauchy + CUDA-core pole 0.135 s, auto 0.156 s,
RNS Cauchy + RNS pole 0.186 s, first version (RNS + elementwise) 0.247 s.  At order 60 the pole step
(4.3 of 6.0 s) dominates; ~75% of it is the schoolbook small-GEMM kernel itself (alignment and
normalisation the rest), which needs 222 registers per thread at 926 bits (152 at 574) and so runs at low
occupancy, ~10-15% of the int32 peak: a warp-cooperative limb layout (or fusing gather + both products +
normalisation) is the known fix, expected 3-5x on this step.  blocks_3d's recursion is itself overhead-
bound (896 vs 512 bits costs it only 1.2x).

Check at a production order (3960X box, blocks_3d + jemalloc, 2j12 = 2j43 = 4, order 130, j_max 90,
832 bits, from blocks_3d's per-level task log): one recursion = 2.5e6 tasks, ~6,260 thread-s on 48
threads over 12 cores, i.e. **~1,600-1,900 core-s per right-hand side**, ~2.4x more than the
MAC-count model above predicts (at this size blocks_3d's tasks become memory-latency bound: each task
gathers ~300 residues scattered over ~10 GB).  The GPU cost for that shape is ~1e8 entries per
right-hand side, i.e. ~3-6 s at the measured 25-50 ns per entry: ~300-600 cores (estimate).

**Post-recursion stages** (`derivatives.py`, measured 2026-10-21 on the laptop).  Per right-hand side
(n = 0, R = Lambda), 2j12 = 2j43 = 8, TTTT q4 = (1,2,1,0) xt_radial task, blocks_3d native + jemalloc at
896 bits (its `residues_to_derivatives` + `shift` timers; `write_output` not included); GPU at 926 bits on
a synthetic operator of the same shape (`bench_derivs.py`), RTX A3000 laptop with ~2.2 GB usable next to
the desktop:

| | blocks_3d, 1 thread: derivs + shift | blocks_3d, 8 threads (wall) | A3000 | A3000 = CPU cores |
|---|---|---|---|---|
| order 30, j_max 30, Lambda 11 | 6.4 + 11.0 = **17.4 core-s** | 0.9 + 1.6 s | **0.43 s** (B=1), 0.32 s (B=4) | 40 / 54 |
| order 60, j_max 60, Lambda 19 (nmax=10 shape) | 76.2 + 134.9 = **211 core-s** | 12.2 + 21.6 s | **4.3 s** (B=1) | **~50** |

For reference, blocks_3d's recursion in the same runs: 9.6 and 129 core-s; its init (single-threaded,
per task) 4.6 and 31 s, including 7.3 s of pole products at order 60 that are computed even without
pole shifting (the GPU constants -- exact integer products in Python -- take 1.6 / 10.5 s once per
operator and R).  GPU breakdown, order 30, B=1 (B=4): moments during the recursion 0.12 (0.11),
residues a_i 0.09 (0.09), G 0.03, polynomial part 0.01-0.02, numerator GEMM 0.17 (0.08) s.  Order 60,
B=1: moments 1.45 s, the rest 2.84 s including streaming the 1.17 GB of moments back from host memory
(OFFLOAD=1: mpf.gemm's ~1 GB working set does not fit next to them on this card; not needed on a
24 GB card, where B can also grow).  The chunk of spins per final GEMM (1-8) does not matter.

What limited it (2026-10-21; (i) and part of the GEMM costs were fixed on 2026-10-22, see "Full chain" below): (i) the moments kernel reads and writes all R+1 accumulators
of an entry at every level (memory-bound; accumulating W levels' residues and adding them with a K = W
GEMM, or keeping the accumulators of a pole in registers across levels, would cut the traffic ~W-fold);
(ii) the a_i step is (R+1) x (R+1) RNS GEMMs batched over (spin, pole) -- K <= 20, CRT-bound: the CUDA-
core small-GEMM kernel fits it better; (iii) B = 1: the numerator GEMM halves from B=1 to B=4.  On a
3090 (3.5x the memory bandwidth, ~4x the A3000's RNS GEMM rate measured on the pole step) the order-60
figure should be ~1.2-1.5 s per right-hand side (~150-180 cores, estimate), i.e. about the recursion's
0.99 s; with (i)-(iii) ~0.5 s.  So the post-recursion stages no longer dominate on the GPU as they
do in blocks_3d (where they are 1.6x the recursion at order 60), but they are not negligible either.

Projection to production orders.  From order 30 to 60 the CPU cost per right-hand side grows ~12x
(it follows the multiprecision MAC count, opcount.py), the GPU cost ~7x (it follows the number of
output entries): the ratio improves ~1.6x per doubling of the order (x1.9 from order 60 to 152).  At order 130-152:
**one RTX 3090 ~ 250-300 CPU cores for the recursion with the current prototype kernels by this scaling
argument, ~300-600 by the order-130 check above, ~1000+ with a warp-cooperative/fused pole-step kernel
and residue tables for the RNS Cauchy step** (estimates, +-2x).  The later
stages (derivatives at r_c: K = order; r^Delta restoration and `together`: K = number of poles ~ 300-380;
coordinate transforms) are large-K linear maps where the RNS GEMMs are efficient (measured 4.5 G MAC/s
on the A3000 at K = 384, ~3x that on a 3090), against GMP loops with temporaries on the CPU; all but
the coordinate transforms are now ported (above), and the output should go to composites/PMP pieces
directly instead of ~TB of JSON.

**Full chain, measured 2026-10-21, optimised 2026-10-22** (`bench_chain.py`, `bench_output.py`; synthetic
operator of the shape of a TTTT task, 2j12 = 2j43 = 8, xt, one task = all its lambda orders, run one lambda
order at a time (B = 1), numerators staged on disk between the stages (staging not counted); blocks_3d
native + jemalloc, one thread, same shape, Delta_1+2 = 6, its stage timers).  Per right-hand side (one task,
one lambda order; averages over the task's lambda orders), "first version -> optimised":

| stage | blocks_3d, order 52, j_max 52, Lambda 19, 896 bits (core-s) | RTX 3090, same shape, 926 bits | RTX 3090, order 60, j_max 60, Lambda 19 (nmax=10) | RTX 3090, order 130, j_max 90, Lambda 43, 862 bits (nmax=22) |
|---|---|---|---|---|
| init (per task in blocks_3d; per operator on the GPU) | 2.1 (21 per task) | 4.2 s | 5.1 s | 22 s + 67 s write_output map (per run) |
| h_inf | 2.3 | not ported (blocks_3d's, dumped) | | |
| recursion (Cauchy + pole) | 73.6 | 0.57 -> **0.47** | 0.87 -> **0.69** | 8.35 -> **4.99** (4.3 + 4.0 -> 1.85 + 3.14) |
| derivatives (moments + numerators) | 23.6 + 47.4 (shift) | 0.76 -> **0.33** | 0.92 -> **0.45** | 7.2 -> **3.2** (avg.; at R=43: 5.5 + 12.7 -> 2.8 + 3.1) |
| write_output | 121.4 (incl. 7.5 GB of JSON per task) | 0.33 -> **0.25** (GEMM only) | 0.27 -> **0.25** | 1.9 -> **0.63** |
| **total (without init, h_inf)** | **266** | 1.66 -> **1.05** (= **254 cores**) | 2.06 -> **1.39** | ~17.5 -> **~8.8** |
| peak GPU memory | | 2.6 GB | 3.3 GB | 19 -> 21 GB |

On the laptop A3000 (whose free memory next to the desktop shrank to ~2.1 GB, so order 48 is now its
largest): 3.29 -> 2.51 s per right-hand side at order 48, and 4.28 s at order 52 in the first version (= 62
cores).  The order-130 figures are lambda orders n = 0, 22, 42 measured (recursion 4.99 s each; moments
2.8 / 1.5 / 0.6 s; numerators 3.1 / 1.3 / 0.3 s) and averaged over the 22 lambda orders by linear /
quadratic fits in R; write_output from `bench_output.py`.  24 GB holds order 130 at B = 1 only (21 GB:
residue slots ~6 GB, moments 6.7 GB, moment staging 1.2 GB (window 8; 4 halves it), constants ~2 GB, GEMM
working sets).

What the optimisation was (2026-10-22, profiles in the commit messages):
* **RNS GEMM alignment**: mpf.gemm's Triton kernel cannot vectorise its int8 loads when K is not a
  multiple of 16 (K = 154: 113 ms vs 38 ms padded, the kernel itself 36x slower) and pays partial tiles in
  M, N.  The blocks code pads K (and M, N >= 64) to multiples of 32 for K >= 64 (`recursion_gpu.gemm`):
  numerator stage 2.7-4x, write_output 1.6-3x.  Small products stay unpadded (padding costs more there).
  **The solver calls the same mpf.gemm** (sdpvibe/solver.py, not touched here): its shapes may profit too.
* **Cauchy step at B = 1**: the CUDA-core kernel beats the RNS GEMM at every level when S^2 B <= 128
  (order 90: 0.56 vs 0.81 s with the old K <= 128 rule, 2.14 s all-RNS); 1/d from a reciprocal table.
* **Pole step**: q L tabulated once per distinct (L, Q) pair of the plan (86k pairs for 5.1M tasks at
  order 130) and the constant operands aligned once (order 90: 1.58 -> 1.23 s).  Tried and dropped: a
  sign-magnitude kernel with fewer registers (same speed), a fused gather + align kernel (same speed).  The
  schoolbook small-GEMM kernel itself is now ~45% of the recursion: a tensor-core (RNS) or warp-
  cooperative formulation is the remaining big step.
* **Moments**: residues of 8 consecutive levels staged, then one flush kernel (thread per (row, column,
  k), exact n^k from a table, the 8 products summed in a fixed-point window with one rounding): 2x.

What this means per point (blocks part only; Haskell section 7).  CPU per right-hand side from their
statistics: TTTe at nmax=22 = 980 core-h / (45 tasks x 22 lambda orders) = ~3,600 core-s (smaller
structures than the 2j12 = 2j43 = 8 shape timed above, which is therefore an upper bound for the GPU);
blocks_3d's own stage split at order 52 (above) has the post-recursion stages and JSON at 2.6x the
recursion.  With ~8.8 s per right-hand side at nmax=22 on a 3090 (B = 1): **one 3090 ~ 400 CPU cores for
the whole blocks_3d chain** at nmax=22 (254 at order 52, measured end to end; estimates below are +-1.5x).

| | CPU now (their stats) | rented CPUs ($0.005-0.01/core-h) | 1 RTX 3090 (~$0.2/h), measured chain, ratio ~400 | 3x 3090 home box |
|---|---|---|---|---|
| nmax=22 per point | 1.5k core-h | $7-15 | ~3.6 h (~$0.75) | 80 points: ~4 days |
| nmax=26 per point | 3.6k core-h | $18-36 | ~6.5-9 h (order 152 not measured) | |
| nmax=26 TTTT (once) | 18.4k core-h | $90-180 | ~35-45 h | |
| 24-core home CPU only, nmax=22 | 62 h per point | | | 80 points: ~200 days |

Remaining at nmax=22 (3090, per right-hand side): pole step 3.1 s (the small-GEMM kernel), Cauchy 1.9 s,
moments 1.7 s, numerators 1.5 s, write_output 0.6 s.  Next: a tensor-core pole step (RNS with the
constant q L / R residues precomputed), and B = 2-4 on 48-96 GB cards, which amortises the GEMM-bound
stages (numerators, write_output) and the per-call overheads.

For comparison, the SDP solve of one nmax=22 point on a 3090 is ~10-60 h (SOTA_PLAN.md: 1900-2600 s
per iteration, 20-90 iterations): with GPU blocks, problem generation becomes a minor part of the GPU time; without them it is the bottleneck by
far on any single machine.

## 7. Haskell stages (ttoBlock, tsMixedBlock, partChunk, part)

From reading stress-tensors-3d and its dependencies at the pinned snapshot (blocks-3d, bootstrap-math,
hyperion-bootstrap, sdpb-haskell) plus the task statistics and our nmax=6 logs.  All arithmetic is MPFR
(`rounded`), polynomials are boxed vectors, D = numerator degree ~ 2.5 order + Lambda ~ 430 at nmax=26.

* **ttoBlock** (`writeCompositeBlockTable`, blocks-3d `CompositeBlock.hs`): a TTO x TTO composite is a sum
  of 13 x 13 = 169 blocks_3d tables (grouped into ~25 (j12, j43) tables) times `BlockRF` coefficients
  (polynomial of degree <= 3 in Delta over (Delta - j)^3).  Point-independent, stored centrally (one-time,
  1.0k core-h at nmax=26).  **Parse-bound**: blocks_3d JSON is read at ~4.6 MB/s per core (~50 us per
  240-byte number: attoparsec rational -> Scientific -> Rational with gcd -> mpfr_set_q); the arithmetic is
  ~10-15% of the runtime; files are re-read 3.2x on average (44k reads of 14k files at nmax=6).
* **tsMixedBlock**: the same code with T-sigma-O coefficients quadratic in Delta_sigma/Delta_eps and
  tables with Delta_12/43 != 0; per point (127 core-h at nmax=26), parse-bound as well.  TTO x SO3
  composites (TTss, TTee) are in fact point-independent but keyed by the external Delta.
* **partChunk** (`writePartChunk` / `continuumConstraint_`): one PMP piece (one crossing equation, one
  spin/parity sector).  Per block term: decode the whole composite file, restrict to the needed
  derivatives, **`DR.shift` by the twist gap 1e-6 = Horner composition p(x + y), O(D^2) per polynomial**,
  rescale, multiply by OPE coefficients, sum over common denominators.  555 core-h over 8.8k chunks at
  nmax=26 (mean 228 s, max 1.9k s, up to 6.3 GB); a ~55 s floor per chunk is decoding whole composites;
  growth nmax 22 -> 26 is 2.2x, as predicted by the O(D^2) shift.  The TTTT equations' chunks (~28%) are
  point-independent but written per point.
* **part** (`encodePart`): concatenates chunks and writes the PMP JSON through `Scientific.fromFloatDigits`
  (Integer arithmetic per digit string; off-diagonal entries twice); 43 core-h; pmp2sdp then re-parses it.
* Wall-clock: in the nmax=6 run (128 threads) the Haskell stages took ~9 min vs ~57 min of blocks_3d;
  composite tasks ran only ~62 at a time, throttled by their memory estimate (3x the largest table).
  Without statistics the default PartChunk estimate (1 GB, 20 s) badly underestimates nmax >= 22.

Fixes, with estimated gains (inferences from code + statistics):

| change | stage gain | confidence |
|---|---|---|
| shift once after summing (or shift the prefactor / sample at x + y in pmp2sdp; or one GEMM with the binomial Toeplitz matrix; or truncate the Taylor series of the 1e-6 shift at ~60 terms) | partChunk 3-4x | medium-high |
| binary block files (mantissa + exponent) or a C parser (mpz_set_str ~1.5 us/number) | ttoBlock/tsMixed 10-30x | high |
| parse each table once for all label pairs; composite assembly as a GEMM | 3-5x more | medium-high |
| shared content-addressed store for point-independent tables/chunks (TTss/TTee tables, TTTT chunks; the hash is already in the chunk name) | -190 core-h blocks, -28% partChunk per point | high |
| decode only the needed derivatives | partChunk ~1.3x | medium |
| binary PMP from `part` | part 5-10x, faster pmp2sdp | medium |

Net: per-point Haskell work ~725 -> ~100 core-h at nmax=26 (~314 -> ~45 at nmax=22).  With the
blocks on GPUs, the remaining Haskell work is small but its JSON round trips (blocks_3d -> Haskell ->
PMP -> pmp2sdp) become the next bottleneck; the long-term fix is to produce composites and PMP pieces
directly from the GPU block engine (all of it is linear algebra on the same data).

## 8. Plan

In order of value per effort.  Effort figures are rough.

1. **Done: jemalloc for the official images** (`vast/st3d_setup.sh`).  Rerun the real nmax=10 point:
   blocks should now take ~15-30 min instead of ~7 h on 64 cores, so the real-shape SDP finally gets
   measured.  Pull hyperion logs periodically to time the Haskell stages there (memory note on vast runs).
2. **Cheap CPU-side fixes in blocks_3d** (C++, 1-3 days, upstreamable):
   * one process per (j12, j43, Delta) group computing all its q4 and n: the M-matrix/pole-shifting init
     (single-threaded; 30-45 of 74-108 s in the nmax=10 task) is computed once instead of 22 times, and
     the threads stay busy (tasks scale poorly beyond ~8 threads today: 37 s on 48 threads vs 32 s on 8);
   * `restore_r_delta`: O(R^2) per pole instead of O(R^3); `together_with_factors`: one matrix product per
     spin for all derivatives; binary output (mantissa + exponent) next to JSON.
3. **Hyperion/Haskell** (2-5 days; section 7): shared store for point-independent tables and chunks,
   parse once and in C, shift once.  ~725 -> ~100 core-h per point at nmax=26.
4. **GPU block engine** (2-4 weeks, in `blocksvibe/`), driven by the plan/kernels prototyped here:
   * production plan builder -- **done 2026-10-21** (vectorised, 6.3 s at order 152) -- and constants
     (M matrices, h_inf) from blocks_3d's own init (via the dump patch, or a small C++ library entry
     point; h_inf is the one blocks_3d stage not ported, 2.3 core-s per right-hand side at order 52);
   * memory: free Cauchy columns after use and accumulate derivative moments on the fly (section 5)
     -- **done 2026-10-21** (`Plan(reuse=True)`, `Derivs.accumulate`); B = 1-2 right-hand sides per
     24 GB card at nmax=22-26, or host-RAM streaming for larger B (the moments can already be moved to
     the host before the final GEMMs, `Derivs.offload`);
   * kernels: warp-cooperative, fused pole step (gather + q L H R^T + normalise; today 222 registers per
     thread at 926 bits), residue tables for the RNS Cauchy step at small B; the post-recursion
     stages up to the `Log_r_Derivatives` numerators are **done 2026-10-21** (`derivatives.py`,
     validated against blocks_3d; next: the moments kernel and small-K steps of section 6, pole shifting
     if kept_pole_order < order is ever needed, and write_output's coordinate transforms, which are
     fixed linear maps on the numerators too);
   * write_output -- **done 2026-10-21** (xt, xt_radial; blocks_3d-compatible JSON writer, validated);
     next: output straight into composite blocks / PMP pieces (no JSON; blocks_3d's write_output with
     JSON is 46% of its time at order 52), validated against SDPB results at nmax=6-10;
   * measured chain (section 6), after the 2026-10-22 optimisation: ~3.6 h of one 3090 per nmax=22 point
     (vs 1.5k core-h), TTTT at nmax=26 ~1.5-2 days of one 3090 (vs 18k core-h); remaining kernel step:
     a tensor-core pole step (the schoolbook small-GEMM kernel is ~45% of the recursion).
5. **Precision experiment** (cheap, any time): blocks at reduced precision (e.g. 300-500 bits instead of
   832), padded, and compare SDPB's verdicts/functionals at nmax=6-14.  If the bounds do not move, GPU
   block costs drop ~2-3x (the kernels cost ~L to ~L^2 in the number of limbs; blocks_3d itself only
   ~1.2x, it is overhead-bound) and Delta_eps interpolation of the TTTe family over an island becomes
   worthwhile.

What was measured here (2026-10-20): allocator effect (laptop and 3090 box), stage splits of
blocks_3d at orders 30-90, GPU recursion correctness (vs an instrumented blocks_3d and vs an exact
port) and speed (A3000, RTX 3090), memory per right-hand side (exact counts).  Not measured:
blocks_3d stage split at order 130-152 with all spins (only the cases in section 2), the
post-recursion stages on the GPU, the GPU recursion at order 130-152 (the Python plan builder is too
slow there), an end-to-end GPU-generated PMP.  RTX 3090 rental for these tests: 1.8 h, $0.51 total.
Added 2026-10-21 (laptop only, no rental): GPU post-recursion stages validated against blocks_3d's
numerators (orders 16-20) and timed at orders 30 and 60 against single-threaded blocks_3d (jemalloc);
not measured: these stages on a 3090, at B > 1 at order 60, or at production orders.
Added later on 2026-10-21: vectorised plan builder (identical tensors), write_output on the GPU validated
against blocks_3d's JSON, full chain timed on the laptop (order 52) against single-threaded blocks_3d and
on a rented RTX 3090 (orders 52, 60, 130; EPYC 7K62 box, 0.53 h, $0.12, so $0.63 in total for these tests).
Not measured: order 152 chain, B > 1 at production orders, h_inf on the GPU, an end-to-end GPU-generated PMP.
2026-10-22: chain optimisation, measured on the laptop and on two more 3090 rentals ($0.26, of which $0.18 on
a host whose network failed; $0.89 in total for these tests).  Raw logs: results/vast_chain_opt/.
Raw logs: results/vast_chain/ (not in the repository).
Raw logs of the rental are not in the repository (results/ is ignored); the numbers above are copied
from them.
