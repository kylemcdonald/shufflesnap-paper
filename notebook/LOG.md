# Lab notebook

Chronological record of decisions, observations and negative results. Times are local
(2026-09-23 onward). Commands that produced numbers are in `experiments/`; raw outputs in
`results/raw/`.

## 2026-09-23 — setup and first observations

- Hardware: Intel i9-12900KS (8P+8E cores, 24 threads), 62 GB RAM, Ubuntu 24.04, Python 3.12.3,
  GCC 13.3. (GPU present but unused.)
- Comparators audited from source before use:
  - RasterFairy PyPI 1.0.6 (header says v1.0.3): uses `np.float` (removed in NumPy >= 1.24, so it
    fails on NumPy 2 without a shim); sorts by `xy.astype(int)` (integer-truncated coordinates), so
    inputs must be pre-scaled to a large range; with a mask holding more cells than points it
    fills the surplus by *randomly* masking cells (`autoAdjustCount`); `rfoptimizer.py` is
    Python-2 syntax and builds a dense N x N `cdist` matrix. Upstream GitHub master (2026-07-08)
    has newer commits including a "scale-normalization fix" — compare both.
  - BSP-OT: official binding `bspot` 1.2.0 (PyPI, nanobind). `compute_matching` requires equal
    counts; `compute_partial_matching` handles |points| < |cells| (injective). The binding always
    runs the cycle-merge step (`MergePlans(..., cycle=true)`); upstream's wrapper disables it for
    N >= 500k. Upstream demo defaults: 16 plans, Gaussian splits; paper figure uses 64.
- SciPy `min_weight_full_bipartite_matching` (LAPJVsp) on a 10k-point k=16 nearest-cell graph ran
  > 8 minutes without finishing (killed). The plain kNN graph turned out to be *infeasible* (some
  cells are nobody's k-nearest), which OR-Tools reports immediately. Candidate graphs must contain
  a feasible matching (we add the lattice neighbourhood of a starting assignment).
- Certified exact solver (OR-Tools min cost flow on candidate edges + SPFA potentials + global
  pricing/Lagrangian bound) matches dense `linear_sum_assignment` to all printed digits at
  N = 900, 2500 (uniform and Gaussian); certified gaps ~1e-10.
- First baseline quality (random start, 6x6 windows): excess over certified optimum
  uniform 0.69% (N=900), 1.38% (N=2500), 1.38% (N=10k); Gaussian 0.009-0.011%.
  Absolute excess per point is similar (~0.015-0.024 grid^2/pt): the excess is a local effect.
- **Failure found:** points placed exactly on a 37x23 lattice (optimum = identity, cost 0).
  Random-start baseline ends at cost 102 (seed 0), 0 (seed 1), 80 (seed 2): closed loops of points
  each displaced by one cell (all displacements 1 or sqrt 2). Every window sees a cyclically
  shifted loop as locally optimal unless the whole loop fits in the window. Bisection / row-sort
  starts recover the identity exactly.

## 2026-09-23 — exact reference, comparators, first ablations (development set = instance seed 0)

- Certified exact solver scales: 100k uniform 16 s (1 pricing round), 100k Gaussian 58–106 s (2–4
  rounds), warm-started from ShuffleSnap. OR-Tools rejects arc costs above ~2^58/(#nodes) with
  BAD_COST_RANGE (found by probing: 2^58 ok, 2^60 rejected at 200k nodes); we scale integer costs
  to 2^56/(#nodes * max cost) and bound the rounding effect by N*4/S in the certificate.
- Bug found and fixed: SPFA used "enqueued > n times" as a negative-cycle test, which is invalid
  with the small-label-first deque; it produced a false "negative cycle" on GMM instances with
  masks/free cells. Now a candidate cycle is verified by walking parent pointers. Regression test
  added (tests/test_exact.py::test_certified_on_masks_and_free_cells).
- Dense exact solvers at n=2500: scipy linear_sum_assignment 0.3–3 s; lap.lapjv and lapjv (src-d)
  2–12 s (JV's augmenting row reduction struggles with near-ties of squared-distance costs).
- RasterFairy: PyPI 1.0.6 needs a prescale (x1000) because it sorts integer-truncated
  coordinates: without it excess 193% vs 82% with it (uniform 2.5k). Upstream master (setup.py
  1.2.0, 2026-07-08) normalizes internally and solves leaves of <= 200 points exactly with scipy:
  9.2% (uniform 2.5k), 1.6% (Gaussian disk). It prints one line per leaf solve (stdout is
  captured in the adapter so printing does not dominate timings).
- BSP-OT (bspot 1.2.0, 16 plans): uniform 25–35%, GMM 2.6–4.6%, Gaussian disk 0.5–1.7% excess;
  partial (surplus cells) 14–130%.
- Window solver: Jonker–Volgenant is correct but slower than the Hungarian SSP on these windows
  (144x144 random-points: 3.7 ms vs 0.46 ms). A greedy warm start (row reduction + match each row
  to its free argmin column) is exact and 5–20% faster -> default "hungarian_greedy".
- Starts: bisection and row-sort starts are much WORSE than the random start
  (uniform 10k: 12.1% / 3.7% vs 2.0%); stratified starts (bisection, then shuffle inside BxB tiles)
  interpolate between the two (B=2: 26.7% ... B=64: 3.9% at 100k). Structured starts create seams
  that the local windows cannot repair.
- Excess decomposition (uniform 100k, baseline vs optimum): the permutation between the two
  solutions has 139 cycles; 40 cycles longer than 100 points (spatial extent ~60 cells) carry
  91% of the excess. Each point in them is shifted by about one cell: vortex-like loops. No window
  smaller than a loop can remove it, and larger strides cannot either (a one-cell shift crosses
  residue classes). Stride-2 polishing rounds after convergence do nothing (verified).
- Windows: 4x4 6.9%, 6x6 2.0%, 8x8 0.97%, 10x10 0.93%, 12x12 0.31% (uniform 10k, 1 seed).
  Mixing small coarse windows with large fine windows is worse than large windows throughout.
- Stride ratio: sqrt(2) instead of 2 improves at equal window size for little extra time
  (5 start seeds, uniform 100k: 2.71±0.55% -> 1.75±0.20% at 0.13 s; w8: 1.34% -> 0.85%).
  Ratios 3 (worse), 1.1 and "every integer stride" (worse than sqrt 2).
- Masks: the 'S' logo is the worst case for the baseline (uniform 24.9%, spiral 11%) because the
  coarse levels route mass across the concave bays; fine windows then leave long one-cell shift
  chains along the arms. w8 + ratio sqrt 2 reduces this to 0.6% / 0.2%. Free cells (M > N) also
  hurt the baseline (6–10% vs 1.3% on the exact-fit square).
- Jittered lattice (optimum is tiny, ~0.125/pt): relative excess is huge for every heuristic;
  report absolute excess per point as well.

## 2026-09-23 (evening) — reference batch, small-N check

- Small-N exactness v0 used grid kind "free" with perfect-square N (100, 400, 900, 2500), where the
  full rectangle has no surplus cells, so "free" silently duplicated "square". Kept as
  results/raw/small_exactness_v0_free_was_square.jsonl; rerun with "free10" (10% surplus).
- v0 (618 instances): certified solver agreed with dense SciPy on 618/618. ShuffleSnap reaches the
  exact optimum only when windows cover much of the grid: quality preset 100% at N=100 (one window
  covers the 10x10 grid), 63-75% at N=400, 0% at N=2500 for every preset; BSP-OT nearly never.
- Reference warm starts: references computed before the package default changed to the "balanced"
  preset were warm-started from the then-default (6x6, halving); later ones from "balanced". The
  warm start only affects the certifier's running time, never the certified optimum. reference.py
  and the timed "Exact-certified" adapter now pass preset="balanced" explicitly.
- Benchmark trimming (to keep the sequential, clean-timing run under ~9 h): certified exact with a
  bisection start only at 100k; masks/primes compare against RasterFairy master only; scaling
  runs capped at 30 min (a timeout is itself a result).
- Tried: merging independent ShuffleSnap runs (different start seeds) per connected component of
  the union of the two assignments, keeping the cheaper side (the BSP-OT merge rule). Helps but
  is dominated by larger windows: uniform 100k balanced 0.88% -> 0.33% with 8 merged runs (~2.6 s)
  vs a single 12x12 run 0.45% (0.64 s); GMM 100k 0.0053% -> 0.0032% (8 runs, ~13 s) vs 12x12
  0.0012% (5.7 s). Components of the union are large (loops of both runs interlock). Not adopted.
- Constructed fixed points (lattice instance, optimum = identity, cost 0): rotating an axis-aligned
  ring of cells by one step. Rings with side <= w/2+1 are always repaired; rings larger than the
  window are never touched at any stride of the full sqrt(2) schedule (w = 6, 8, 12; 0 moves), so
  relative excess is unbounded on this instance; a 6x6 ring even survives 6x6 windows when it
  straddles all four tilings. Added as tests/test_assign.py::test_rotated_ring_larger_than_windows_is_a_fixed_point.
- The full benchmark (experiments/run_all.sh) runs against a frozen snapshot of shufflesnap commit
  57ffcb1 (git archive + the in-place built extension), recorded in every result. The package HEAD
  moved on afterwards only to add input validation in the Python API (a840017), which does not change
  any result.
- Reference batch finished: 211/211 certified. Median wall time for the 47 instances at ~100k was
  157 s (6 in parallel), but heavily duplicated inputs are slow: duplicates-100k took 941-2036 s and
  10-12 pricing rounds; heavy tails up to 428 s. Full benchmark started 2026-09-23T21:54.

## 2026-09-24 — benchmark results (held-out seeds)

- main_t1 (946 runs) took 8.6 h (21:54-06:32), incl. three 1-h timeouts of the certified solver
  started from bisection (gmm, duplicates, near-line at 100k); a 4th bisection-started run
  (spiral) stopped after 50 pricing rounds without a certificate. All ShuffleSnap-started exact
  runs certified. All 943 completed outputs valid; all 396 ShuffleSnap traces monotone.
- main_t24: ShuffleSnap bit-identical to 1 thread in all 396 paired runs; 13-15x speed-up.
- Scaling: certified solver certified 1M uniform points in 706 s (4.3 GB); dense SciPy 50k uniform
  372 s / 23.4 GB; gmm 20k 1319 s. Peak-RSS rise under-reports ShuffleSnap on gmm (heap reuse
  after instance generation) -> added heap-trimmed memory experiment (bench.py trim_heap).
- Masks (20k): baseline 18.3% on uniform/S-mask, worse than BSP-OT-64 (13.2%); balanced 1.3%.
- Primes: full rectangle with <200 surplus cells is much harder on uniform points (baseline
  12.6-14.5% vs 2.1-3.0% exact-fit; absolute 0.15-0.16 vs 0.04 cells^2/pt).
- Real embeddings: RasterFairy's kNN recall on Fashion (0.611) exceeds the optimum's (0.606)
  despite +24.6% displacement; BSP-OT halves kNN recall and lattice precision.
- Bug (mine): the "coarse" ablation group (cap on the starting stride) failed in all 54 runs with
  TypeError because the m_shufflesnap adapter did not accept `start_stride`. Fixed the adapter
  (verified it reproduces ss.build_schedule for ratio 2), removed the 54 error records
  (original kept as results/raw/ablations_t24.with_adapter_errors.jsonl.bak) and reran that group
  after the memory and polish checks.
- Held-out ablations (uniform/gmm 100k, mnist 70k; seeds 1-3; 24 threads) confirm the development
  findings: structured starts 3-18x worse than random; ratio sqrt2 best or near-best; window side
  the main knob; a single tiling fails completely (uniform 21,000% excess), 2 tilings 5.5%, 4
  tilings 2.5%, 9 tilings 1.9% at 2x time; solvers identical in quality, JV 2-10x slower;
  skip-unchanged identical in quality, 35-40% faster.
- Correction: the greedy warm start of the window solver (default) is 5-20% faster only in the
  synthetic micro-benchmark; in held-out full runs it is up to 15% faster on uniform inputs and
  2-3% slower on clustered inputs. Kept as default (identical results), paper text corrected.
- Held-out polish check (uniform/gmm 100k, mnist 70k; seeds 1-3): three extra [stride 2, stride 1]
  cycles after the baseline changed nothing on uniform (3/3 identical) and removed <3% of the
  excess on clustered inputs; merging 4 independent baseline runs cut uniform excess 23-38% at 4x
  cost (fast preset: 1.5% at ~1x). Heap-trimmed memory: ShuffleSnap 22-37 B/pt, BSP-OT ~243,
  RasterFairy ~647, certified exact 4-8 kB/pt.

## 2026-09-25 — comparison with the other implementations; fixes; shorter editions

Compared this rebuild with the original implementation and paper (`../../shufflesnap`,
`../../shufflesnap-paper`, including its 10/6/3-page editions) and with the Fable rebuild
(`../fable`). The Codex rebuild's files no longer exist (its symlinks point to a deleted
directory on `/media/kyle/data`); only the original's review of it (`REBUILD_REVIEW.md`) was
available.

- Adopted: (1) an executable serial reference in NumPy/SciPy, like the original's
  `paper/reference.py`, as `shufflesnap.reference` (95 lines). It reproduces the kernel's
  assignments exactly: 48/48 ad-hoc cases and 24 parametrized tests (four presets x exact fit,
  partial row, surplus cells, thin rectangle, disk, random mask). Exact equality needs the
  same sequential summation order for the acceptance test (NumPy's pairwise `sum` and
  Python 3.12's compensated `sum()` differ in the last bits). (2) Codex's sparse-mask
  counterexample (usable columns 0 and 7 of a one-row mask never share a window under the
  baseline): a regression test and a paragraph in the anatomy section. Here empty cells are
  movable, so the example needs *masked* cells between the two. (3) The original's graph-paper
  lines (source-space grid lines darkened in the gallery), now in the MNIST, mask and progress
  figures (`figstyle.source_colors`). The palette itself was not brightened: even the current
  one clips 17% of colors to sRGB, and higher chroma clips 22-49%.
- Bug found and fixed: `time_budget=0` (or a budget used up by normalization and the start)
  ran the whole schedule, because the kernel treated budgets <= 0 as "none"; NaN and negative
  budgets were accepted too. Now 0 returns the start and NaN/negative raise. No benchmark run
  used a budget, so no result changes.
- BSP-OT: its binding exposes no seed; the paired 1- and 24-thread runs (198 instances) never
  returned the same cost (-8% to +13%, median 0). Fable reported a systematic degradation with
  more threads for the POT port; there is none here. Added a sentence to the setup.
- Not adopted: the original's memory-lean kernel layout (per-axis center coordinates instead
  of an M x 2 array, one dirty byte per cell instead of an int64 stamp). It matters only near
  10^9 cells, and the paper's timings would no longer describe the code. Fable's density-based
  choice of initial empty cells (Fable measured it as ineffective) and its CLI.
- Related work: added Volgenant's core-oriented LAP (LAPMOD), verified via Crossref and the
  abstract. Considered the AKT log N law for uniform matching and dropped it: the measured
  uniform optima per point are not monotone in N here (1.48, 1.30, 1.80 cells^2 at 10k, 50k,
  100k), because each size has a different grid shape and partial last row.
- Abstract: "a dense exact solver" -> "SciPy's dense exact solver" (the measured one).
- Also from the original's review: a stage with unlimited rounds and no stop rule looped forever;
  `run_schedule` now rejects it (unless a time budget bounds it), empty stage tables and strides
  below 1. No experiment used such a table.
- Shorter editions: `versions/{long,medium,short}` (10/6/3 body pages), built and checked by
  `experiments/build_versions.py` (page count, overfull boxes, undefined references); compact
  figures from `experiments/fig_editions.py`, and `tab_main_compact_100k_t1` from `make_tables.py`.
  They reuse existing results only. Proofreading against the full paper corrected several
  condensations (e.g. "5-37x worse and slower" -> "5-37x worse and 5-90x slower"; the serial
  reference is 70 code lines, not 60).

## 2026-09-25 — window solver at the balanced preset; auction solver

- The solver ablation only covered 6x6 and 12x12 at ratio 2. Rerun at the balanced preset (8x8,
  ratio sqrt2; uniform/gmm 100k, mnist 70k; seeds 1-3; 24 threads and 1 thread pinned to CPU 8):
  `experiments/solver_balanced.py` -> `results/raw/solver_balanced_t{24,1}.jsonl`.
- New window solvers: `auction` (eps-scaled forward auction, then a price-tightening pass and an
  exact shortest-path finish from the auction prices; exact, like Hungarian) and `adaptive`
  (per window: auction if >= 50% of rows collide on their cheapest column, else Hungarian with
  greedy start). Tuned (eps0 0.05, epsf 1e-3 of the cost range, factor 8) on ~22k real 8x8
  window matrices captured from the serial reference; exact on all of them vs SciPy.
- Medians vs hungarian_greedy (24 threads / 1 thread); final cost identical for all solvers:
  mnist: auction 0.49x / 0.53x, adaptive 0.50x / 0.54x, hungarian 1.01x / 1.00x, jv 6.0x / 6.1x.
  gmm: auction 0.52x / 0.55x, adaptive 0.52x / 0.56x, hungarian 0.99x / 0.98x, jv 6.5x / 6.7x.
  uniform: auction 1.35x / 1.51x, adaptive 1.09x / 1.06x, hungarian 1.14x / 1.14x, jv 2.7x / 2.8x.
- Why: clustered windows (a dense group of points far from most cells) give near-degenerate
  costs and long augmenting paths; the auction's prices remove most of them (hungarian_greedy
  ~200 us vs auction ~95 us per 64x64 MNIST window). Uniform windows are nearly sorted already.
- A first attempt at this run was discarded: a stray micro-benchmark sweep pinned to CPU 8 ran
  concurrently. Default solver unchanged (hungarian_greedy) pending a decision.

## 2026-09-26 — geometric window solver; all ShuffleSnap timings rerun

- Solver search on ~79k real 8x8 windows captured from the serial reference at the balanced
  preset (20k points of mnist/gmm/uniform, unchanged windows skipped as in the kernel; exact
  optimum per window from SciPy). Per-window time vs hungarian_greedy (mnist / gmm / uniform):
  auction 2.1x / 2.1x / 0.64x faster; plain Crouse SSP ~1.4x; affine Brenier start 2.9x / 3.4x /
  1.5x; per-axis quantile start 3.8x / 4.5x / 1.6x; quantile + column reduction + Kuhn on tight
  edges 5.1x / 5.6x / 1.9x (adopted: `geometric`). Exact on every window.
- Did not pay off: full-covariance Gaussian start (worse than quantile), duals integrated from
  the current assignment (better on clustered windows, worse on uniform; picking per window
  cost more than it saved), auction refinement of the quantile start (fewer augmenting rows but
  bids cost as much as search steps), extra row/column reduction rounds (stall after one),
  dense branch-free augmenting scan (slower), branch-free Crouse loop (2-3%, not kept).
- Why clustered windows are slow for generic solvers: in 96% of MNIST windows (>=16 points)
  the points' cheapest cells number at most half the points (uniform: 1%).
- Rectangular windows: padding with zero-cost rows made uniform grids with 50% surplus 11x
  slower (one augmenting path per empty cell). Fix: windows with empty cells use the
  rectangular v=0 path unless more than half their rows collide (GEO_RECT_COLLIDE = 0.5).
- `adaptive` (auction or greedy per window) was dropped: geometric beats it everywhere.
- Rerun of every ShuffleSnap-dependent run (incl. exact solver seeded by ShuffleSnap) against a
  frozen snapshot of the working tree (`../frozen/shufflesnap-geometric-20260925`, base
  860598c + worktree.patch; identical to the working tree incl. the .so): `experiments/
  run_geometric.sh`, 2 h 27 min. Results go to `results/raw_geometric/` (comparator records
  copied unchanged from `results/raw/`, which is untouched); `SS_RESULTS_RAW` selects the set
  for analysis, figures and tables. Solver ablation extended to auction/geometric and to 8x8 at
  ratio sqrt2.
- Checks: 2016 ShuffleSnap runs valid, monotone, finished; 396/396 identical 1 vs 24 threads;
  1846/1917 runs repeating an old spec have bit-identical cost, the rest differ <= 5e-8 rel.
  (ties). Quality numbers in the paper are unchanged.
- Timing at 1e5, one thread, median: baseline 5.07 -> 0.72 s, fast 6.89 -> 1.09, balanced
  15.9 -> 2.25, quality 52.4 -> 6.37 s. 24 threads: 0.35 -> 0.064, 1.08 -> 0.17 s. Balanced
  preset solver comparison: geometric 5.0-5.2x faster than hungarian_greedy on clustered data,
  2.0x on uniform (1 and 24 threads). Derived prose numbers: `experiments/paper_numbers.py`.
- Memory: ShuffleSnap now measures 55-67 B/pt (52 MB at 1e6), up from 22-37. Not the solver
  (hungarian_greedy and geometric both 52.2 MB on the same code): the old memory table was
  measured on 57ffcb1, before a840017 added `np.unique` validation of the start, which the main
  benchmark already used. Possible follow-up: validate the start without N-sized temporaries.
- small_exactness rerun: ShuffleSnap columns identical; BSP-OT columns differ (unseeded), so
  the paper's hand-written table keeps its original BSP-OT numbers.
- Paper: new method subsection (geometric solver), all timing statements updated in the full
  paper and the three editions (still 10/6/3 pages). Compile time on one core 7.8 -> 8.1 s.

## 2026-09-26 — presets re-chosen with the fast solver; baseline preset removed

- Development search redone (`experiments/dev_ablation.py --geometric`, seed 0, 23 instances x
  35 configurations x 5 start seeds; `results/raw_geometric/dev_ablation_geometric.jsonl`):
  window sides 6-24 x ratios 2, 1.68, sqrt2, 1.19, plus two rounds per level for w 8/12/16.
  Geomean excess vs geomean time relative to 8x8/sqrt2: no knee; each window step (8->12->16)
  costs 2-2.6x the time and removes 1/2-2/3 of the remaining excess; sqrt2 on or next to the
  Pareto front at every window size (clearly best at 24x24). 20x20 and 24x24: 12-21x the time.
- New presets: fast 8x8, balanced 12x12 (default), quality 16x16, all ratio sqrt2. `baseline`
  removed (6x6 halving stays available as window=6, schedule="halving"). Kernel window cap
  16 -> 32. Serial reference defaults to the new default. Tests: mask-disconnect example
  rebuilt for 12-wide windows (columns 0 and 13; 16-wide repairs it); rotated-ring fixed point
  extended to w=16.
- Rerun of every ShuffleSnap-dependent spec (`experiments/run_presets.sh`, 4 h 10 min) against
  `../frozen/shufflesnap-presets-20260926` into `results/raw_presets/` (comparators copied
  from raw_geometric; ShuffleSnap-seeded exact runs rerun because they now start from the new
  default, although their spec ids did not change). 1714 runs, all valid/monotone/finished;
  297/297 identical 1 vs 24 threads. The working tree differs from the snapshot only by a
  docstring in api.py.
- Ablations, polish check, solver comparison and small-exactness re-centred on the default.
- Held-out results at 1e5 (default): within 0.02% on 9/11 distributions (uniform 0.44%,
  jitter 7.4%), median 6.4 s on one core, 0.50 s on 24 threads; better than BSP-OT-64 on 11/11.
  Real embeddings: within 0.001%. S mask (uniform) 0.13% (fast 1.3%). Surplus <1 row: 1.4%.
  Polishing (V-cycles) changed nothing on all 9 instances; merging 4 default runs halves the
  uniform excess at 4.0x the time, the quality preset does better at 2.0x.
- Changed claims: the certified solver from the new default took 21 min / 4.9 GB at 1e6 uniform
  (12 min before) and 8 s-15 min at 1e5 (median 131 s). On uniform points the default's
  excess is not monotone in N (0.10, 0.20, 0.71, 0.58% at 1e3-1e6, seed 1); the fast preset's
  grows (0.11 -> 1.33%). Stride ratio at 12x12 held-out: sqrt2 best on gmm, within 10% of
  the best on uniform (1.1) and MNIST (1.68).
- Lessons from the 3-page edition adopted in the full paper and the 10/6-page editions: the
  default preset is presented as the method (no baseline/halving narrative; halving appears
  only as an ablation setting and as the mechanism example in the failure figure); the
  certified solver is framed as exact on the dense problem and compared with SciPy where it
  fits (720 small instances and all 11 distributions at 1e4, max rel. diff 8e-11); one unit
  per comparison sentence (no "x times further from optimal" next to percentages); figures
  show only the presets and comparators (no baseline, no bisection); author Kyle McDonald in
  every edition. Edition figures unified (fig_schematic/fig_progress/fig_summary at the default).

## 2026-09-26 — presets time-matched to BSP-OT

- User request: the former fast preset (8x8) becomes balanced, a new fast preset should take
  about as long as BSP-OT with 16 plans, and balanced about as long as 64 plans, to show
  improvements at equal running time. Quality is now 12x12 (one step up from balanced).
- Single-thread sweep on development instances at 1e5 (seed 0, CPU 8): configurations at
  BSP-OT-16's time (0.71 s median) were 5x5/sqrt2 (0.67 s) and 6x6/halving (0.70 s); 5x5/sqrt2
  had lower excess on 5 of 7 instances and keeps every preset at ratio sqrt2 -> fast = 5x5.
  8x8/sqrt2 took 1.8 s against BSP-OT-64's 2.8 s.
- Rerun (`experiments/run_bspmatch.sh`, 2.5 h; snapshot `../frozen/shufflesnap-bspmatch-20260926`,
  identical to the working tree; results in `results/raw_bspmatch/`): 1714 ShuffleSnap-dependent
  runs, all valid; 297/297 identical 1 vs 24 threads.
- Held-out, 1e5, one core: fast 0.74 s vs BSP-OT-16 0.72 s, closer to the optimum on 9/11
  (behind on near-line 0.010 vs 0.008% and 90/10 clusters 0.050 vs 0.032%); balanced 2.29 s vs
  BSP-OT-64 2.85 s, closer on 10/11 (near-line 0.0022 vs 0.0021%). 24 threads: fast 0.09 s vs
  0.13 s, balanced 0.19 s vs 0.42 s. Real embeddings: fast 0.031%/0.008% vs BSP-16 3.7%/9.2%.
- Other changes: with 8x8 windows a row-sorted start beat the random start on the mixture
  (0.0011 vs 0.0023%; bisection is worse everywhere), so the paper now says bisection starts
  are worse and row-sorted starts on most inputs. V-cycles changed 2 of 9 instances (MNIST,
  <=5% of a tiny excess). Certified solver from the 8x8 default: 11 min / 4.2 GB at 1e6 uniform.
- Tables pair each preset with its time-matched BSP-OT column. Full paper and 10/6/3-page
  editions updated (page counts kept; the 10-page edition dropped its small-instance sentence
  and the 3-page edition keeps the equal-time claim on the results page only).

## 2026-09-26 — jittered lattice dropped; rectilinear S mask

- The jittered lattice was dropped from the benchmark (main tables, small-instance check,
  editions). Remaining: ten synthetic distributions; 600 small instances (five distributions).
- The font-based S mask ("logo") was replaced by the blocky rectilinear S of the earlier
  ShuffleSnap work (`symbol()` in the original's bench_rebuild_lessons.py: bars at y<.18,
  .41-.59, >.82; left bar in the upper half, right bar in the lower half), as grid kind
  "srect" (`datasets.s_mask`, exactly n cells via Grid.from_image). 12 new certified
  references (20k points x 4 inputs x 3 seeds), all certified.
- Result set `results/raw_final/` = raw_bspmatch minus jitter and logo records, plus every
  method on srect (`experiments/run_final.sh`, same frozen package snapshot). 202 reference
  instances in total.
- S mask results (uniform / spiral): default 1.4% / 0.26% vs BSP-OT-64 9.3% / 4.8%; fast
  17% / 3.3% vs BSP-OT-16 12% / 12.5% (uniform S is the one mask case BSP-OT wins at equal
  time); quality 0.88% / 0.19%.
- Failure figure (c-e) now uses 4000 points drawn uniformly inside the rectilinear S (optimum
  0.17 cells^2/pt): 6x6 halving 32, default 6.3; misrouted points line up along the bays.
  (Uniform points over the square did not show the mechanism: the optimum itself moves 36/pt.)
- Summary figure's fourth panel is now the spiral. Equal-time counts over ten distributions:
  fast vs BSP-OT-16 8/10, default vs BSP-OT-64 9/10.

## 2026-09-27 — billion-point meander with the current kernel

- Instance: the earlier work's 1e9 meander (side 31623, 1,000,014,129 points; generator ported
  to `experiments/billion.py`, bit-identical at 1e6), same start (default_rng(0) permutation).
  Balanced preset, geometric solver, 24 threads, 3600 s budget counted from the bbox scaling
  (as `assign(time_budget=)`); instance generation (57 s) and grid construction (8.5 s) excluded.
  The driver calls `_core.run_schedule` directly to avoid assign()'s point-sized temporaries;
  at 9e4 points it returns exactly assign()'s cells.
- Memory: peak 53.0 GiB (57 B/pt: P 16, centers 16, stamps 8, pos/occ/index 4 each, windows ~5).
  A file-backed memmap for the centers was tried and dropped (disk had 15 GB free: bus error).
  First attempt 2026-09-26 aborted by hand after 20 s: an unrelated 8-thread job started with it.
- Result: stopped in the stride-16 level (68 phases, ~240 s per level; every window solved at
  every level). Start 438% above the endpoint; within 1% after 11 min, 0.1% after 22, 0.01% after 34.
- Same-frame prices (grid frame; old frame is a uniform x1.1317e9 scale): new 4.777420e16;
  earlier implementation's 6 h run 4.777499e16 (+0.0016%, never reached our endpoint; we passed
  its 6 h endpoint after 44 min and its 1 h cost after 42 min); single-plan BSP-OT port 5.16695e16
  (+8.2%, 392 s, 1 thread).
- Paper: full paper gets a "A billion points" paragraph + Figure (fig_billion: 75% interpolation
  render and progress vs the 6 h run), a sentence in the abstract, intro and limitations. Long and
  medium editions get one sentence each (page counts kept; long dropped "The exact solver changes
  the question." and the exact-fit-mask sentence of its Discussion). Short edition unchanged.

## 2026-09-28 — billion-point run continued toward 6 h (converged at 4.1 h)

- Asked to extend the 1 h run to 6 h in total. `billion.py` gained `--resume/--resume-stride/
  --time-offset/--tag` (resuming at a level boundary is bit-identical to an uninterrupted run,
  checked at 1.6e5). The 1 h run had stopped inside the stride-16 level, so that level was rerun
  from its first tiling (a small deviation from one uninterrupted run).
- Segment 2 (`run_billion_6h.sh`, balanced, from stride 16): converged, fixed point at 1 h 43 min
  total (stride 1 from 1 h 30 min, 78 stride-1 rounds). Segment 3 (`run_billion_quality.sh`,
  quality preset 12x12, strides 4..1 from that fixed point): converged at 4.1 h total after 1059
  stride-1 rounds (long tail of tiny moves), gaining only 1.2e-6 %. No further segment: every
  preset had converged and larger windows would chase gains of that size.
- Final cost 4.777415931e16. 1 h result +0.000095 %; balanced fixed point +1.2e-6 %; earlier 6 h
  run +0.0017 % (passed after 44 min); single-plan BSP-OT +8.2 %. Within 1/0.1/0.01/0.001 % of
  final after 11/22/34/46 min. Peak 53 GiB (segments 2-3: 50 and 49 GiB).
- Correction: an interim chat message said the balanced continuation gained 0.00094 %; it is
  0.000094 %.
- Paper: full-paper paragraph, figure (three segments, y axis to 1e-7 %), abstract and
  limitations updated; long and medium editions' sentence now reads "converged in 1 h 43 min ...
  passing a six-hour run of an earlier implementation after 44 minutes". Short edition unchanged.
