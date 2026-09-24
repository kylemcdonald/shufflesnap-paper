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
