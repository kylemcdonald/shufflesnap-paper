# Random cell-group experiment

This is an isolated research experiment. It does not change the package, paper,
publication figures, or existing benchmark data.

## Implementations

All variants begin with exactly the same NumPy seed-0 random bijection unless an
initial-seed control is explicitly selected. Groups are **target grid cells**, not
nearest neighbors of the input points: each group rematches its current occupants
to those same targets. The production 36-cell JV solver and its strict improvement
threshold are reused verbatim. Disjoint groups run in parallel on 24 threads.

- `baseline`: the production power-of-two stride schedule and four phases, then
  stride-1 rounds until no changes.
- `linear1`: region counts 1, 2, 3, ..., ceil(side/6) along each axis; one fresh
  random partition into groups of at most 36 cells within every region per level.
- `linear4`: the same with four fresh random groupings per level.
- `geometric1`: region counts 1, 2, 4, ..., ceil(side/6); one grouping per level.
- `geometric4`: the same with four groupings per level.
- `shifted2` (focused follow-up): two geometric passes with unshifted and
  diagonally half-shifted partitions.
- `shifted4`: geometric region counts, four passes per level with partition
  origins (0,0), (0,half), (half,0), (half,half). Region edges are clipped;
  cells never wrap. Every pass covers every cell exactly once.

Every random variant finishes with **the same production stride-1 cleanup**.
Final regions have at most 6 rows and 6 columns, hence are dense exact solves.
Regions with more than 36 cells have a possible smaller final group. Fixed-region
variants repeat identical full-region solves at the finest level; those extra
passes are deliberately retained as part of their stated schedules.

Additional controls: `noshuffle` omits the index shuffle from the shifted
construction; `cycles` repeats the full default schedule twice before polishing.
`local` runs only production stride-1 cleanup from the random
start; `alloffsets` uses all 36 offsets at every production scale and during
convergence. The latter is an experimental control, not the package's optional
single final all-offset sweep.

## Measurement

`build.py` extracts the native core from the pinned sibling source, omitting only
nanobind bindings, and compiles it together with the driver using `-O3`, C++20,
and pthreads. Neither production source nor the installed module is modified.
The baseline and proposals run in the same executable. Each timed run includes
state setup, group allocation/enumeration, Fisher–Yates shuffling, cost
construction, exact solves, thread creation/join, and local cleanup. Input point
construction, the common initial permutation, process launch, and assignment
file output are excluded for all methods. Trace bookkeeping is enabled for all.

`grouping` includes region construction, index allocation, shuffling, and group
boundaries. `solving` includes costs, JV, application, and thread overhead.
`polishing` is the final call into production cleanup: **the entire multiscale
solve for baseline**, but only stride-1 cleanup for random variants. Component
medians need not add up to the median total. Trace timestamps inside production
cleanup omit its small setup/copy overhead; total runtime includes it. Cleanup caps are checked between complete rounds; inspect `converged`. The
initial full/focused sweeps used a 30-second cap (every run converged). Subsequent
stages allow 120 seconds, except local-only and no-shuffle controls capped at 30 seconds.

Full comparison: five families × three input instances × five grouping seeds,
with five timing repetitions of the deterministic baseline, in randomized order.
Runs are sequential to avoid timing interference. Large-grid comparisons use
meander inputs and compare costs directly, **not gaps to an exact optimum**.
The same-initialization control varies initial permutations independently of the
random grouping seed. Every saved result checks bijectivity, independently
recomputes its cost, and verifies non-increasing trace costs.

## Coordinate update

The recorded experiment predates margin removal and uses the inset coordinates
of ShuffleSnap 0.3.0. Its measurements are retained unchanged and are not part
of the paper. The driver now follows the unit-square API for future runs; use
the revisions in `environment.json` to reproduce the historical measurements.
Do not compare old absolute costs directly with new unit-square costs.

## Reproduction

From the paper repository, after installing the scientific requirements:

```bash
.venv/bin/python experiments/random_groups/build.py
.venv/bin/python experiments/random_groups/run.py --stage full
.venv/bin/python experiments/random_groups/run.py --stage focused
.venv/bin/python experiments/random_groups/run.py --stage controls
.venv/bin/python experiments/random_groups/run.py --stage scale
.venv/bin/python experiments/random_groups/run.py --stage init
.venv/bin/python experiments/random_groups/run.py --stage ablations
.venv/bin/python experiments/random_groups/validate.py
.venv/bin/python experiments/random_groups/analyze.py
.venv/bin/python experiments/random_groups/analyze_controls.py
```

Run stages sequentially. Generated native source/binary are in ignored `build/`.
`environment.json` records the solver revision and compiler. JSONL files retain
all checkpoints; CSV summaries reduce grouping seeds per instance before
summarizing instances. Threshold-time summaries must be read with hit counts:
unsuccessful runs are missing values, not successes at an averaged time.
