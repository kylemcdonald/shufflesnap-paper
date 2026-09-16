# Exact Solver Implementations

This folder contains standalone research implementations used by the paper.

These are intentionally separate from the published `shufflesnap` package. The goal here is to support paper benchmarks and algorithm comparisons, including implementations that are useful for analysis but are not part of the package API.

## C++ exact grid solvers

`cpp/exact_grid_lap.cpp` implements two numerical reference solvers for square squared-Euclidean point-to-grid assignment:

- Jonker--Volgenant (`--solver jv`)
- Auction (`--solver auction`)

Each solver supports two cost-access modes:

- `--cost-mode dense`
  Builds the full dense cost matrix first, then solves the assignment.

- `--cost-mode jit`
  Computes squared Euclidean distances just in time during the solve instead of storing the dense matrix.

The auction implementation parallelizes bidder value scans. The dense modes also parallelize matrix construction. The Jonker--Volgenant augmenting-path core is otherwise sequential in this implementation.

## Manual build

From the repository root:

```bash
c++ -O3 -std=c++20 -pthread solvers/cpp/exact_grid_lap.cpp -o solvers/cpp/exact_grid_lap
```

## CLI

```bash
./solvers/cpp/exact_grid_lap \
  --points /path/to/points.bin \
  --width 64 \
  --height 64 \
  --solver jv \
  --cost-mode jit \
  --threads 8
```

The points file is a raw binary array of `float64` point coordinates with shape `(n, 2)` in C order.

## Auction accuracy

The auction uses epsilon scaling down to `max(1e-12/n, ulp(cost_scale))`. The saved benchmark runs do not include independently checked dual certificates. Comparisons with SciPy and dense JV support its use as a high-accuracy reference on the tested small instances, not a formal floating-point certificate at all scales.

## BSP-OT stress-test provenance

`benchmarks/results/billion_bspot_31623_seed0.json` retains summary measurements for the official one- and 16-plan implementations at 100 million points, and the memory-lean single-plan port at a billion points. The 100-million-point official 16-plan memory measurement used four OpenMP threads (the recorded `omp4` field). Original stdout logs, exact upstream revision, and the complete invocation history were not retained. These summaries support the qualified appendix comparison; they do not establish equivalence between the randomized official and ported implementations. Their one-plan costs differ by about 3%.

For future reruns, pin and record the upstream commit and Eigen version, compiler version, `OMP_NUM_THREADS`, the raw driver stdout, `/usr/bin/time -v` stderr, input seed, and source hashes. An example invocation (not a recovered historical command) is:

```bash
OMP_NUM_THREADS=4 /usr/bin/time -v ./solvers/cpp/bspot_grid \
  --points /path/to/normalized-10000x10000-seed0.float64.bin \
  --rows 10000 --cols 10000 --plans 16 --seed 0
```

The binary input must be the exact normalized point array shared by all compared runs. Build the separate port using:

```bash
c++ -O3 -std=c++20 solvers/cpp/bspot_grid_giga.cpp -o solvers/cpp/bspot_grid_giga
```
