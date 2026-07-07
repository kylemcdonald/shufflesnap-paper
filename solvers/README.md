# Exact Solver Implementations

This folder contains standalone research implementations used by the paper.

These are intentionally separate from the published `megalap` package. The goal here is to support paper benchmarks and algorithm comparisons, including implementations that are useful for analysis but are not part of the package API.

## C++ exact grid solvers

`cpp/exact_grid_lap.cpp` implements two global exact solvers for square Euclidean point-to-grid assignment:

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
