# ShuffleSnap paper

Source, benchmark results, and figures for **ShuffleSnap: Multiscale Window Descent for Large Point-to-Grid Assignment**.

The native library is in [kylemcdonald/shufflesnap](https://github.com/kylemcdonald/shufflesnap). The manuscript describes a heuristic with linear work per round and empirically small assignment gaps; it does not claim a worst-case approximation or convergence-rate bound.

## Build the reviewer PDF and arXiv source

Install [Tectonic](https://tectonic-typesetting.github.io/), then run:

```bash
python3 scripts/build_submission.py
```

This compiles the manuscript, copies only the required sources and figures into a clean directory, compiles that copy, and writes:

- `dist/shufflesnap-review.pdf`
- `dist/shufflesnap-arxiv-source.zip` (including the generated bibliography)
- `dist/manifest.json` (artifact checksums)

No benchmark reruns or Python scientific dependencies are needed to build from the supplied figures and statistics. The source archive is intended for a later arXiv submission, after human review; it has not been submitted.

## Reproduce figures and statistics

ShuffleSnap 0.3.0 is [available on PyPI](https://pypi.org/project/shufflesnap/0.3.0/). The paper now uses the unreleased source revision below (0.4.0.dev0), which removes the coordinate margin. PyPI 0.3.0 still uses the previous inset. From this repository, with Python 3.12 and a C++ compiler:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install "git+https://github.com/kylemcdonald/shufflesnap.git@afd6626c3353b41c0931a7e5111b31c4b2c3e8f2"
.venv/bin/python benchmarks/export_stats.py
.venv/bin/python benchmarks/build_figures.py
.venv/bin/python benchmarks/build_diagrams.py
.venv/bin/python benchmarks/build_overview.py
.venv/bin/python benchmarks/build_method_figure.py
```

The numerical figures use stored CSV/JSON results and cached small assignments. Diagram generation executes small solver examples. The MNIST raster figures and billion-point image are supplied; rebuilding those images is separate from regenerating the numerical plots.

## Re-run experiments

These commands overwrite the corresponding stored results, including historical timings. Preserve a copy if comparing environments. Exact-reference costs are cached in `benchmarks/results/exact_cache.json`, with input-coordinate hashes and grid geometry to reject stale entries. Reference generation also saves the seed-0 assignments used in the gallery, avoiding a second identical solve. Runtime depends on the machine and experiment; the large runs can take hours and tens of GiB of memory.

```bash
.venv/bin/python benchmarks/precompute_exact.py
.venv/bin/python benchmarks/bench_convergence.py
.venv/bin/python benchmarks/bench_distributions.py
.venv/bin/python benchmarks/bench_ablations.py
.venv/bin/python benchmarks/bench_init.py
.venv/bin/python benchmarks/bench_oracle_check.py
.venv/bin/python benchmarks/bench_scaling.py
.venv/bin/python benchmarks/bench_mnist.py
```

`bench_mnist.py --rebuild-embedding` additionally needs `umap-learn`. The stored embedding is the basis of the reported example; rebuilding a stochastic embedding under another dependency version need not reproduce it bit-for-bit.

For the BSP-OT comparisons, build the driver with Eigen and the upstream header:

```bash
c++ -O3 -std=c++20 -fopenmp -I path/to/eigen -I path/to/BSP-OT \
    solvers/cpp/bspot_grid.cpp -o solvers/cpp/bspot_grid
.venv/bin/python benchmarks/bench_bspot.py --help
```

The driver evaluates Gaussian bijective matching with explicitly chosen plan counts. Its timing excludes input file loading and target-grid construction. The paired ShuffleSnap budget comparison excludes seed construction and keeps only completed round checkpoints. Historical timings should not be interpreted as end-to-end latency.

`bench_billion.py --help` describes the budgeted stress test. The custom single-plan BSP-OT memory experiment uses `solvers/cpp/bspot_grid_giga.cpp`, built with `c++ -O3 -std=c++20`; it is not the official multi-plan comparator used in the main table. Billion-point assignments and position buffers are intentionally excluded from Git.

## Evidence and provenance

- `benchmarks/results/environment_unit_square.json` records the margin-free rerun: compiler, installed packages, source hashes, upstream BSP-OT revision, explicit comparator thread limit, and completed experiment scripts. All 39 auction references and non-billion paper experiments were regenerated.
- `benchmarks/results/environment.json` records the machine and basic environment for these reruns. The older `environment_convergence.json` is retained as historical provenance.
- The 100-million-point comparison was refreshed with raw driver logs. Only the billion-point solver measurements retain the original inset coordinates and timings; the paper labels them separately. The saved billion assignment is rendered in unit-square coordinates, without changing the measured run.
- The historical billion-point runs did not record full dependency/compiler/upstream provenance. Those missing details cannot be reconstructed with certainty from their summaries.
- Auction reference costs use a very small floating-point epsilon and agree with independent dense solvers on 12 small instances. Stored runs do not contain independently verified dual certificates. “Reference gap” therefore means a numerical comparison, not a certified error bound.
- Headline statistics and the comparison table are generated by `export_stats.py`; some descriptive numbers and experiment settings are written directly in the manuscript.
- See `PUBLICATION_REVIEW.md` for review changes, validation, and remaining handoff items.

## Additional scripts

- `benchmarks/refresh_bspot_references.py` updates stress-test comparisons against the stored descent summaries.
- `benchmarks/render_billion_4x_box_area_tiled.py` and `benchmarks/render_billion_antialias.py` generate exploratory antialiased print renders, kept locally under the ignored render-variant paths.
- `benchmarks/render_billion_test.py` renders interpolation views from the large local artifacts; see `--help`.
- `scripts/check_results.py` reproduces all 15 retained distribution outcomes and checks cached assignments and time-matched checkpoints without changing recorded timings.

```bash
.venv/bin/python scripts/check_results.py
```

The main paper is a short introduction to the technique and its headline comparison;
algorithm details and experimental evidence follow in appendices. The current
benchmark suite and generated manuscript use five distribution families. Archived
raw measurements from the earlier suite remain available for provenance, but are
excluded from the current figures, tables, and headline statistics.

The 100-million-point comparison can be regenerated separately (several GiB of
input storage and roughly 30 GiB of process memory for the 16-plan comparator):

```bash
.venv/bin/python benchmarks/bench_billion.py --side 10000 --seed 0 --budget 500 --no-save-assignment
.venv/bin/python benchmarks/bench_hundred_m.py
.venv/bin/python benchmarks/export_stats.py
```

`bench_hundred_m.py` requires the recorded unit-square environment metadata and
rebuilt drivers. It uses four OpenMP threads for the 16-plan run and one for
each single-plan run, and saves raw output and peak-memory measurements.
