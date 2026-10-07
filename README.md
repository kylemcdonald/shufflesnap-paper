# ShuffleSnap paper: experiments, raw results, and manuscript

This repository holds everything behind the paper *ShuffleSnap: strided windowed exact
reassignment for point-to-grid layout, evaluated against certified optima*. The
implementation lives in a separate repository
([kylemcdonald/shufflesnap](https://github.com/kylemcdonald/shufflesnap), released on PyPI as
`shufflesnap` 0.4.0), expected as a sibling directory `../shufflesnap`. Benchmarks ran against
frozen snapshots of specific package commits (`../frozen/`, see `experiments/run_*.sh`).

## Layout

| path | contents |
|---|---|
| `paper/` | LaTeX source (`main.tex`, `sections/`), verified bibliography (`refs.bib`), generated `figures/` and `tables/` |
| `experiments/` | datasets, method adapters, benchmark harness, experiment specs, figure/table scripts |
| `results/raw/` | one JSON line per run: full spec, timings, peak memory, validity, cost, displacement statistics, neighbourhood metrics, environment (versions, commits) |
| `results/raw_final/` | the result set the paper now reports: `raw_bspmatch` without the jittered lattice and the font S mask, plus all methods on the rectilinear S mask of the earlier work (grid `srect`; `experiments/run_final.sh`). Select it with `SS_RESULTS_RAW=results/raw_final` (and `SS_RESULTS_ASSIGN=results/assignments_final`) |
| `results/raw_bspmatch/` | the previous result set: presets fast/balanced/quality = 5/8/12 windows (ratio sqrt 2), with fast and balanced time-matched to BSP-OT 16 and 64 plans (`experiments/run_bspmatch.sh`), plus unchanged comparator records. Select it with `SS_RESULTS_RAW=results/raw_bspmatch` (and `SS_RESULTS_ASSIGN=results/assignments_bspmatch`) |
| `results/raw_presets/` | an earlier result set (presets 8/12/16): every ShuffleSnap-dependent run redone with the fast/balanced/quality presets (8/12/16 windows, ratio sqrt 2; `experiments/run_presets.sh`), plus unchanged copies of the comparator records. Select it with `SS_RESULTS_RAW=results/raw_presets` (and `SS_RESULTS_ASSIGN=results/assignments_presets`) for `make_figures.py`, `make_tables.py`, `paper_numbers.py` and the figure scripts |
| `results/raw_geometric/` | the previous result set (old presets, geometric solver): every ShuffleSnap-dependent run redone with the geometric window solver (`experiments/run_geometric.sh`), plus unchanged copies of the comparator records. Select it with `SS_RESULTS_RAW=results/raw_geometric` for `make_figures.py`, `make_tables.py` and `paper_numbers.py` |
| `results/reference/` | certified optimal assignments (`.npz`: cells + lower bound, gap, rounds) for every instance up to 200k points |
| `results/assignments/` | assignments used in the visual figures |
| `results/billion/` | the billion-point meander run (`experiments/run_billion.sh`: `billion.py`, then `render_billion.py` and `fig_billion.py`): summary JSON, per-phase trace, run log, renders; the 4 GB assignment `.npy` is not tracked |
| `notebook/LOG.md` | chronological lab notebook, including negative results and bugs found |
| `notebook/citation_verification.md` | how each reference was verified |
| `versions/` | the 10-, 6- and 3-page editions (`versions/README.md`); PDFs in `dist/` |

## Reproducing

```bash
# 1. environment (Python 3.12 used in the paper)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "../shufflesnap[exact]"          # or: cd ../shufflesnap && ./dev_build.sh
bash experiments/fetch_third_party.sh        # comparator sources at the pinned commits

# 2. data and certified references
python experiments/make_embeddings.py mnist fashion
python experiments/references_all.py 6       # parallel; run before any timing

# 3. benchmarks (sequential; ~9 h on the paper's machine)
bash experiments/run_all.sh
python experiments/small_exactness.py
python experiments/dev_ablation.py           # development-set search (seed 0 only)

# 4. figures, tables, paper
python experiments/make_figures.py all && python experiments/make_tables.py
python experiments/fig_method.py && python experiments/fig_failures.py
python experiments/fig_visual.py mnist && python experiments/fig_visual.py fashion
python experiments/fig_masks.py
bash experiments/run_billion.sh             # 1e9 points, 1 h, needs ~53 GiB and an idle machine
bash experiments/run_billion_6h.sh && bash experiments/run_billion_quality.sh   # continue to convergence
python experiments/render_billion.py --assignment results/billion/billion_assignment_31623_seed0_q.npy --name final
python experiments/fig_billion.py
cd paper && tectonic main.tex

# 5. shorter editions (10, 6 and 3 pages; see versions/README.md)
cd .. && python experiments/fig_editions.py all && python experiments/build_versions.py
```

Timing boundaries, pinning, warm-up and memory measurement are described in
`experiments/bench.py` and in the paper's setup section. Development decisions used
instance seed 0 only; reported results use seeds 1-3.
