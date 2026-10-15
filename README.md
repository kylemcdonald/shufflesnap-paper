# ShuffleSnap paper: experiments, raw results, and manuscript

This repository holds everything behind the paper *ShuffleSnap: strided windowed exact
reassignment for point-to-grid layout, evaluated against certified optima*. The
implementation lives in a separate repository
([kylemcdonald/shufflesnap](https://github.com/kylemcdonald/shufflesnap), released on PyPI as
`shufflesnap` 0.4.0), expected as a sibling directory `../shufflesnap`. Benchmarks ran against a
frozen snapshot of the package (`../frozen/`, see `experiments/run_all.sh`), which equals
release tag v0.4.0 apart from its version string; `experiments/make_frozen_snapshot.sh`
rebuilds it from that tag. The PDFs of all four editions are in `dist/`.

## Layout

| path | contents |
|---|---|
| `paper/` | LaTeX source (`main.tex`, `sections/`), verified bibliography (`refs.bib`), generated `figures/` and `tables/` |
| `experiments/` | datasets, method adapters, benchmark harness, experiment specs, figure/table scripts |
| `results/raw/` | the reported results, one JSON line per run: full spec, timings, peak memory, validity, cost, displacement statistics, neighbourhood metrics, environment (versions, commits). Written by `experiments/run_all.sh` and, for RasterFairy's swap optimizer at N=2,500, `experiments/run_swap.sh`. The development-set search (instance seed 0) is in `dev_ablation*.jsonl`, `explore_dev.jsonl` and `mask_matrix_dev.txt` |
| `results/reference/` | certified optimal assignments (`.npz`: cells + lower bound, gap, rounds) for every instance up to 200k points |
| `results/assignments/` | assignments used in the visual and mask figures |
| `results/seed_study/` | spread over inputs and random starts: 50 uniform clouds of 10^4 points x 50 starts, with each cloud's dense optimum (`experiments/seed_study.py`, summary by `experiments/seed_study_report.py`) |
| `results/billion/` | the billion-point meander run (`experiments/run_billion.sh`, `run_billion_6h.sh`, `run_billion_quality.sh`, then `render_billion.py` and `fig_billion.py`): summary JSON and per-phase trace of each segment, the 10,000-pixel render, and in `earlier/` the earlier implementation's six-hour run and the BSP-OT port it is compared with. The 4 GB assignments are not tracked |
| `data/` | the UMAP embeddings of MNIST and Fashion-MNIST (`experiments/make_embeddings.py` downloads the datasets from OpenML and recomputes them) |
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
bash experiments/make_frozen_snapshot.sh     # package snapshot used by run_all.sh and run_swap.sh
bash experiments/run_all.sh
bash experiments/run_swap.sh
python experiments/seed_study.py && python experiments/seed_study_report.py
python experiments/dev_ablation.py           # development-set search (seed 0 only)

# 4. figures, tables, paper
python experiments/make_figures.py all && python experiments/make_tables.py
python experiments/fig_method.py && python experiments/fig_failures.py
python experiments/fig_visual.py mnist && python experiments/fig_visual.py fashion
python experiments/fig_masks.py
python experiments/fig_reference.py          # annotated reference implementation
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

## License

The code (`experiments/`, `paper/reference.py` and the build scripts) is released under the
MIT License (`LICENSE`). The paper's text and figures (`paper/`, `versions/`, the PDFs in
`dist/`) are released under the Creative Commons Attribution 4.0 International License
(`LICENSE-CC-BY-4.0.txt`). The UMAP embeddings in `data/` are derived from MNIST and
Fashion-MNIST and keep those datasets' licenses.
