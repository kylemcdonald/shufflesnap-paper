# ShuffleSnap paper: experiments, raw results, and manuscript

This repository holds everything behind the paper *ShuffleSnap: strided windowed exact
reassignment for point-to-grid layout, evaluated against certified optima*. The
implementation lives in a separate repository (`shufflesnap`, expected as a sibling
directory `../shufflesnap`).

## Layout

| path | contents |
|---|---|
| `paper/` | LaTeX source (`main.tex`, `sections/`), verified bibliography (`refs.bib`), generated `figures/` and `tables/` |
| `experiments/` | datasets, method adapters, benchmark harness, experiment specs, figure/table scripts |
| `results/raw/` | one JSON line per run: full spec, timings, peak memory, validity, cost, displacement statistics, neighbourhood metrics, environment (versions, commits) |
| `results/reference/` | certified optimal assignments (`.npz`: cells + lower bound, gap, rounds) for every instance up to 200k points |
| `results/assignments/` | assignments used in the visual figures |
| `notebook/LOG.md` | chronological lab notebook, including negative results and bugs found |
| `notebook/citation_verification.md` | how each reference was verified |

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
cd paper && tectonic main.tex
```

Timing boundaries, pinning, warm-up and memory measurement are described in
`experiments/bench.py` and in the paper's setup section. Development decisions used
instance seed 0 only; reported results use seeds 1-3.
