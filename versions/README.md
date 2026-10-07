# Paper editions

| Edition | Body length | PDF | Source |
| --- | --- | --- | --- |
| Full | 36 pages with appendices | `dist/shufflesnap-full.pdf` | `paper/main.tex` |
| Long | 10 pages | `dist/shufflesnap-long.pdf` | `versions/long/main.tex` |
| Medium | 6 pages | `dist/shufflesnap-medium.pdf` | `versions/medium/main.tex` |
| Short | 3 pages | `dist/shufflesnap-short.pdf` | `versions/short/main.tex` |

References start on a new page after the body and do not count toward its length; the
PDFs have 12, 7 and 4 pages in total. Each shorter edition is a standalone manuscript, not
an excerpt: it is written for its length and keeps the full paper's claims, numbers and
figures (it adds no new result).

## Editorial scope

- **Short (3 pages):** the problem, the method (schematic and a real progress strip), the
  certified solver in one paragraph, and one page of evidence: quality against time on
  four inputs, the real-embedding result, and the limits. No tables.
- **Medium (6 pages):** adds the method as numbered steps, the certified solver and setup,
  a table for all eleven distributions, the MNIST gallery with neighbourhood maps, the
  anatomy of the error, masks and surplus cells, and when to use which method.
- **Long (10 pages):** adds the problem statement with the lower-bound proposition,
  boundary and mask rules, implementation and design changes, the solver's details and
  correctness checks, the full quality/time figure, matched budgets and scaling, the full
  MNIST figure, masks with the design-choice ablations, and limitations.
- **Full:** everything, including the ablation table and figure, parallel scaling, prime
  point counts, memory, small-instance exactness, the tables at 10k and 50k points, and the
  serial reference implementation as a listing.

All editions use 11-point body text, one-inch margins on US letter, and figures at the full
paper's size (text width). Captions stay with their figures, and page breaks follow
editorial units (each shorter edition places its figures itself, without floats). There
are no reduced fonts or compressed line spacing; tables use `\small` at most.

## Figures

The shorter editions share the full paper's figures and add compact ones drawn for them by
`experiments/fig_editions.py`: `fig_schematic` (one window solve, a strided window, shifted
tilings), `fig_progress` (a real run on 10,000 MNIST points, next to its certified
optimum), `fig_summary` (four quality/time panels), `fig_gallery_mnist` (three rows of the
full MNIST figure), `fig_failures_row` and `fig_masks_row`. They reuse the benchmark data
and cached assignments; no benchmark was rerun for them. Every edition presents the default preset
(8x8 windows, stride ratio sqrt 2) and uses these same figures; `fig_progress` shows every second
stride, and `fig_summary` shows only the three presets, BSP-OT and RasterFairy. The older
short-only variants (`fig_schematic_w8`, `fig_progress_balanced`, `fig_summary_default`) are no
longer used. `tab_main_compact_100k_t1` is the main table's key columns (`experiments/make_tables.py`).

## Build

```bash
python experiments/fig_editions.py all        # compact figures (needs the package on PYTHONPATH)
python experiments/build_versions.py          # full + all three editions
python experiments/build_versions.py short    # one edition
python experiments/build_versions.py medium --draft   # do not enforce the page count
```

The builder stages each edition with only the files it uses, compiles it with tectonic,
and fails on any overfull box, undefined reference or undefined citation, and on a body
length other than the target (unless `--draft`). It writes the PDF and a source archive to
`dist/` and a `build-manifest.json` next to each edition. `paper/reference.py` is printed
in the full paper; the builder checks that it equals the package's
`shufflesnap/reference.py`.
