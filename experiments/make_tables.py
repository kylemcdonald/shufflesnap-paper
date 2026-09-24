"""LaTeX tables from results/raw/*.jsonl (written to paper/tables/)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis  # noqa: E402

OUT = os.path.join(HERE, "..", "paper", "tables")
DIST_NAMES = {"uniform": "uniform", "gaussian": "Gaussian", "gmm": "Gaussian mixture", "unequal2": "90/10 clusters",
              "ring": "ring", "spiral": "spiral", "checker": "checkerboard", "heavy": "heavy tails",
              "duplicates": "duplicates", "line": "near-line", "jitter": "jittered lattice", "mnist": "MNIST UMAP",
              "fashion": "Fashion UMAP"}
MAIN_COLS = [("SS-baseline", "SS base"), ("SS-fast", "SS fast"), ("SS-balanced", "SS bal."), ("SS-quality", "SS qual."),
             ("BSP-OT-16", "BSP 16"), ("BSP-OT-64", "BSP 64"), ("RasterFairy-master", "RF master"),
             ("RasterFairy-pypi", "RF PyPI"), ("Bisection", "Bisect.")]


def _ok(df):
    return df[df["error"].isna() & df["valid"].fillna(False).astype(bool)]


def fmt_pct(x):
    if x is None or not np.isfinite(x):
        return "--"
    if x < 0.001:
        return "$<$0.001"
    if x < 0.1:
        return f"{x:.3f}"
    if x < 10:
        return f"{x:.2f}"
    if x < 1000:
        return f"{x:.1f}"
    return f"{x:.0f}"


def fmt_t(x):
    if x is None or not np.isfinite(x):
        return "--"
    if x < 0.1:
        return f"{x:.3f}"
    if x < 10:
        return f"{x:.2f}"
    if x < 100:
        return f"{x:.1f}"
    return f"{x:.0f}"


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)
    print("wrote", name)


def table_main(df, n=100000, threads=1):
    d = _ok(df[(df["n"] == n) & (df["grid"] == "square") & (df["threads"] == threads) &
               df["file"].str.startswith("main")])
    rows = []
    for dist in DIST_NAMES:
        s = d[d["dist"] == dist]
        if s.empty:
            continue
        cells = [DIST_NAMES[dist]]
        for lab, _ in MAIN_COLS:
            e = s.loc[s["label"] == lab, "excess_pct"].mean()
            cells.append(fmt_pct(e))
        ex = s.loc[s["label"] == "Exact-certified", "time_s"]
        cells.append(fmt_t(ex.median()) if len(ex) else "--")
        rows.append(" & ".join(cells) + r" \\")
    # median time row
    trow = [r"\emph{median time (s)}"]
    for lab, _ in MAIN_COLS:
        trow.append(fmt_t(d.loc[d["label"] == lab, "time_s"].median()))
    trow.append(fmt_t(d.loc[d["label"] == "Exact-certified", "time_s"].median()))
    head = " & ".join(["distribution"] + [c for _, c in MAIN_COLS] + ["exact (s)"]) + r" \\"
    text = (r"\begin{tabular}{l" + "r" * (len(MAIN_COLS) + 1) + "}\n\\toprule\n" + head + "\n\\midrule\n" +
            "\n".join(rows) + "\n\\midrule\n" + " & ".join(trow) + r" \\" + "\n\\bottomrule\n\\end{tabular}\n")
    write(f"tab_main_{n // 1000}k_t{threads}.tex", text)


def table_abs(df, n=100000, threads=1):
    """Absolute excess per point (cells^2), same layout."""
    d = _ok(df[(df["n"] == n) & (df["grid"] == "square") & (df["threads"] == threads) &
               df["file"].str.startswith("main")])
    rows = []
    for dist in DIST_NAMES:
        s = d[d["dist"] == dist]
        if s.empty:
            continue
        opt = s["ref_cost"].mean() / n
        cells = [DIST_NAMES[dist], fmt_t(opt)]
        for lab, _ in MAIN_COLS:
            cells.append(fmt_t(s.loc[s["label"] == lab, "excess_per_pt"].mean()))
        rows.append(" & ".join(cells) + r" \\")
    head = " & ".join(["distribution", "optimum"] + [c for _, c in MAIN_COLS]) + r" \\"
    text = (r"\begin{tabular}{l" + "r" * (len(MAIN_COLS) + 1) + "}\n\\toprule\n" + head + "\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write(f"tab_abs_{n // 1000}k_t{threads}.tex", text)


def main():
    df = analysis.load()
    for n in (10000, 50000, 100000):
        if ((df["n"] == n) & df["file"].str.startswith("main")).any():
            table_main(df, n=n)
            table_abs(df, n=n)


if __name__ == "__main__":
    main()
