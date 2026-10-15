"""LaTeX tables from results/raw/*.jsonl (written to paper/tables/)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis  # noqa: E402
import specs  # noqa: E402

OUT = os.path.join(HERE, "..", "paper", "tables")
DIST_NAMES = {"uniform": "uniform", "gaussian": "Gaussian", "gmm": "Gaussian mixture", "unequal2": "90/10 clusters",
              "ring": "ring", "spiral": "spiral", "checker": "checkerboard", "heavy": "heavy tails",
              "duplicates": "duplicates", "line": "near-line", "jitter": "jittered lattice", "mnist": "MNIST UMAP",
              "fashion": "Fashion UMAP"}
# each ShuffleSnap preset next to the BSP-OT setting that takes about as long
MAIN_COLS = [("SS-fast", "SS fast"), ("BSP-OT-16", "BSP 16"), ("SS-balanced", "SS bal."), ("BSP-OT-64", "BSP 64"),
             ("SS-quality", "SS qual."), ("RasterFairy-master", "RF master"),
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


def _exact_cell(df, dist, n, threads, label):
    """Time of a timed exact run on seed 1, or why it has no time (timeout / not certified)."""
    r = df[(df["dist"] == dist) & (df["n"] == n) & (df["grid"] == "square") & (df["threads"] == threads) &
           (df["label"] == label) & df["file"].str.startswith("main")]
    if r.empty:
        return "--"
    r = r.iloc[0]
    st = r.get("exact_status", "")
    if st == "timeout":
        return r"$>$3600\textsuperscript{t}"
    if st == "not certified":
        return fmt_t(r["time_s"]) + r"\textsuperscript{n}"
    return fmt_t(r["time_s"])


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
        cells.append(_exact_cell(df, dist, n, threads, "Exact-certified"))
        if n >= 100000:
            cells.append(_exact_cell(df, dist, n, threads, "Exact-certified-bisect"))
        if n <= 10000:
            cells.append(_exact_cell(df, dist, n, threads, "Exact-dense-scipy"))
        rows.append(" & ".join(cells) + r" \\")
    # median time row
    trow = [r"\emph{median time (s)}"]
    for lab, _ in MAIN_COLS:
        trow.append(fmt_t(d.loc[d["label"] == lab, "time_s"].median()))
    trow.append(fmt_t(d.loc[d["label"] == "Exact-certified", "time_s"].median()))
    extra = ["exact (s)"] + (["exact, bis. (s)"] if n >= 100000 else []) + (["dense (s)"] if n <= 10000 else [])
    if n >= 100000:
        trow.append("")
    if n <= 10000:
        trow.append(fmt_t(d.loc[d["label"] == "Exact-dense-scipy", "time_s"].median()))
    head = " & ".join(["distribution"] + [c for _, c in MAIN_COLS] + extra) + r" \\"
    text = (r"\begin{tabular}{l" + "r" * (len(MAIN_COLS) + len(extra)) + "}\n\\toprule\n" + head + "\n\\midrule\n" +
            "\n".join(rows) + "\n\\midrule\n" + " & ".join(trow) + r" \\" + "\n\\bottomrule\n\\end{tabular}\n")
    write(f"tab_main_{n // 1000}k_t{threads}.tex", text)


def table_main_compact(df, n=100000, threads=1):
    """The main table's key columns, for the shorter editions."""
    d = _ok(df[(df["n"] == n) & (df["grid"] == "square") & (df["threads"] == threads) &
               df["file"].str.startswith("main")])
    cols = [("SS-fast", "SS fast"), ("BSP-OT-16", "BSP-OT 16"), ("SS-balanced", "SS bal."), ("BSP-OT-64", "BSP-OT 64"),
            ("SS-quality", "SS qual."), ("RasterFairy-master", "RasterFairy")]
    rows = []
    for dist in DIST_NAMES:
        s = d[d["dist"] == dist]
        if s.empty:
            continue
        cells = [DIST_NAMES[dist]] + [fmt_pct(s.loc[s["label"] == lab, "excess_pct"].mean()) for lab, _ in cols]
        cells.append(_exact_cell(df, dist, n, threads, "Exact-certified"))
        rows.append(" & ".join(cells) + r" \\")
    trow = [r"\emph{median time (s)}"] + [fmt_t(d.loc[d["label"] == lab, "time_s"].median()) for lab, _ in cols]
    trow.append(fmt_t(d.loc[d["label"] == "Exact-certified", "time_s"].median()))
    head = " & ".join(["distribution"] + [c for _, c in cols] + ["exact (s)"]) + r" \\"
    text = (r"\begin{tabular}{lrrrrrrr}" + "\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows) +
            "\n\\midrule\n" + " & ".join(trow) + r" \\" + "\n\\bottomrule\n\\end{tabular}\n")
    write(f"tab_main_compact_{n // 1000}k_t{threads}.tex", text)


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


GRID_NAMES = {"free": "surplus ($<$1 row)", "free10": "10\\% surplus", "wide": "4:1 rectangle", "disk": "disk",
              "ring": "ring", "logo": "S mask (font)", "srect": "S mask", "square": "exact fit"}
MASK_COLS = [("SS-fast", "SS fast"), ("BSP-OT-16", "BSP 16"), ("SS-balanced", "SS bal."), ("BSP-OT-64", "BSP 64"),
             ("SS-quality", "SS qual."), ("RasterFairy-master", "RF master")]


def table_masks(df):
    d = _ok(df[df["file"].str.startswith("masks")])
    if d.empty:
        return
    rows = []
    for grid in ["free", "free10", "wide", "disk", "ring", "srect"]:
        for dist in ["uniform", "gmm", "spiral", "mnist"]:
            s = d[(d["grid"] == grid) & (d["dist"] == dist)]
            if s.empty:
                continue
            cells = [GRID_NAMES[grid], DIST_NAMES[dist]]
            for lab, _ in MASK_COLS:
                v = s.loc[s["label"] == lab, "excess_pct"]
                cells.append(fmt_pct(v.mean()) if len(v) else "--")
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    head = " & ".join(["target", "points"] + [c for _, c in MASK_COLS]) + r" \\"
    text = (r"\begin{tabular}{ll" + "r" * len(MASK_COLS) + "}\n\\toprule\n" + head + "\n\\midrule\n" +
            "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_masks.tex", text)


def table_primes(df):
    d = _ok(df[df["file"].str.startswith("primes")])
    if d.empty:
        return
    rows = []
    for n in sorted(d["n"].unique()):
        for grid in ["square", "free"]:
            for dist in ["uniform", "gmm"]:
                s = d[(d["n"] == n) & (d["grid"] == grid) & (d["dist"] == dist)]
                if s.empty:
                    continue
                g = s.iloc[0]
                shape = f"{int(g['n_cells'])} cells"
                cells = [f"{n:,}", "exact fit" if grid == "square" else "full rectangle", DIST_NAMES[dist]]
                for lab in ("SS-fast", "SS-balanced", "SS-quality", "BSP-OT-16", "RasterFairy-master"):
                    v = s.loc[s["label"] == lab, "excess_pct"]
                    cells.append(fmt_pct(v.mean()) if len(v) else "--")
                rows.append(" & ".join(cells) + r" \\")
    head = r"$N$ & target & points & SS fast & SS bal. & SS qual. & BSP 16 & RF master \\"
    text = ("\\begin{tabular}{lllrrrrr}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_primes.tex", text)


REAL_ROWS = [("Exact-certified", "exact (certified)"), ("SS-fast", "ShuffleSnap fast"), ("SS-balanced", "ShuffleSnap balanced"),
             ("SS-quality", "ShuffleSnap quality"), ("BSP-OT-16", "BSP-OT, 16 plans"), ("BSP-OT-64", "BSP-OT, 64 plans"),
             ("RasterFairy-master", "RasterFairy master"), ("RasterFairy-pypi", "RasterFairy PyPI"),
             ("Bisection", "bisection only")]


def table_real(df):
    d = _ok(df[df["file"].str.startswith("real")])
    if d.empty:
        return
    t24 = d[d["threads"] == 24]
    d1 = d[d["threads"] == 1]
    rows = []
    for dist in ["mnist", "fashion"]:
        s = d1[d1["dist"] == dist]
        s24 = t24[t24["dist"] == dist]
        for lab, name in REAL_ROWS:
            r = s[s["label"] == lab]
            if r.empty:
                continue
            r24 = s24[s24["label"] == lab]
            exc = "0" if lab.startswith("Exact") else fmt_pct(r["excess_pct"].mean())
            cells = [DIST_NAMES[dist] if lab == REAL_ROWS[0][0] else "", name,
                     exc, f"{r['mean_sq'].mean():.1f}",
                     fmt_t(r["time_s"].median()), fmt_t(r24["time_s"].median()) if len(r24) else "--",
                     f"{r['peak_rss_mb'].median():.0f}", f"{r['knn10'].mean():.3f}", f"{r['lattice_nbr'].mean():.3f}"]
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    head = (r" & method & excess (\%) & mean sq. & time 1T (s) & time 24T (s) & mem (MiB) & kNN recall & "
            r"lattice prec. \\")
    text = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows[:-1]) +
            "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_real.tex", text)


ABL_NAMES = {"window": "window side (halving)", "window-r1.41": r"window side (ratio $\sqrt2$)",
             "ratio": rf"stride ratio ({specs.DEFAULT['window']}$\times${specs.DEFAULT['window']})", "start": "start", "rounds": "rounds per level", "offsets": "tilings per round",
             "solver": "window solver", "skip_clean": "skip unchanged windows", "coarse": "first stride"}


def _abl_setting(group, r):
    if group in ("window", "window-r1.41"):
        w = int(r["p_window"])
        return f"{w}$\\times${w}" + (" (default)" if group == "window-r1.41" and w == specs.DEFAULT["window"] else "")
    if group == "ratio":
        v = float(r["p_ratio"])
        return {3.0: "3", 2.0: "2 (halving)", 1.414: "$\\sqrt2$ (default)", 1.1: "1.1"}.get(round(v, 3), f"{v:.2f}")
    if group == "start":
        return {"random": "random (default)", "bisect": "bisection", "rowsort": "row sort"}[r["p_init"]]
    if group == "rounds":
        return f"{int(r['p_rounds'])}" + (" (default)" if int(r["p_rounds"]) == 1 else "")
    if group == "offsets":
        import json as _json
        k = len(_json.loads(r["p_offsets"]))
        return {1: "1", 2: "2 (diagonal)", 4: "4 (default)", 9: "9"}.get(k, str(k))
    if group == "solver":
        name = {"hungarian": "Hungarian", "hungarian_greedy": "Hungarian + greedy start", "jv": "Jonker--Volgenant",
                "auction": "auction + exact finish", "geometric": "geometric (default)"}
        ratio = "" if abs(float(r["p_ratio"]) - 2.0) < 1e-9 else r", ratio $\sqrt2$"
        return f"{name[r['p_solver']]}, {int(r['p_window'])}$\\times${int(r['p_window'])}{ratio}"
    if group == "skip_clean":
        return "yes (default)" if bool(r["p_skip_clean"]) else "no"
    if group == "coarse":
        v = r.get("p_start_stride")
        return "automatic (default)" if v is None or v != v else f"capped at {int(v)}"
    return ""


def table_ablation(df):
    d = _ok(df[df["file"].str.startswith("ablations")])
    if d.empty:
        return
    d = d.copy()
    d["group"] = d["label"].str.replace("abl-", "", regex=False)
    d["setting"] = [_abl_setting(g, r) for g, (_, r) in zip(d["group"], d.iterrows())]
    lines = []
    for group in ["window-r1.41", "window", "ratio", "start", "rounds", "offsets", "coarse", "solver", "skip_clean"]:
        sub = d[d["group"] == group]
        if sub.empty:
            continue
        first = True
        for setting, s2 in sub.groupby("setting", sort=False):
            e = {x: s2.loc[s2["dist"] == x, "excess_pct"].mean() for x in ["uniform", "gmm", "mnist"]}
            t = {x: s2.loc[s2["dist"] == x, "time_s"].median() for x in ["uniform", "gmm", "mnist"]}
            lines.append((ABL_NAMES[group] if first else "") + f" & {setting} & " + " & ".join(
                f"{fmt_pct(e[x])} & {fmt_t(t[x])}" for x in ["uniform", "gmm", "mnist"]) + r" \\")
            first = False
        lines.append(r"\addlinespace")
    head = (r"change & setting & \multicolumn{2}{c}{uniform 100k} & \multicolumn{2}{c}{mixture 100k} & "
            r"\multicolumn{2}{c}{MNIST 70k} \\")
    sub = r" & & excess \% & time (s) & excess \% & time (s) & excess \% & time (s) \\"
    text = ("\\begin{tabular}{llrrrrrr}\n\\toprule\n" + head + "\n" + sub + "\n\\midrule\n" + "\n".join(lines[:-1]) +
            "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_ablation.tex", text)


def table_memory(df):
    d = _ok(df[df["file"].str.startswith("memory")])
    if d.empty:
        return
    rows_def = [("SS-fast", "ShuffleSnap fast"), ("SS-balanced", "ShuffleSnap balanced"),
                ("SS-quality", "ShuffleSnap quality"),
                ("BSP-OT-16", "BSP-OT, 16 plans"), ("RasterFairy-master", "RasterFairy master"),
                ("Exact-certified", "exact, certified sparse"), ("Exact-dense-scipy", "exact, dense SciPy"),
                ("Bisection", "bisection only")]
    sizes = [10000, 100000, 1000000]
    lines = []
    for lab, name in rows_def:
        cells = [name]
        for dist in ["uniform", "gmm"]:
            for n in sizes:
                v = d[(d["label"] == lab) & (d["dist"] == dist) & (d["n"] == n)]["peak_rss_mb"]
                cells.append(f"{v.median():.1f}" if len(v) and v.median() < 100 else (f"{v.median():.0f}" if len(v) else "--"))
        lines.append(" & ".join(cells) + r" \\")
    head = (r" & \multicolumn{3}{c}{uniform} & \multicolumn{3}{c}{Gaussian mixture} \\" + "\n" +
            r"method & $10^4$ & $10^5$ & $10^6$ & $10^4$ & $10^5$ & $10^6$ \\")
    text = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_memory.tex", text)


def table_swap(df, n=2500):
    """RasterFairy's swap optimizer (swap_t1 / swap_long_t1) next to the other methods."""
    d = _ok(df[(df["n"] == n) & df["file"].str.startswith("swap")])
    cols = [("SS-fast", "SS fast"), ("SS-balanced", "SS bal."), ("SS-quality", "SS qual."),
            ("BSP-OT-16", "BSP 16"), ("BSP-OT-64", "BSP 64"), ("RasterFairy-master", "RF"),
            ("RasterFairy-swap-1e+04", "$10^4$"), ("RasterFairy-swap-1e+05", "$10^5$"),
            ("RasterFairy-swap-1e+06", "$10^6$"), ("RasterFairy-swap-1e+07", "$10^7$\\textsuperscript{a}"),
            ("RasterFairy-swap-random-1e+06", "rand.\\ $10^6$")]
    rows = []
    for dist in DIST_NAMES:
        s = d[d["dist"] == dist]
        if s.empty:
            continue
        cells = [DIST_NAMES[dist]] + [fmt_pct(s.loc[s["label"] == lab, "excess_pct"].mean()) for lab, _ in cols]
        rows.append(" & ".join(cells) + r" \\")
    trow = [r"\emph{median time (s)}"] + [fmt_t(d.loc[d["label"] == lab, "time_s"].median()) for lab, _ in cols]
    head = (r" & & & & & & & \multicolumn{4}{c}{RF + swap optimizer} & \\" + "\n" +
            r"\cmidrule(lr){8-11}" + "\n" +
            " & ".join(["distribution"] + [c for _, c in cols]) + r" \\")
    text = ("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n" + head + "\n\\midrule\n" +
            "\n".join(rows) + "\n\\midrule\n" + " & ".join(trow) + r" \\" + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_swap.tex", text)


def main():
    df = analysis.load()
    for n in (10000, 50000, 100000):
        if ((df["n"] == n) & df["file"].str.startswith("main")).any():
            table_main(df, n=n)
            if n == 100000:
                table_main_compact(df, n=n)
            table_abs(df, n=n)
    table_masks(df)
    table_primes(df)
    table_real(df)
    table_ablation(df)
    table_memory(df)
    if df["file"].str.startswith("swap").any():
        table_swap(df)


if __name__ == "__main__":
    main()
