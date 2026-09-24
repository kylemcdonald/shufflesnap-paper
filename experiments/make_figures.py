"""Result figures (tradeoff, scaling, anytime, ablations, threads) from results/raw/*.jsonl.

    python experiments/make_figures.py [tradeoff|scaling|anytime|ablation|threads|all]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis  # noqa: E402
import figstyle as fs  # noqa: E402

OUT = os.path.join(HERE, "..", "paper", "figures")
DIST_TITLES = {"uniform": "uniform", "gaussian": "Gaussian", "gmm": "Gaussian mixture", "unequal2": "90/10 clusters",
               "ring": "ring", "spiral": "spiral", "checker": "checkerboard", "heavy": "heavy tails (t, ν=1.5)",
               "duplicates": "16 duplicated sites", "line": "near-line", "jitter": "jittered lattice",
               "mnist": "MNIST UMAP", "fashion": "Fashion-MNIST UMAP"}
FLOOR = 1e-4  # plot floor for relative excess (%)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
    plt.close(fig)
    print("wrote", name)


def _ok(df):
    return df[df["error"].isna() & df["valid"].fillna(False).astype(bool)]


def _family_points(ax, sub, xcol="time_s", ycol="excess_pct"):
    """Mean over seeds per label; SS presets and BSP-OT plan counts joined by lines."""
    g = sub.groupby("label").agg(x=(xcol, "median"), y=(ycol, "mean"), ylo=(ycol, "min"), yhi=(ycol, "max"))
    g["y"] = g["y"].clip(lower=FLOOR)
    ss_order = ["SS-baseline", "SS-fast", "SS-balanced", "SS-quality"]
    have = [l for l in ss_order if l in g.index]
    if have:
        ax.plot(g.loc[have, "x"], g.loc[have, "y"], color=fs.SS_RAMP["SS-balanced"], lw=0.8, zorder=2)
    bsp = sorted([l for l in g.index if l.startswith("BSP-OT")], key=lambda s: int(s.split("-")[-1]))
    if len(bsp) > 1:
        ax.plot(g.loc[bsp, "x"], g.loc[bsp, "y"], color=fs.COLORS["BSP-OT"], lw=0.8, zorder=2)
    for lab, r in g.iterrows():
        if lab.startswith("Exact"):
            continue
        fam = analysis.family(lab)
        if fam not in fs.COLORS:
            continue
        filled = fs.marker(lab) not in ("x", "+")
        ax.errorbar(r["x"], r["y"], yerr=[[max(r["y"] - max(r["ylo"], FLOOR), 0)], [max(r["yhi"] - r["y"], 0)]],
                    fmt=fs.marker(lab), color=fs.color(lab), ms=4.5 if filled else 5, mec="white" if filled else None,
                    mew=0.5 if filled else 1.2, elinewidth=0.6, capsize=0, zorder=3)
    ex = g.loc[[l for l in g.index if l == "Exact-certified"]]
    for _, r in ex.iterrows():
        ax.axvline(r["x"], color=fs.INK, lw=0.8, zorder=1)
    return g


def fig_tradeoff(df, n=100000, threads=1, name="fig_tradeoff"):
    d = _ok(df[(df["n"] == n) & (df["grid"] == "square") & (df["threads"] == threads) &
               (df["file"].str.startswith("main"))])
    # heavy tails: every method is within 0.001% in relative terms (outliers dominate the
    # optimum), so the panel carries no information; it is reported in absolute terms in the table
    dists = [x for x in DIST_TITLES if x in set(d["dist"]) and x != "heavy"]
    ncol = 4
    nrow = int(np.ceil((len(dists) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.75 * nrow + 0.3), layout="constrained")
    axes = axes.ravel()
    for ax, dist in zip(axes, dists):
        _family_points(ax, d[d["dist"] == dist])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(DIST_TITLES[dist], loc="left")
        fs.recess(ax, grid_axis="both")
    for ax in axes[len(dists):]:
        ax.axis("off")
    # legend panel
    lax = axes[len(dists)]
    handles = []
    for lab in ["SS-baseline", "SS-fast", "SS-balanced", "SS-quality", "BSP-OT", "RasterFairy-master",
                "RasterFairy-pypi", "Bisection"]:
        filled = fs.marker(lab) not in ("x", "+")
        handles.append(plt.Line2D([], [], ls="", marker=fs.marker(lab), color=fs.color(lab), ms=5,
                                  mec="white" if filled else None, mew=0.5 if filled else 1.2,
                                  label=fs.NAMES[lab] if lab != "BSP-OT" else "BSP-OT (16, 64 plans)"))
    handles.append(plt.Line2D([], [], color=fs.INK, lw=0.8, label="exact, certified (time)"))
    lax.legend(handles=handles, loc="center left", fontsize=6.3, handlelength=1.2)
    for ax in axes[::ncol]:
        ax.set_ylabel("excess over optimum (%)")
    for ax in axes[max(0, len(dists) - ncol):len(dists)]:
        ax.set_xlabel(f"time (s, {'1 thread' if threads == 1 else f'{threads} threads'})")
    _save(fig, name)


def fig_scaling(df, name="fig_scaling"):
    d = _ok(df[df["file"].str.startswith("scaling") & (df["grid"] == "square")])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), layout="constrained", sharex=True)
    for c, dist in enumerate(["uniform", "gmm"]):
        for r, (col, ylabel) in enumerate([("time_s", "time (s, 1 thread)"), ("peak_rss_mb", "peak extra memory (MB)")]):
            ax = axes[r, c]
            sub = d[(d["dist"] == dist) & (d["threads"] == 1)]
            for lab in ["SS-baseline", "SS-balanced", "SS-quality", "BSP-OT-16", "RasterFairy-master",
                        "Exact-certified", "Exact-dense-scipy"]:
                s2 = sub[sub["label"] == lab].groupby("n")[col].median()
                if s2.empty:
                    continue
                y = s2.clip(lower=0.05 if col == "peak_rss_mb" else 1e-4)
                ax.plot(s2.index, y.values, marker=fs.marker(lab), color=fs.color(lab), ms=3.5, mec="white", mew=0.4,
                        lw=1.2, label=fs.NAMES[analysis.family(lab)] if lab != "BSP-OT-16" else "BSP-OT (16 plans)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            fs.recess(ax, grid_axis="both")
            if r == 0:
                ax.set_title(DIST_TITLES[dist], loc="left")
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 1:
                ax.set_xlabel("points N")
    axes[0, 0].legend(fontsize=6.3, loc="upper left")
    _save(fig, name)


def fig_anytime(df, threads=1, name="fig_anytime"):
    d = _ok(df[df["file"].str.startswith("anytime") & (df["threads"] == threads)])
    dists = [x for x in ["uniform", "gmm", "mnist"] if x in set(d["dist"])]
    fig, axes = plt.subplots(1, len(dists), figsize=(7.0, 2.5), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, dist in zip(axes, dists):
        sub = d[d["dist"] == dist]
        ref = sub["ref_cost"].dropna()
        if ref.empty:
            continue
        opt = ref.iloc[0]
        for lab in ["SS-baseline", "SS-fast", "SS-balanced", "SS-quality"]:
            r = sub[sub["label"] == lab]
            if r.empty or not isinstance(r.iloc[0].get("trace"), list):
                continue
            tr = np.array(r.iloc[0]["trace"])
            t = tr[:, 0]
            y = np.clip(100 * (tr[:, 1] / opt - 1), FLOOR, None)
            ax.step(t, y, where="post", color=fs.color(lab), lw=1.2, label=fs.NAMES[lab])
            ax.plot(t[-1], y[-1], marker="o", color=fs.color(lab), ms=4, mec="white", mew=0.5)
        bsp = sub[sub["label"].str.startswith("BSP-OT")].copy()
        if not bsp.empty:
            bsp["k"] = bsp["label"].str.split("-").str[-1].astype(int)
            bsp = bsp.sort_values("k")
            ax.plot(bsp["time_s"], bsp["excess_pct"].clip(lower=FLOOR), marker="s", color=fs.COLORS["BSP-OT"], ms=3.5,
                    mec="white", mew=0.4, lw=1.0, label="BSP-OT, 1–1024 plans")
            for _, r in bsp.iterrows():
                if r["k"] in (1, 16, 1024):
                    ax.annotate(str(r["k"]), (r["time_s"], max(r["excess_pct"], FLOOR)), textcoords="offset points",
                                xytext=(3, 3), fontsize=5.5, color=fs.INK2)
        rf = sub[sub["label"].str.startswith("RasterFairy")]
        for _, r in rf.iterrows():
            ax.plot(r["time_s"], max(r["excess_pct"], FLOOR), marker="^", color=fs.COLORS["RasterFairy-master"], ms=4,
                    mec="white", mew=0.4, ls="")
        if not rf.empty:
            ax.plot([], [], marker="^", color=fs.COLORS["RasterFairy-master"], ls="", label="RasterFairy (leaf 1–1000)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"time (s, {'1 thread' if threads == 1 else f'{threads} threads'})")
        ax.set_title(DIST_TITLES[dist] + f", N = {int(sub['n'].iloc[0]):,}", loc="left")
        fs.recess(ax, grid_axis="both")
    axes[0].set_ylabel("excess over optimum (%)")
    axes[-1].legend(fontsize=6, loc="upper right")
    _save(fig, name)


ABL_GROUPS = [("window", "window side w (ratio 2)", "p_window"), ("window-r1.41", "window side w (ratio √2)", "p_window"),
              ("ratio", "stride ratio (w = 6)", "p_ratio"), ("start", "start (w = 6)", "p_init"),
              ("rounds", "rounds per level (w = 6)", "p_rounds"), ("offsets", "tilings per round (w = 6)", "p_offsets"),
              ("coarse", "largest stride (w = 6)", "p_start_stride"), ("solver", "window solver", "p_solver")]


def _abl_label(group, v, row=None):
    if group == "offsets":
        k = len(json.loads(v)) if isinstance(v, str) else 4
        return {1: "1", 2: "2 (diag.)", 4: "4 (default)", 9: "9"}.get(k, str(k))
    if group == "ratio":
        return {3.0: "3", 2.0: "2", 2 ** 0.75: "1.68", 2 ** 0.5: "1.41", 2 ** 0.25: "1.19", 1.1: "1.1"}.get(
            round(float(v), 6) if v == v else v, f"{float(v):.2f}")
    if group == "coarse":
        return "auto" if v is None or v != v else str(int(v))
    if group == "solver":
        return f"{v}\nw={int(row['p_window'])}" if row is not None else str(v)
    if group in ("window", "window-r1.41", "rounds"):
        return str(int(v))
    return str(v)


def fig_ablation(df, name="fig_ablation"):
    d = _ok(df[df["file"].str.startswith("ablations")])
    if d.empty:
        return
    d = d.copy()
    d["group"] = d["label"].str.replace("abl-", "", regex=False)
    dists = [x for x in ["uniform", "gmm", "mnist"] if x in set(d["dist"])]
    groups = [g for g in ABL_GROUPS if g[0] in set(d["group"])]
    fig, axes = plt.subplots(2, len(groups) // 2 + len(groups) % 2, figsize=(7.0, 4.6), layout="constrained")
    axes = axes.ravel()
    shades = {"uniform": "#2a78d6", "gmm": "#eb6834", "mnist": "#1baf7a"}
    marks = {"uniform": "o", "gmm": "s", "mnist": "^"}
    for ax, (group, title, col) in zip(axes, groups):
        sub = d[d["group"] == group]
        for dist in dists:
            s2 = sub[sub["dist"] == dist]
            if s2.empty:
                continue
            key = s2[col] if col in s2 else pd.Series(["?"] * len(s2))
            if group == "solver":
                keyv = s2["p_solver"].astype(str) + "|" + s2["p_window"].astype(str)
            elif group == "coarse":
                keyv = s2[col].fillna(-1)
            else:
                keyv = key
            gg = s2.assign(k=keyv).groupby("k", sort=False).agg(e=("excess_pct", "mean"), t=("time_s", "median"),
                                                                   rep=(col if col in s2 else "label", "first"))
            ax.plot(gg["t"], gg["e"].clip(lower=FLOOR), marker=marks[dist], color=shades[dist], ms=3.5, lw=0.9,
                    mec="white", mew=0.4, label=DIST_TITLES[dist])
            if dist == dists[0]:
                for kk, r in gg.iterrows():
                    if group == "solver":
                        sv, w = str(kk).split("|")
                        lab = {"hungarian": "H", "hungarian_greedy": "H+g", "jv": "JV"}.get(sv, sv) + f" w{int(float(w))}"
                    elif group == "coarse":
                        lab = "auto" if kk == -1 else str(int(kk))
                    else:
                        lab = _abl_label(group, r["rep"])
                    ax.annotate(lab, (r["t"], max(r["e"], FLOOR)), textcoords="offset points", xytext=(3, 2),
                                fontsize=5.5, color=fs.INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=7.5)
        fs.recess(ax, grid_axis="both")
    for ax in axes[len(groups):]:
        ax.axis("off")
    axes[0].legend(fontsize=6, loc="best")
    fig.supxlabel("time (s, 24 threads)", fontsize=8)
    fig.supylabel("excess over optimum (%)", fontsize=8)
    _save(fig, name)


def fig_threads(df, name="fig_threads"):
    import glob
    d = _ok(df[df["file"].str.startswith("threads")])
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), layout="constrained")
    for ax, dist in zip(axes, ["uniform", "gmm"]):
        sub = d[d["dist"] == dist]
        for lab in ["SS-baseline", "SS-balanced", "BSP-OT-16"]:
            s2 = sub[sub["label"] == lab].groupby("threads")["time_s"].median()
            if s2.empty:
                continue
            ax.plot(s2.index, s2.iloc[0] / s2.values, marker=fs.marker(lab), color=fs.color(lab), ms=4, mec="white",
                    mew=0.4, label=f"{fs.NAMES[analysis.family(lab)] if lab != 'BSP-OT-16' else 'BSP-OT (16 plans)'}"
                                   f" ({s2.iloc[0]:.1f} s on 1 thread)")
        ax.plot([1, 24], [1, 24], color=fs.GRID, lw=0.8)
        ax.axvline(16, color=fs.GRID, lw=0.6)
        ax.set_xlabel("threads (≤16: performance cores)")
        ax.set_title(DIST_TITLES[dist] + ", N = 100,000", loc="left")
        fs.recess(ax, grid_axis="y")
        ax.legend(fontsize=5.8, loc="upper left")
    axes[0].set_ylabel("speed-up over 1 thread")
    _save(fig, name)


def main(which="all"):
    fs.setup()
    df = analysis.load()
    if which in ("tradeoff", "all"):
        for n in (10000, 50000, 100000):
            if ((df["n"] == n) & df["file"].str.startswith("main")).any():
                fig_tradeoff(df, n=n, name=f"fig_tradeoff_{n // 1000}k")
    if which in ("scaling", "all") and df["file"].str.startswith("scaling").any():
        fig_scaling(df)
    if which in ("ablation", "all"):
        fig_ablation(df)
    if which in ("threads", "all"):
        fig_threads(df)
    if which in ("anytime", "all") and df["file"].str.startswith("anytime").any():
        for t in sorted(df.loc[df["file"].str.startswith("anytime"), "threads"].unique()):
            fig_anytime(df, threads=int(t), name=f"fig_anytime_t{int(t)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
