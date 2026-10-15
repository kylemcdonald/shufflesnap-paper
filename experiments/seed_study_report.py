"""Summary and figure for experiments/seed_study.py.

    python experiments/seed_study_report.py [--n 10000]

Reads results/seed_study/n{N}/runs.csv; writes summary.txt and fig_seed_study.{pdf,png}
next to it, and for N = 10,000 also paper/figures/fig_seed_{study,hists}.{pdf,png}.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle

ROOT = Path(__file__).resolve().parents[1]
BLUE = figstyle.SS_RAMP["SS-balanced"]
LIGHT = figstyle.SS_RAMP["SS-fast"]


def summarize(runs: pd.DataFrame, clouds: pd.DataFrame) -> str:
    ex = runs["excess_pct"].to_numpy()
    by = runs.groupby("cloud_seed")["excess_pct"]
    per = pd.DataFrame({"mean": by.mean(), "min": by.min(), "max": by.max(), "std": by.std()})
    worst = runs.loc[runs["excess_pct"].idxmax()]
    n = round(clouds.opt_cost.iloc[0] / clouds.opt_per_point.iloc[0])
    abs_ex = (runs["ss_cost"] - runs["opt_cost"]) / n
    within = runs.groupby("cloud_seed")["excess_pct"].var().mean()
    between = by.mean().var()
    q = np.quantile(ex, [0.5, 0.9, 0.95, 0.99])
    L = []
    L.append(f"runs: {len(runs)} ({runs.cloud_seed.nunique()} clouds x {runs.init_seed.nunique()} inits), "
             f"default preset, one thread")
    L.append(f"certified solver reached the dense optimum on {int(runs.certified_matches.sum())}/{len(runs)} pairs "
             f"(certified flag set on {int(runs.certified.sum())})")
    L.append(f"optimum per point: mean {clouds.opt_per_point.mean():.3f}, range {clouds.opt_per_point.min():.3f}"
             f"-{clouds.opt_per_point.max():.3f} cells^2")
    L.append("")
    L.append("excess over the optimum, % (all runs)")
    L.append(f"  mean {ex.mean():.3f}  sd {ex.std(ddof=1):.3f}  min {ex.min():.3f}")
    L.append(f"  median {q[0]:.3f}  p90 {q[1]:.3f}  p95 {q[2]:.3f}  p99 {q[3]:.3f}  max {ex.max():.3f}")
    L.append(f"  worst / mean = {ex.max() / ex.mean():.2f}x;  worst - mean = {ex.max() - ex.mean():.3f} points of %")
    L.append(f"  absolute excess per point: mean {abs_ex.mean():.4f}, max {abs_ex.max():.4f} cells^2")
    L.append(f"  worst run: cloud {int(worst.cloud_seed)}, init {int(worst.init_seed)}, "
             f"{worst.on_optimal_cell:.1%} of points on their optimal cell")
    L.append(f"  points on their optimal cell: mean {runs.on_optimal_cell.mean():.1%}, "
             f"min {runs.on_optimal_cell.min():.1%}")
    L.append("")
    L.append("where the spread comes from (variance of excess %)")
    L.append(f"  within a cloud (init): {within:.4f}   between clouds (cloud mean): {between:.4f}   "
             f"init share {within / (within + between):.0%}")
    L.append(f"  per-cloud max - min: median {(per['max'] - per['min']).median():.3f}, "
             f"largest {(per['max'] - per['min']).max():.3f}")
    L.append(f"  per-cloud worst / per-cloud mean: median {(per['max'] / per['mean']).median():.2f}x, "
             f"largest {(per['max'] / per['mean']).max():.2f}x")
    L.append(f"  cloud means range {per['mean'].min():.3f}-{per['mean'].max():.3f}; "
             f"corr(cloud mean excess, optimum/pt) = "
             f"{np.corrcoef(per['mean'], clouds.set_index('cloud_seed').loc[per.index, 'opt_per_point'])[0, 1]:.2f}")
    L.append("")
    L.append("shape of the distribution over starts (absolute excess per point)")
    za = abs_ex.groupby(runs.cloud_seed).transform(lambda x: (x - x.mean()) / x.std())
    log_ex = np.log(abs_ex)
    zl = log_ex.groupby(runs.cloud_seed).transform(lambda x: (x - x.mean()) / x.std())
    rej = lambda v: sum(stats.shapiro(x).pvalue < 0.05 for _, x in v.groupby(runs.cloud_seed))
    L.append(f"  within-cloud skewness {stats.skew(za):.2f} (log scale {stats.skew(zl):.2f}); "
             f"clouds failing Shapiro-Wilk at 5%: {rej(abs_ex)}/{runs.cloud_seed.nunique()} "
             f"(log scale {rej(log_ex)})")
    L.append(f"  runs > 3 sd above their cloud mean: {int((za > 3).sum())} "
             f"(normal: {len(runs) * stats.norm.sf(3):.1f}; log scale {int((zl > 3).sum())})")
    s, _, med = stats.lognorm.fit(abs_ex, floc=0)
    L.append(f"  pooled log-normal fit: median {med:.4f} cells^2, sigma {s:.3f} "
             f"(KS p = {stats.kstest((log_ex - log_ex.mean()) / log_ex.std(), 'norm').pvalue:.2f}; "
             f"normal KS p = {stats.kstest((abs_ex - abs_ex.mean()) / abs_ex.std(), 'norm').pvalue:.1g})")
    qa = np.quantile(abs_ex, [0.05, 0.5, 0.95])
    qr = np.quantile(ex, [0.05, 0.95])
    L.append(f"  5th-95th percentile: absolute {qa[0]:.4f}-{qa[2]:.4f} cells^2 "
             f"({qa[0] / qa[1]:.2f}-{qa[2] / qa[1]:.2f}x the median), relative {qr[0]:.2f}-{qr[1]:.2f}%")
    wa = abs_ex.groupby(runs.cloud_seed).var().mean()
    ba = abs_ex.groupby(runs.cloud_seed).mean().var()
    ks = sum(stats.ks_2samp(x, abs_ex).pvalue < 0.05 for _, x in abs_ex.groupby(runs.cloud_seed))
    L.append(f"  variance share of the start: absolute {wa / (wa + ba):.0%} (relative {within / (within + between):.0%}); "
             f"clouds whose absolute distribution differs from the pooled one (KS, 5%): {ks}")
    rng = np.random.default_rng(0)
    best = {k: np.mean([np.mean([rng.choice(x, k, replace=False).min() for _ in range(400)])
                        for _, x in by]) for k in (1, 2, 4, 10)}
    L.append("  best of k random starts, mean over clouds (%): "
             + ", ".join(f"k={k} {v:.3f}" for k, v in best.items()))
    L.append("")
    L.append(f"time: ShuffleSnap median {runs.ss_time.median():.3f}s, certified median "
             f"{runs.certified_time.median():.2f}s, dense SciPy median {clouds.dense_time.median():.1f}s "
             f"(parallel workers, one thread each)")
    return "\n".join(L)


def figure(runs: pd.DataFrame, clouds: pd.DataFrame, path: Path):
    """(a) absolute excess of every run, clouds sorted by optimum; (b) all runs with a log-normal fit."""
    figstyle.setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[3, 1.0]), sharey=True)
    n = round(clouds.opt_cost.iloc[0] / clouds.opt_per_point.iloc[0])
    abs_ex = (runs["ss_cost"] - runs["opt_cost"]) / n
    order = clouds.sort_values("opt_per_point").cloud_seed
    rng = np.random.default_rng(0)
    for x, c in enumerate(order):
        y = abs_ex[runs.cloud_seed == c].to_numpy()
        a.scatter(x + rng.uniform(-0.25, 0.25, len(y)), y, s=4, color=LIGHT, alpha=0.55, lw=0)
        a.plot([x - 0.38, x + 0.38], [np.median(y)] * 2, color=BLUE, lw=1.6, solid_capstyle="round")
    q = np.quantile(abs_ex, [0.05, 0.5, 0.95])
    for v, ls in zip(q, ("--", "-", "--")):
        a.axhline(v, color=figstyle.INK2, lw=0.8, ls=ls, zorder=0)
    a.set_xlim(-1, len(order))
    a.set_xticks([])
    a.set_xlabel(f"{len(order)} point clouds, sorted by their optimum")
    a.set_ylabel("excess (cells$^2$ per point)")
    a.set_ylim(0, None)
    bins = np.linspace(0, abs_ex.max() * 1.02, 41)
    b.hist(abs_ex, bins=bins, orientation="horizontal", color=BLUE, edgecolor="white", linewidth=0.5)
    s, _, med = stats.lognorm.fit(abs_ex, floc=0)
    y = np.linspace(1e-4, bins[-1], 300)
    b.plot(stats.lognorm.pdf(y, s, scale=med) * len(abs_ex) * (bins[1] - bins[0]), y, color=figstyle.INK, lw=0.9)
    for v, ls in zip(q, ("--", "-", "--")):
        b.axhline(v, color=figstyle.INK2, lw=0.8, ls=ls)
    b.set_xlabel("runs")
    for v, name in zip(q, ("5th", "median", "95th")):
        b.text(1.02, v, f"{name} {v:.4f}", transform=b.get_yaxis_transform(), va="center", ha="left",
               color=figstyle.INK2, fontsize=6.5, linespacing=1.0)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    a.set_title("(a) by cloud", loc="left", fontsize=8)
    b.set_title(f"(b) all {len(runs)} runs", loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)


def figure_hists(runs: pd.DataFrame, clouds: pd.DataFrame, path: Path):
    """One histogram per cloud over its random starts, clouds sorted by their optimum."""
    figstyle.setup()
    order = clouds.sort_values("opt_per_point")
    ncol = 10
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.0 * nrow + 0.45), sharex=True, sharey=True,
                             layout="constrained")
    hi = np.ceil(runs["excess_pct"].max() * 4) / 4
    bins = np.arange(0, hi + 1e-9, 0.125)
    for ax, (_, c) in zip(axes.flat, order.iterrows()):
        y = runs.loc[runs.cloud_seed == c.cloud_seed, "excess_pct"].to_numpy()
        ax.hist(y, bins=bins, color=BLUE, edgecolor="white", linewidth=0.4)
        ax.axvline(y.mean(), color=figstyle.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.text(0.97, 0.95, f"{c.opt_per_point:.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=6, color=figstyle.INK2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6, length=2)
    for ax in axes.flat[len(order):]:
        ax.set_visible(False)
    ax0 = axes[0, 0]
    ax0.set_xlim(0, hi)
    ax0.set_xticks(np.arange(0, hi + 1e-9, 1.0))
    fig.supxlabel("excess over the optimum (%)", fontsize=8)
    fig.supylabel("random starts", fontsize=8)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    out = ROOT / "results" / "seed_study" / f"n{a.n}"
    runs, clouds = pd.read_csv(out / "runs.csv"), pd.read_csv(out / "clouds.csv")
    text = summarize(runs, clouds)
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    figure(runs, clouds, out / "fig_seed_study")
    if a.n == 10000:
        figure(runs, clouds, ROOT / "paper" / "figures" / "fig_seed_study")
        figure_hists(runs, clouds, ROOT / "paper" / "figures" / "fig_seed_hists")
    if os.environ.get("AGENT_FILES_URL"):
        from urllib.parse import quote
        print(f"{os.environ['AGENT_FILES_URL']}/#{quote(str(out))}")


if __name__ == "__main__":
    main()
