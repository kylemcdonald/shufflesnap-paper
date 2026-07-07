"""Build all CSV-driven paper figures into paper/figures/.

Style: Okabe-Ito colorblind-safe palette, direct labels, recessive grids.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import make_points  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
FIGURES = REPO / "paper" / "figures"

C = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "black": "#111111",
    "gray": "#888888",
}

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
})

METHOD_STYLE = {
    "random+multiscale": (C["blue"], "-", "multiscale, random seed"),
    "raster+multiscale": (C["green"], "-", "multiscale, raster seed"),
    "raster+local": (C["vermillion"], "--", "local only, raster seed"),
    "rasterfairy+local": (C["orange"], "--", "local only, RasterFairy seed"),
}


def median_trace(df: pd.DataFrame, x_col: str, y_col: str, n_points: int = 200):
    xs = np.linspace(df[x_col].min(), df[x_col].max(), n_points)
    curves = []
    for _, group in df.groupby("instance_seed"):
        group = group.sort_values(x_col)
        curves.append(np.interp(xs, group[x_col], group[y_col]))
    stack = np.vstack(curves)
    return xs, np.median(stack, axis=0), np.min(stack, axis=0), np.max(stack, axis=0)


def fig_convergence() -> None:
    df = pd.read_csv(RESULTS / "convergence_traces.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for method, (color, ls, label) in METHOD_STYLE.items():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        by_round = sub.groupby("round")["gap_pct"].median()
        axes[0].plot(by_round.index, np.maximum(by_round.values, 1e-4),
                     color=color, linestyle=ls, label=label)
        sub_t = sub.copy()
        sub_t["t"] = sub_t["elapsed_s"] + sub_t["seed_time_s"]
        xs, med, lo, hi = median_trace(sub_t[sub_t["round"] > 0], "t", "gap_pct")
        axes[1].plot(xs, np.maximum(med, 1e-4), color=color, linestyle=ls)
        axes[1].fill_between(xs, np.maximum(lo, 1e-4), np.maximum(hi, 1e-4),
                             color=color, alpha=0.15, linewidth=0)

    for ax in axes:
        ax.set_yscale("log")
        ax.axhline(0.1, color=C["gray"], linewidth=0.8, linestyle=":")
        ax.text(0.98, 0.105, "0.1% gap", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=7.5, color=C["gray"])
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("gap above exact optimum (%)")
    axes[0].set_xlim(0, 40)
    axes[1].set_xlabel("wall time incl. initialization (s)")
    axes[1].set_xscale("log")
    axes[0].set_title("per round")
    axes[1].set_title("per wall-clock time (median, min-max band)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), fontsize=7.2, frameon=False,
               columnspacing=1.2, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.png", bbox_inches="tight")
    plt.close(fig)


def fig_scaling() -> None:
    df = pd.read_csv(RESULTS / "scaling.csv")
    med = df.groupby("side").median(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    oracle = med.dropna(subset=["oracle_time_s"])
    ax.plot(oracle["n"], oracle["oracle_time_s"], "^--", color=C["vermillion"],
            markersize=3.5, label="exact auction oracle", zorder=3)
    ax.plot(med["n"], med["elapsed_s"], "o-", color=C["blue"], markersize=3.5,
            label="time to converge", zorder=3)
    ax.plot(med["n"], med["time_within_0p1pct_of_final_s"], "s-", color=C["green"],
            markersize=3.5, label="within 0.1% of converged", zorder=3)
    exact = med.dropna(subset=["time_to_0p1pct_s"])
    ax.plot(exact["n"], exact["time_to_0p1pct_s"], "d-", color=C["orange"],
            markersize=4.5, label="to 0.1% true gap", zorder=4)
    bspot_path = RESULTS / "bspot_scaling.csv"
    if bspot_path.exists():
        bspot = pd.read_csv(bspot_path)
        ax.plot(bspot["n"], bspot["time_s"], "v-", color=C["purple"],
                markersize=3.5, label="BSP-OT construction (~2% gap)", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("points")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("time to quality vs. problem size")

    ax = axes[1]
    ax.plot(med["n"], med["per_round_s"], "o-", color=C["blue"], markersize=3.5,
            label="per-round time")
    ref = med["per_round_s"].iloc[-1] * (med["n"] / med["n"].iloc[-1])
    ax.plot(med["n"], ref, ":", color=C["gray"], linewidth=1.0, label="linear reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("points")
    ax.set_ylabel("seconds per round")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("round cost is linear in n")
    fig.tight_layout()
    fig.savefig(FIGURES / "scaling.png", bbox_inches="tight")
    plt.close(fig)


def fig_ablations() -> None:
    window = pd.read_csv(RESULTS / "ablation_window.csv")
    sched = pd.read_csv(RESULTS / "ablation_schedule.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), width_ratios=[1, 1, 1.5])

    med = window.groupby("window_size").median(numeric_only=True)

    ax = axes[0]
    ax.plot(med.index, med["final_gap_pct"], "o-", color=C["blue"], markersize=4)
    ax.set_yscale("log")
    ax.set_xticks(med.index)
    ax.set_xlabel("window size")
    ax.set_ylabel("gap after 20 s (%)")
    ax.set_title("window size: quality")

    ax = axes[1]
    reached = med.dropna(subset=["time_to_0p1pct_s"])
    ax.plot(reached.index, reached["time_to_0p1pct_s"], "o-", color=C["blue"], markersize=4)
    missing = med.index.difference(reached.index).tolist()
    if missing:
        label = f"w = {', '.join(str(int(w)) for w in missing)}:\nnever reach 0.1%"
        ax.text(0.05, 0.93, label, transform=ax.transAxes, fontsize=6.8,
                color=C["gray"], ha="left", va="top")
    ax.set_xticks(med.index)
    ax.set_ylim(0, None)
    ax.set_xlabel("window size")
    ax.set_ylabel("time to 0.1% gap (s)")
    ax.set_title("window size: speed")

    ax = axes[2]
    med = sched.groupby("schedule").median(numeric_only=True)
    order = [
        "local-only (stride 1)", "fine-to-coarse", "half-depth descent",
        "cycled descent", "ratio-4 descent", "coarse-to-fine (default)",
    ]
    med = med.loc[[o for o in order if o in med.index]]
    y = np.arange(len(med))
    final = med["final_gap_pct"].to_numpy()
    ax.barh(y, final, color=[C["blue"] if "default" in idx else C["gray"] for idx in med.index],
            height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(med.index, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 3)
    ax.set_xticks([1e-2, 1e-1, 1])
    ax.set_xticklabels(["0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.set_xlabel("gap after 20 s (%)")
    ax.set_title("stride schedule")
    ax.invert_yaxis()
    for yi, v in zip(y, final):
        ax.text(v * 1.2, yi, f"{v:.3g}%", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "ablations.png", bbox_inches="tight")
    plt.close(fig)


def _cached_solution(name: str, compute):
    """Load a solver's assignment + solve time from the cache, computing once.

    compute() must return (assignment, time_s).
    """
    path = RESULTS / "assignments" / f"{name}.npz"
    if path.exists():
        data = np.load(path)
        return np.asarray(data["assignment"], dtype=np.int64), float(data["time_s"])
    assignment, time_s = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, assignment=np.asarray(assignment, dtype=np.int64),
             time_s=float(time_s))
    return np.asarray(assignment, dtype=np.int64), float(time_s)


def graph_paper_colors(points: np.ndarray) -> np.ndarray:
    """Position colors multiplied by a graph-paper grid: points within a line
    (spacing 0.1, thickness 0.02) render at half brightness, so solutions show
    the distortion of input space as wavy lines."""
    from build_diagrams import lab_to_srgb

    colors = lab_to_srgb(points)
    spacing, half_thickness = 0.1, 0.01
    offset = np.abs(points - np.round(points / spacing) * spacing)
    on_line = (offset <= half_thickness).any(axis=1)
    colors[on_line] *= 0.5
    return colors


def _fmt_time(t: float) -> str:
    if t < 1.0:
        return f"{t:.2f} s"
    if t < 100.0:
        return f"{t:.1f} s"
    return f"{t:.0f} s"


def fig_distributions() -> None:
    import time as time_mod

    import megalap
    from bench_common import (
        assignment_cost,
        build_target_grid,
        exact_instance,
        exact_square_assignment,
        gap_pct,
        rasterfairy_assignment,
        seed_raster,
    )
    from bench_bspot import run_bspot

    df = pd.read_csv(RESULTS / "distributions.csv")
    side = int(np.sqrt(df["n"].iloc[0]))
    targets = build_target_grid(side, side)
    dists = ["meander", "uniform", "two_blobs", "ring", "spiral", "mixture"]
    labels = {
        "meander": "meander", "uniform": "uniform", "two_blobs": "two blobs",
        "ring": "ring", "spiral": "spiral", "mixture": "mixture",
    }
    row_labels = ["input cloud", "RasterFairy", "BSP-OT",
                  "ours", "exact optimum"]

    def megalap_solution(pts, side):
        result = megalap.window_cleanup(pts, seed_raster(pts), rows=side, cols=side)
        return result["assignment"], float(result["elapsed_s"])

    def bspot_solution(pts, side):
        r = run_bspot(pts, side, side, plans=16, seed=0, want_assignment=True)
        return r["assignment"], r["time_s"]

    def exact_solution(pts, side, dist):
        # reuse a previously cached assignment (from the .npy era) if present
        legacy = RESULTS / "assignments" / f"exact_{dist}_{side}_seed0.npy"
        if legacy.exists():
            return np.load(legacy), float(exact_instance(dist, side, 0)["solve_time_s"])
        t0 = time_mod.time()
        assignment = exact_square_assignment(pts, side, side)
        return assignment, time_mod.time() - t0

    fig, axes = plt.subplots(5, 6, figsize=(7.2, 6.6))
    for col, dist in enumerate(dists):
        pts = make_points(dist, side * side, seed=0)
        colors = graph_paper_colors(pts)
        optimum = exact_instance(dist, side, 0)["cost"]
        tag = f"{dist}_{side}_seed0"

        solutions = {
            "rasterfairy": _cached_solution(
                f"rasterfairy_{tag}", lambda: rasterfairy_assignment(pts, side, side)),
            "bspot": _cached_solution(
                f"bspot_{tag}", lambda: bspot_solution(pts, side)),
            "megalap": _cached_solution(
                f"megalap_{tag}", lambda: megalap_solution(pts, side)),
            "exact": _cached_solution(
                f"exact_{tag}", lambda: exact_solution(pts, side, dist)),
        }

        ax = axes[0][col]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.12, c=colors, linewidths=0)
        ax.invert_yaxis()
        ax.set_title(labels[dist], fontsize=8)

        for row, method in enumerate(["rasterfairy", "bspot", "megalap", "exact"], start=1):
            assignment, time_s = solutions[method]
            image = np.ones((side * side, 3))
            image[assignment] = colors
            ax = axes[row][col]
            ax.imshow(image.reshape(side, side, 3), interpolation="nearest")
            if method == "exact":
                note = _fmt_time(time_s)
            else:
                gap = gap_pct(assignment_cost(pts, targets, assignment), optimum)
                gap_note = f"{gap:.3f}%" if gap < 1 else f"{gap:.1f}%"
                note = f"{gap_note} · {_fmt_time(time_s)}"
            ax.set_xlabel(note, fontsize=6.5, labelpad=1.5)

    for row in range(5):
        axes[row][0].set_ylabel(row_labels[row], fontsize=7.5)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#cccccc")
    fig.tight_layout()
    fig.savefig(FIGURES / "distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig_holes() -> None:
    df = pd.read_csv(RESULTS / "holes.csv")
    med = df.groupby("fill_ratio").median(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), width_ratios=[1.15, 1])

    ax = axes[0]
    ax.plot(med["fill_ratio"], np.maximum(med["pinned_gap_pct"], 1e-3), "s--",
            color=C["vermillion"], markersize=4, label="holes fixed in advance (trailing cells)")
    ax.plot(med["fill_ratio"], np.maximum(med["free_gap_pct"], 1e-3), "o-",
            color=C["blue"], markersize=4, label="free holes (solver decides)")
    ax.set_yscale("log")
    ax.set_xlabel("fill ratio  n / (rows·cols)")
    ax.set_ylabel("gap above exact optimum (%)")
    ax.legend(fontsize=7)
    ax.set_title("hole handling quality, 96×96 grid")

    import megalap
    from bench_common import seed_raster
    side = 96
    n = int(0.7 * side * side)
    pts = make_points("meander", n, seed=0)
    seed = seed_raster(pts, total_cells=side * side)
    res = megalap.window_cleanup(pts, seed, rows=side, cols=side)
    occupied = np.zeros(side * side, dtype=bool)
    occupied[res["assignment"]] = True
    grid = occupied.reshape(side, side)
    ax = axes[1]
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#ffffff", C["blue"]]),
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("occupied cells at fill 0.7 (holes drift\nto the cloud's sparse regions)", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")
    fig.tight_layout()
    fig.savefig(FIGURES / "holes.png", bbox_inches="tight")
    plt.close(fig)


def fig_mnist_trace() -> None:
    trace = pd.read_csv(RESULTS / "mnist_trace.csv")
    summary = json.loads((RESULTS / "mnist_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(trace["elapsed_s"], trace["cost"], "-", color=C["blue"])
    coarse = trace[trace["stride"] > 1]
    ax.plot(coarse["elapsed_s"], coarse["cost"], "o", color=C["vermillion"], markersize=3,
            label="coarse-stride rounds")
    ax.set_xscale("log")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("assignment cost")
    ax.legend(fontsize=7)
    ax.set_title(f"MNIST 70k cleanup ({summary['rounds']} rounds)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "mnist_trace.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["convergence", "scaling", "ablations", "distributions", "holes", "mnist"]
    builders = {
        "convergence": fig_convergence,
        "scaling": fig_scaling,
        "ablations": fig_ablations,
        "distributions": fig_distributions,
        "holes": fig_holes,
        "mnist": fig_mnist_trace,
    }
    for name in which:
        builders[name]()
        print("built", name)


if __name__ == "__main__":
    main()
