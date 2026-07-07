"""Algorithm-explanation figures: strided window schematic and a
multiscale-in-action sequence.

Writes paper/figures/strided_windows.png and paper/figures/multiscale_action.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))

import megalap
from bench_common import make_points, seed_raster

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "paper" / "figures"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
GRAY = "#bbbbbb"


def fig_windows_and_phases() -> None:
    """Three panels: a window move, one tiling (phase), and overlapping phases."""
    w = 6
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.75))

    def draw_grid(ax, side):
        ax.set_xlim(-0.5, side - 0.5)
        ax.set_ylim(side - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for k in range(side + 1):
            ax.axhline(k - 0.5, color="#e8e8e8", linewidth=0.4, zorder=0)
            ax.axvline(k - 0.5, color="#e8e8e8", linewidth=0.4, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#999999")

    # (a) one window move: points inside a window, lines to their new cells
    ax = axes[0]
    side = 12
    draw_grid(ax, side)
    rng = np.random.default_rng(4)
    ax.add_patch(Rectangle((2.5, 2.5), w, w, facecolor="#f2f5f8",
                           edgecolor=VERMILLION, linewidth=2.0, zorder=1))
    cells = [(r, c) for r in range(3, 9) for c in range(3, 9)]
    picked = rng.choice(len(cells), size=8, replace=False)
    targets_rc = [cells[i] for i in rng.permutation(picked)]
    for idx, tgt in zip(picked, targets_rc):
        r, c = cells[idx]
        px = c + rng.uniform(-1.4, 1.4)
        py = r + rng.uniform(-1.4, 1.4)
        px, py = np.clip([px, py], 2.8, 8.2)
        ax.plot([px, tgt[1]], [py, tgt[0]], color="#b8c4cc", linewidth=0.9, zorder=2)
        ax.plot(tgt[1], tgt[0], "s", color="#c9d4da", markersize=4.5, zorder=2)
        ax.plot(px, py, "o", color=BLUE, markersize=4, zorder=3)
    ax.set_title("a window move: the points in a\nwindow are re-matched to its cells", fontsize=8)

    # (b) one phase: a tiling of disjoint windows
    ax = axes[1]
    side = 18
    draw_grid(ax, side)
    for i in range(side // w):
        for j in range(side // w):
            shade = "#f2f5f8" if (i + j) % 2 == 0 else "#dfe8ef"
            ax.add_patch(Rectangle((j * w - 0.5, i * w - 0.5), w, w,
                                   facecolor=shade, edgecolor="#8ab0cc",
                                   linewidth=1.2, zorder=1))
    ax.set_title("a phase: disjoint windows tile the\ngrid and are solved in parallel", fontsize=8)

    # (c) phases overlap: shifted tiling on top
    ax = axes[2]
    draw_grid(ax, side)
    for i in range(side // w):
        for j in range(side // w):
            ax.add_patch(Rectangle((j * w - 0.5, i * w - 0.5), w, w,
                                   facecolor="#f2f5f8", edgecolor="#8ab0cc",
                                   linewidth=1.2, zorder=1))
    ax.add_patch(Rectangle((w / 2 - 0.5 + w, w / 2 - 0.5), w, w, facecolor="none",
                           edgecolor=VERMILLION, linewidth=2.0, linestyle="--", zorder=3))
    ax.set_title("phases are offset by half a window,\nso neighboring windows exchange", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "windows_and_phases.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_strided_windows() -> None:
    """Left: stride-1 window tiling. Right: one stride-4 window's cells."""
    side = 24
    w = 6
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))

    for ax in axes:
        ax.set_xlim(-0.5, side - 0.5)
        ax.set_ylim(side - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for k in range(side + 1):
            ax.axhline(k - 0.5, color="#eeeeee", linewidth=0.4, zorder=0)
            ax.axvline(k - 0.5, color="#eeeeee", linewidth=0.4, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#999999")

    # stride 1: shade the 16 tiles, highlight one
    ax = axes[0]
    for i in range(side // w):
        for j in range(side // w):
            shade = "#f2f5f8" if (i + j) % 2 == 0 else "#e2eaf1"
            ax.add_patch(Rectangle((j * w - 0.5, i * w - 0.5), w, w,
                                   facecolor=shade, edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((w - 0.5, w - 0.5), w, w, facecolor="none",
                           edgecolor=VERMILLION, linewidth=2.0, zorder=3))
    for r in range(w, 2 * w):
        for c in range(w, 2 * w):
            ax.plot(c, r, "s", color=BLUE, markersize=3.4, zorder=2)
    ax.set_title("stride 1: one window covers\n6×6 adjacent cells", fontsize=9)

    # stride 4: cells of one window are spread across the grid
    ax = axes[1]
    stride = 4
    for r in range(w):
        for c in range(w):
            ax.plot(c * stride, r * stride, "s", color=BLUE, markersize=3.4, zorder=2)
    ax.add_patch(Rectangle((-0.5, -0.5), (w - 1) * stride + 1, (w - 1) * stride + 1,
                           facecolor="none", edgecolor=VERMILLION, linewidth=2.0, zorder=3))
    # a second coset's cells, faint
    for r in range(w):
        for c in range(w):
            ax.plot(c * stride + 1, r * stride + 1, "s", color=GRAY, markersize=2.4, zorder=1)
    ax.set_title("stride 4: the same 6×6 window\ncovers every 4th cell", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "strided_windows.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def lab_to_srgb(points: np.ndarray) -> np.ndarray:
    """Map normalized 2D positions into Lab (a, b) plane and convert to sRGB."""
    lab_l = np.full(points.shape[0], 74.0)
    lab_a = (points[:, 0] - 0.5) * 160.0
    lab_b = (points[:, 1] - 0.5) * -160.0

    def invf(t):
        delta = 6.0 / 29.0
        return np.where(t > delta, t ** 3, 3.0 * delta * delta * (t - 4.0 / 29.0))

    fy = (lab_l + 16.0) / 116.0
    fx = fy + lab_a / 500.0
    fz = fy - lab_b / 200.0
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x = xn * invf(fx)
    y = yn * invf(fy)
    z = zn * invf(fz)
    r = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b = 0.0557 * x - 0.2040 * y + 1.0570 * z
    rgb = np.stack([r, g, b], axis=1)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0.0, 1.0)


def fig_multiscale_action() -> None:
    side = 512
    n = side * side
    points = make_points("meander", n, seed=0)
    colors = lab_to_srgb(points)

    assignment = seed_raster(points)
    schedule = megalap.default_stride_schedule(side, side)

    snapshots = [("seed (raster)", assignment.copy(), None)]
    capture_after = {0, 2, 4, len(schedule) - 1}
    for i, (sr, sc) in enumerate(schedule):
        result = megalap.window_cleanup(
            points, assignment, rows=side, cols=side,
            budget_seconds=0.0, strides=[(sr, sc)],
        )
        assignment = result["assignment"]
        if i in capture_after:
            snapshots.append((f"after stride-{sr} round", assignment.copy(), sr))

    result = megalap.window_cleanup(points, assignment, rows=side, cols=side)
    snapshots.append(("converged", result["assignment"], None))

    fig, axes = plt.subplots(1, len(snapshots), figsize=(7.2, 1.45))
    for ax, (title, snap, _) in zip(axes, snapshots):
        image = np.zeros((n, 3))
        image[snap] = colors
        ax.imshow(image.reshape(side, side, 3), interpolation="nearest")
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURES / "multiscale_action.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _grid_image(side: int, points: np.ndarray, assignment: np.ndarray,
                usable: np.ndarray | None = None) -> np.ndarray:
    """Grid raster: each cell takes its occupant's position color; empty
    usable cells are near-white, masked-out cells pure white."""
    colors = lab_to_srgb(points)
    image = np.ones((side * side, 3))
    if usable is not None:
        image[usable] = [0.93, 0.93, 0.93]
    else:
        image[:] = [0.93, 0.93, 0.93]
    image[assignment] = colors
    return image.reshape(side, side, 3)


def fig_masks() -> None:
    """Where the empty cells go: free drift vs. shaped masks."""
    side = 128
    m = side * side
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))

    # (a) free holes at fill 0.85
    n = int(0.85 * m)
    points = make_points("meander", n, seed=0)
    _, assignment, _ = megalap.snap_to_grid(points, width=side, height=side)
    axes[0].imshow(_grid_image(side, points, assignment), interpolation="nearest")
    axes[0].set_title("free holes (fill 0.85):\nempty cells drift to sparse regions", fontsize=8)

    # (b) circle mask
    yy, xx = np.mgrid[0:side, 0:side]
    circle = (xx - (side - 1) / 2) ** 2 + (yy - (side - 1) / 2) ** 2 <= (side / 2 - 1) ** 2
    n = int(circle.sum())
    points = make_points("meander", n, seed=0)
    _, assignment, _ = megalap.snap_to_grid(points, mask=circle)
    axes[1].imshow(_grid_image(side, points, assignment, np.flatnonzero(circle.reshape(-1))),
                   interpolation="nearest")
    axes[1].set_title("circle mask:\nonly masked-in cells are used", fontsize=8)

    # (c) cut corners
    r = side // 3
    rounded = (np.minimum(xx, side - 1 - xx) + np.minimum(yy, side - 1 - yy)) >= r // 2
    n = int(rounded.sum())
    points = make_points("meander", n, seed=0)
    _, assignment, _ = megalap.snap_to_grid(points, mask=rounded)
    axes[2].imshow(_grid_image(side, points, assignment, np.flatnonzero(rounded.reshape(-1))),
                   interpolation="nearest")
    axes[2].set_title("cut-corner mask:\nany shape works", fontsize=8)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
    fig.tight_layout()
    fig.savefig(FIGURES / "masks.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_windows_and_phases()
    print("built windows_and_phases")
    fig_strided_windows()
    print("built strided_windows")
    fig_multiscale_action()
    print("built multiscale_action")
    fig_masks()
    print("built masks")
