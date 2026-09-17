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

import shufflesnap
from bench_common import make_points, seed_default

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

    # (a) one dense window.  The cell set is spatially compact, but the source
    # positions of the points currently assigned to those cells may lie inside
    # or outside the window.
    ax = axes[0]
    side = 12
    draw_grid(ax, side)
    ax.add_patch(Rectangle((2.5, 2.5), w, w, facecolor="#f2f5f8",
                           edgecolor=VERMILLION, linewidth=2.0, zorder=1))
    for r in range(3, 9):
        for c in range(3, 9):
            ax.plot(c, r, "s", markerfacecolor="white", markeredgecolor="#aebbc4",
                    markeredgewidth=0.7, markersize=4.2, zorder=2)
    points = np.array([
        [3.4, 3.7], [5.0, 4.2], [7.5, 3.4],
        [3.9, 6.8], [6.2, 7.2], [7.8, 5.5],
        [1.0, 1.2], [10.1, 1.7], [0.9, 6.1],
        [10.2, 8.8], [5.3, 10.3], [1.7, 9.7],
    ])
    targets = np.array([
        [4, 4], [5, 4], [7, 4], [4, 7], [6, 7], [8, 6],
        [3, 3], [8, 3], [3, 6], [8, 8], [6, 8], [3, 8],
    ])
    for (px, py), (tx, ty) in zip(points, targets):
        ax.plot([px, tx], [py, ty], color="#b8c4cc", linewidth=0.8, zorder=2)
        ax.plot(px, py, "o", color=BLUE, markersize=4, zorder=3)
    ax.set_title("one dense window: all cells participate;\nassigned points may lie inside or outside", fontsize=8)

    # (b) one phase: show the entire 3 x 3 tiling of disjoint windows.
    ax = axes[1]
    side = 18
    draw_grid(ax, side)
    outline_inset = 0.12
    for i in range(side // w):
        for j in range(side // w):
            ax.add_patch(Rectangle((j * w - 0.5 + outline_inset,
                                    i * w - 0.5 + outline_inset),
                                   w - 2 * outline_inset, w - 2 * outline_inset,
                                   facecolor="none", edgecolor=VERMILLION,
                                   linewidth=1.4, linestyle=(0, (4, 2)), zorder=2))
    ax.set_title("one phase: all 9 disjoint windows\nare solved in parallel", fontsize=8)

    # (c) the actual half-window-offset phase: complete windows plus the
    # clipped windows at the far row and column edges.
    ax = axes[2]
    draw_grid(ax, side)
    shifted_bands = [(w / 2 - 0.5, w),
                     (w / 2 - 0.5 + w, w),
                     (w / 2 - 0.5 + 2 * w, w / 2)]
    for y, height in shifted_bands:
        for x, width in shifted_bands:
            ax.add_patch(Rectangle((x + outline_inset, y + outline_inset),
                                   width - 2 * outline_inset,
                                   height - 2 * outline_inset,
                                   facecolor="none", edgecolor=VERMILLION,
                                   linewidth=1.4, linestyle=(0, (4, 2)), zorder=2))
    ax.set_title("half-offset phase: 4 full and 5 clipped\nwindows match the kernel", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "windows_and_phases.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_strided_windows() -> None:
    """Left: one stride-4 window's cells. Right: stride-1 window tiling."""
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

    # stride 4: cells of one window are spread across the grid
    ax = axes[0]
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

    # stride 1: shade the 16 tiles, highlight one
    ax = axes[1]
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

    assignment = seed_default(points, n)
    schedule = shufflesnap.default_stride_schedule(side, side)

    snapshots = [("seed (random)", assignment.copy(), None)]
    capture_after = {0, 2, 4, len(schedule) - 1}
    for i, (sr, sc) in enumerate(schedule):
        result = shufflesnap.window_cleanup(
            points, assignment, rows=side, cols=side,
            budget_seconds=0.0, strides=[(sr, sc)],
        )
        assignment = result["assignment"]
        if i in capture_after:
            snapshots.append((f"after stride-{sr} round", assignment.copy(), sr))

    # The coarse-to-fine schedule above is complete; only local rounds remain.
    result = shufflesnap.window_cleanup(points, assignment, rows=side, cols=side, strides=1)
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
    """Two examples of cell occupancy fixed by a mask."""
    side = 128
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.6))

    # (a) circle mask
    yy, xx = np.mgrid[0:side, 0:side]
    circle = (xx - (side - 1) / 2) ** 2 + (yy - (side - 1) / 2) ** 2 <= (side / 2 - 1) ** 2
    n = int(circle.sum())
    points = make_points("meander", n, seed=0)
    _, assignment, _ = shufflesnap.snap_to_grid(points, mask=circle)
    axes[0].imshow(_grid_image(side, points, assignment, np.flatnonzero(circle.reshape(-1))),
                   interpolation="nearest")
    axes[0].set_title("circle mask", fontsize=8)

    # (b) cut corners
    r = side // 3
    rounded = (np.minimum(xx, side - 1 - xx) + np.minimum(yy, side - 1 - yy)) >= r // 2
    n = int(rounded.sum())
    points = make_points("meander", n, seed=0)
    _, assignment, _ = shufflesnap.snap_to_grid(points, mask=rounded)
    axes[1].imshow(_grid_image(side, points, assignment, np.flatnonzero(rounded.reshape(-1))),
                   interpolation="nearest")
    axes[1].set_title("cut-corner mask", fontsize=8)

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
