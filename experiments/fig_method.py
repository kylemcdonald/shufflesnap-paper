"""Figure 1: strided windows, shifted tilings, and coarse-to-fine progress."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import shufflesnap as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
import figstyle as fs  # noqa: E402
import reference  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
BLUE = "#2a78d6"
BLUE_LIGHT = "#b7d3f6"
GRAY = "#d7d6cf"


def panel_strided(ax):
    W, H, s, w = 32, 30, 4, 8
    a, b = 1, 2
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    ax.scatter(xs.ravel(), ys.ravel(), s=0.8, c=GRAY, linewidths=0)
    cls = (xs % s == a) & (ys % s == b)
    ax.scatter(xs[cls], ys[cls], s=5, facecolors="none", edgecolors=BLUE, linewidths=0.5)
    wx = a + s * np.arange(w)
    wy = b + s * np.arange(w)
    WX, WY = np.meshgrid(wx, wy)
    ax.scatter(WX.ravel(), WY.ravel(), s=6, c=BLUE, linewidths=0)
    ax.add_patch(Rectangle((wx[0] - 0.5, wy[0] - 0.5), wx[-1] - wx[0] + 1, wy[-1] - wy[0] + 1, fill=False,
                           ec=fs.INK, lw=0.8))
    ax.annotate("", xy=(wx[-1] + 0.5, H + 0.8), xytext=(wx[0] - 0.5, H + 0.8),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=fs.INK2, shrinkA=0, shrinkB=0))
    ax.text((wx[0] + wx[-1]) / 2, H + 1.4, f"window spans (w−1)s+1 = {(w - 1) * s + 1} cells", ha="center", va="top",
            fontsize=6.5,
            color=fs.INK2)
    ax.set_xlim(-1, W)
    ax.set_ylim(H + 3.2, -1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"(a) stride s = {s}: residue class ({a}, {b})\nand one {w}×{w} window ({w * w} cells)", loc="left")


def panel_tilings(axes):
    Wa, Hb, w = 20, 16, 8
    for ax, (ox, oy) in zip(axes, ss.default_offsets(w)):
        shx = (w - ox % w) % w
        shy = (w - oy % w) % w
        for ky in range((Hb + shy + w - 1) // w):
            for kx in range((Wa + shx + w - 1) // w):
                u0, v0 = kx * w - shx, ky * w - shy
                i0, i1 = max(0, u0), min(Wa, u0 + w)
                j0, j1 = max(0, v0), min(Hb, v0 + w)
                if i0 >= i1 or j0 >= j1:
                    continue
                clipped = (i1 - i0 < w) or (j1 - j0 < w)
                ax.add_patch(Rectangle((i0 - 0.5, j0 - 0.5), i1 - i0, j1 - j0, fc=BLUE_LIGHT if clipped else "white",
                                       ec=fs.INK, lw=0.7))
        xs, ys = np.meshgrid(np.arange(Wa), np.arange(Hb))
        ax.scatter(xs.ravel(), ys.ravel(), s=0.5, c=fs.INK2, linewidths=0)
        ax.set_xlim(-0.8, Wa - 0.2)
        ax.set_ylim(Hb - 0.2, -0.8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"shift ({ox}, {oy})", fontsize=7)


def panel_progress(fig, gs_imgs, ax_curve):
    n = 10000
    Pn, g, _, raw = datasets.instance("mnist", n, 0)
    cols = fs.source_colors(Pn)
    res = ss.assign(Pn, g, normalize="none", preset="balanced", seed=1, snapshots=True)
    ref = reference.get("mnist", n, 0, compute=False)
    snaps = res.snapshots
    # the start, every second level and the converged layout
    picks = sorted(set([0] + list(range(1, len(snaps) - 1, 2)) + [len(snaps) - 1]))
    axes = [fig.add_subplot(gs_imgs[0, k]) for k in range(len(picks) + 1)]
    ax0 = axes[0]
    ax0.scatter(Pn[:, 0], Pn[:, 1], s=0.15, c=cols, linewidths=0, rasterized=True)
    ax0.set_xlim(-1, g.width)
    ax0.set_ylim(g.height, -1)
    ax0.set_aspect("equal")
    ax0.set_xticks([])
    ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_color(fs.AXIS)
    ax0.set_title("input (UMAP)", fontsize=7)
    for ax, k in zip(axes[1:], picks):
        name, cell = snaps[k]
        ax.imshow(fs.grid_image(g, cell, cols), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(fs.AXIS)
        c = ss.assignment_cost(Pn, g, cell) / n
        label = "random start" if name == "start" else ("s = " + name.split("s=")[1].split("x")[0])
        if k == len(snaps) - 1:
            label = "s = 1 (end)"
        ax.set_title(f"{label}\n{c:.1f}", fontsize=7)
    # cost after every phase
    tr = res.trace
    y = tr[:, 1] / n
    x = np.arange(len(y))
    ax_curve.plot(x, y, color=BLUE, lw=1.4, marker="o", ms=2.5, mec="white", mew=0.5)
    if ref is not None:
        ax_curve.axhline(ref["cost"] / n, color=fs.INK, lw=0.8)
        ax_curve.text(len(y) - 1, ref["cost"] / n * 0.93, "certified optimum", ha="right", va="top", fontsize=6.5,
                      color=fs.INK2)
    ax_curve.set_yscale("log")
    stages = tr[:, 2]
    for st in np.unique(stages[stages >= 0]):
        idx = np.nonzero(stages == st)[0]
        sx = int(tr[idx[0], 5])
        ax_curve.axvspan(idx[0] - 0.5, idx[-1] + 0.5, color="#f0efec" if st % 2 == 0 else "white", zorder=0, lw=0)
        ax_curve.text((idx[0] + idx[-1]) / 2, y.max() * 1.05, f"s={sx}", ha="center", va="bottom", fontsize=6,
                      color=fs.INK2)
    ax_curve.set_xlabel("phase (four shifted tilings per round)")
    ax_curve.set_ylabel("mean squared displacement\n(cells², log scale)")
    ax_curve.set_xlim(-0.5, len(y) - 0.5)
    ax_curve.set_ylim(ref["cost"] / n * 0.8 if ref else y.min() * 0.8, y.max() * 1.6)
    fs.recess(ax_curve)
    ax_curve.set_title(f"(d) cost after every phase (MNIST UMAP subset, N = {n}, default preset)", loc="left")
    return res


def main():
    fs.setup()
    os.makedirs(OUT, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 6.2))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.3, 0.62, 1.0], hspace=0.38)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.08)
    panel_strided(fig.add_subplot(top[0]))
    tgs = top[1].subgridspec(2, 2, wspace=0.10, hspace=0.32)
    taxes = [fig.add_subplot(tgs[i // 2, i % 2]) for i in range(4)]
    panel_tilings(taxes)
    taxes[0].text(0.0, 1.32, "(b) one round = four shifted tilings of each subgrid\n     (clipped edge windows shaded)",
                  fontsize=8.5, ha="left", va="bottom", transform=taxes[0].transAxes)
    mid = outer[1].subgridspec(1, 8, wspace=0.12)
    fig.text(0.125, 0.565, "(c) coarse-to-fine progress (color = input position; number = mean squared "
             "displacement)", fontsize=8.5, ha="left")
    ax_curve = fig.add_subplot(outer[2])
    panel_progress(fig, mid, ax_curve)
    fig.savefig(os.path.join(OUT, "fig_method.pdf"))
    fig.savefig(os.path.join(OUT, "fig_method.png"), dpi=150)
    print("wrote", os.path.join(OUT, "fig_method.pdf"))


if __name__ == "__main__":
    main()
