"""Figure: matched visual comparison on the full MNIST (or Fashion-MNIST) UMAP embedding.

Rows: (1) whole grid, every point colored by its input position (same colors for all
methods); (2) a zoomed crop around a cluster boundary; (3) displacement magnitude;
(4) neighbourhood: fraction of a cell's occupied 8-neighbours whose points are among the
point's 8 nearest input neighbours.  Assignments are recomputed here with the same
settings as the benchmark (instance seed 1 for method randomness) and cached.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

import shufflesnap as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
import figstyle as fs  # noqa: E402
import methods  # noqa: E402
import reference  # noqa: E402
import specs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figures")
CACHE = os.path.join(HERE, "..", "results", "assignments", "visual")

METHODS = [
    ("Exact-certified", None),
    ("SS-balanced", ("shufflesnap", dict(specs.DEFAULT, seed=1))),
    ("BSP-OT-64", ("bspot", dict(num_plans=64, variant="gaussian"))),
    ("RasterFairy-master", ("rasterfairy", dict(version="master"))),
]
SHORT = {"Exact-certified": "exact optimum", "SS-balanced": "ShuffleSnap", "BSP-OT-64": "BSP-OT (64 plans)",
         "RasterFairy-master": "RasterFairy (master)"}


def assignments(dist, n=70000, seed=1):
    os.makedirs(CACHE, exist_ok=True)
    Pn, g, labels, raw = datasets.instance(dist, n, seed)
    out = {}
    for lab, spec in METHODS:
        tag = f"-w{spec[1]['window']}" if spec and "window" in spec[1] else ""  # cache per configuration
        path = os.path.join(CACHE, f"{dist}-{n}-{seed}-{lab}{tag}.npy")
        if os.path.exists(path):
            out[lab] = np.load(path)
            continue
        if spec is None:
            cell = reference.get(dist, n, seed)["cell"]
        else:
            m, p = spec
            cell, _ = methods.METHODS[m](Pn, g, **p)
        ss.validate_assignment(cell, n, g)
        np.save(path, np.asarray(cell, dtype=np.int32))
        out[lab] = np.asarray(cell)
    return Pn, g, labels, out


def neighbourhood_map(Pn, g, cell):
    n = len(Pn)
    occ = np.full(g.mask.shape, -1, dtype=np.int64)
    lat = g.lattice[cell]
    occ[lat[:, 1], lat[:, 0]] = np.arange(n)
    _, nn = cKDTree(Pn).query(Pn, k=9)
    nn = nn[:, 1:]
    H, W = g.mask.shape
    hit = np.zeros(n)
    tot = np.zeros(n)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            x, y = lat[:, 0] + dx, lat[:, 1] + dy
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            j = np.full(n, -1)
            j[ok] = occ[y[ok], x[ok]]
            ok &= j >= 0
            tot += ok
            hit += ok & (nn == j[:, None]).any(1)
    return hit / np.maximum(tot, 1)


def main(dist="mnist"):
    fs.setup()
    Pn, g, labels, cells = assignments(dist)
    n = len(Pn)
    cols = fs.source_colors(Pn)
    # crop window: centered on the densest cluster boundary region (fixed, same for all)
    cx, cy, half = int(g.width * 0.52), int(g.height * 0.45), 32
    k = len(METHODS)
    fig, axes = plt.subplots(4, k, figsize=(7.0, 6.2), layout="constrained",
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 1]))
    disp_max = None
    stats = {}
    for c, (lab, _) in enumerate(METHODS):
        cell = cells[lab]
        cost = ss.assignment_cost(Pn, g, cell)
        d = np.sqrt(((Pn - g.centers[cell]) ** 2).sum(1))
        nb = neighbourhood_map(Pn, g, cell)
        stats[lab] = dict(mean_sq=cost / n, mean_disp=float(d.mean()), p99=float(np.quantile(d, 0.99)),
                          lattice_precision=float(nb.mean()))
        if disp_max is None:
            disp_max = float(np.quantile(d, 0.995))
        img = fs.grid_image(g, cell, cols)
        ax = axes[0, c]
        ax.imshow(img, interpolation="nearest")
        ax.add_patch(Rectangle((cx - half - 0.5, cy - half - 0.5), 2 * half, 2 * half, fill=False, ec=fs.INK, lw=0.6))
        ax.set_title(f"{SHORT[lab]}\n{cost / n:.1f}", fontsize=7)
        ax = axes[1, c]
        ax.imshow(img[cy - half:cy + half, cx - half:cx + half], interpolation="nearest")
        ax = axes[2, c]
        dimg = np.full((g.height, g.width), np.nan)
        lat = g.lattice[cell]
        dimg[lat[:, 1], lat[:, 0]] = d
        cm = matplotlib.colormaps["Blues"].copy()
        cm.set_bad("white")
        im_d = ax.imshow(dimg, cmap=cm, vmin=0, vmax=disp_max, interpolation="nearest")
        ax = axes[3, c]
        nimg = np.full((g.height, g.width), np.nan)
        nimg[lat[:, 1], lat[:, 0]] = nb
        cm2 = matplotlib.colormaps["Oranges"].copy()
        cm2.set_bad("white")
        im_n = ax.imshow(nimg, cmap=cm2, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xlabel(f"mean {nb.mean():.2f}", fontsize=6.5)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(fs.AXIS)
            sp.set_linewidth(0.5)
    axes[0, 0].set_ylabel("color = input position", fontsize=7)
    axes[1, 0].set_ylabel("zoom (64×64 cells)", fontsize=7)
    axes[2, 0].set_ylabel("displacement", fontsize=7)
    axes[3, 0].set_ylabel("lattice precision", fontsize=7)
    cb = fig.colorbar(im_d, ax=axes[2, :], shrink=0.8, pad=0.01)
    cb.set_label("cells", fontsize=6.5)
    cb.outline.set_visible(False)
    cb2 = fig.colorbar(im_n, ax=axes[3, :], shrink=0.8, pad=0.01)
    cb2.set_label("fraction", fontsize=6.5)
    cb2.outline.set_visible(False)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"fig_visual_{dist}.pdf"), dpi=300)
    fig.savefig(os.path.join(OUT, f"fig_visual_{dist}.png"), dpi=150)
    with open(os.path.join(HERE, "..", "results", f"visual_{dist}_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    # the input embedding itself, same colors (small companion panel)
    fig2, ax = plt.subplots(figsize=(2.2, 2.2), layout="constrained")
    ax.scatter(Pn[:, 0], Pn[:, 1], s=0.05, c=cols, linewidths=0, rasterized=True)
    ax.set_xlim(-1, g.width)
    ax.set_ylim(g.height, -1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(fs.AXIS)
    ax.add_patch(Rectangle((cx - half - 0.5, cy - half - 0.5), 2 * half, 2 * half, fill=False, ec=fs.INK, lw=0.6))
    fig2.savefig(os.path.join(OUT, f"fig_input_{dist}.pdf"), dpi=300)
    fig2.savefig(os.path.join(OUT, f"fig_input_{dist}.png"), dpi=150)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mnist")
