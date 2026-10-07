"""Figure: MNIST UMAP (20k subset) on masks and on a grid with 10% surplus cells.

Every point is colored by its input position (same colors in all panels); empty cells
are white.  Assignments use instance seed 1 and are cached.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shufflesnap as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
import figstyle as fs  # noqa: E402
import methods  # noqa: E402
import reference  # noqa: E402
import specs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figures")
CACHE = os.path.join(HERE, "..", "results", "assignments", "masks")

ROWS = [("Exact-certified", None, "exact optimum"),
        ("SS-balanced", ("shufflesnap", dict(specs.DEFAULT, seed=1)), "ShuffleSnap"),
        ("BSP-OT-64", ("bspot", dict(num_plans=64)), "BSP-OT (64 plans)"),
        ("RasterFairy-master", ("rasterfairy", dict(version="master")), "RasterFairy (master)")]
GRIDS = [("disk", "disk"), ("ring", "ring"), ("srect", "'S' mask"), ("free10", "10% surplus cells")]


def get_cell(dist, n, seed, grid, lab, spec, Pn, g):
    os.makedirs(CACHE, exist_ok=True)
    tag = f"-w{spec[1]['window']}" if spec and "window" in spec[1] else ""  # cache per configuration
    path = os.path.join(CACHE, f"{dist}-{n}-{seed}-{grid}-{lab}{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    if spec is None:
        cell = reference.get(dist, n, seed, grid)["cell"]
    else:
        m, p = spec
        p = dict(p)
        if m == "bspot":
            p["variant"] = "gaussian" if g.n_cells == n else "default"
        cell, _ = methods.METHODS[m](Pn, g, **p)
    ss.validate_assignment(cell, n, g)
    np.save(path, np.asarray(cell, dtype=np.int32))
    return np.asarray(cell)


def main(dist="mnist", n=20000, seed=1):
    fs.setup()
    fig, axes = plt.subplots(len(ROWS), len(GRIDS), figsize=(7.0, 8.6), layout="constrained")
    for c, (grid, gname) in enumerate(GRIDS):
        Pn, g, _, _ = datasets.instance(dist, n, seed, grid)
        cols = fs.source_colors(Pn)
        opt = None
        for r, (lab, spec, name) in enumerate(ROWS):
            cell = get_cell(dist, n, seed, grid, lab, spec, Pn, g)
            cost = ss.assignment_cost(Pn, g, cell)
            if lab == "Exact-certified":
                opt = cost
            ax = axes[r, c]
            ax.imshow(fs.grid_image(g, cell, cols, fill=(1, 1, 1)), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            tag = "optimum" if lab == "Exact-certified" else f"+{100 * (cost / opt - 1):.2f}%"
            ax.set_title(f"{gname}: {tag}" if r == 0 else tag, fontsize=7, loc="left")
            if c == 0:
                ax.set_ylabel(name, fontsize=7)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_masks.pdf"), dpi=300)
    fig.savefig(os.path.join(OUT, "fig_masks.png"), dpi=130)
    print("wrote fig_masks")


if __name__ == "__main__":
    main()
