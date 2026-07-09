"""Render a bench_billion assignment: overview, 1:1 crop, input cloud.

Memory-lean: points are regenerated chunk by chunk (same stream as
bench_billion.generate_meander), colors are computed per chunk, and the
overview is accumulated with bincount into an out_size^2 buffer, so beyond
the coordinates and the int32 assignment nothing point-sized is held.

Writes into results/renders/:
  billion_{tag}_points.png     downsampled input cloud scatter
  billion_{tag}_overview.png   full grid, block-averaged to out_size^2
  billion_{tag}_overview_t.png same, colored by walk index instead of position
  billion_{tag}_crop.png       crop_size^2 cells at 1:1 from the grid center
  billion_{tag}_crop_t.png     same crop, walk-index colors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_billion import CHUNK, generate_meander
from bench_common import results_dir
from build_diagrams import lab_to_srgb


def graph_paper_dim(colors: np.ndarray, pts: np.ndarray,
                    spacing: float = 0.1, half_thickness: float = 0.01) -> np.ndarray:
    offset = np.abs(pts - np.round(pts / spacing) * spacing)
    colors[(offset <= half_thickness).any(axis=1)] *= 0.5
    return colors


def walk_index_colors(idx: np.ndarray, n: int) -> np.ndarray:
    """Cyclic hue over walk time: shows which epoch of the walk went where."""
    cmap = plt.get_cmap("twilight")
    return cmap(idx.astype(np.float64) / n)[:, :3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-size", type=int, default=4096)
    parser.add_argument("--crop-size", type=int, default=1024)
    args = parser.parse_args()

    side, out = args.side, args.out_size
    n = side * side
    tag = f"{side}_seed{args.seed}"
    render_dir = results_dir() / "renders"
    render_dir.mkdir(exist_ok=True)

    t0 = time.time()
    pts = generate_meander(n, seed=args.seed)
    print(f"points regenerated in {time.time() - t0:.1f}s", flush=True)
    assignment = np.load(results_dir() / "assignments" / f"billion_meander_{tag}.npy")
    assert assignment.shape[0] == n

    # ---- input cloud scatter (downsampled)
    sample = np.unique(np.random.default_rng(0).integers(0, n, 2_500_000))
    sample_pts = pts[sample]
    colors = graph_paper_dim(lab_to_srgb(sample_pts), sample_pts)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=256)
    ax.scatter(sample_pts[:, 0], sample_pts[:, 1], s=0.05, c=colors, linewidths=0)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.1)
    fig.savefig(render_dir / f"billion_{tag}_points.png", bbox_inches="tight")
    plt.close(fig)
    del sample, sample_pts, colors
    print("input cloud rendered", flush=True)

    # ---- overview: block-average cell colors into out x out
    t0 = time.time()
    acc = np.zeros((2, 3, out * out), dtype=np.float64)  # [variant, channel, pixel]
    count = np.zeros(out * out, dtype=np.int64)
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        block = pts[i0:i1]
        cells = assignment[i0:i1].astype(np.int64)
        cell_row = cells // side
        cell_col = cells - cell_row * side
        flat = (cell_row * out // side) * out + (cell_col * out // side)
        del cells, cell_row, cell_col
        colors_pos = graph_paper_dim(lab_to_srgb(block), block)
        colors_t = walk_index_colors(np.arange(i0, i1), n)
        for ch in range(3):
            acc[0, ch] += np.bincount(flat, weights=colors_pos[:, ch], minlength=out * out)
            acc[1, ch] += np.bincount(flat, weights=colors_t[:, ch], minlength=out * out)
        count += np.bincount(flat, minlength=out * out)
    denom = np.maximum(count, 1).astype(np.float64)
    for variant, name in ((0, "overview"), (1, "overview_t")):
        image = (acc[variant] / denom).T.reshape(out, out, 3)
        plt.imsave(render_dir / f"billion_{tag}_{name}.png", np.clip(image, 0.0, 1.0))
    del acc, count, denom
    print(f"overview rendered in {time.time() - t0:.1f}s", flush=True)

    # ---- 1:1 crop from the grid center
    t0 = time.time()
    owner = np.full(side * side, -1, dtype=np.int32)
    owner[assignment.astype(np.int64)] = np.arange(n, dtype=np.int32)
    cs = args.crop_size
    r0 = (side - cs) // 2
    ids = owner.reshape(side, side)[r0:r0 + cs, r0:r0 + cs].ravel().astype(np.int64)
    del owner
    assert (ids >= 0).all(), "crop contains unassigned cells"
    crop_pts = pts[ids]
    image = graph_paper_dim(lab_to_srgb(crop_pts), crop_pts).reshape(cs, cs, 3)
    plt.imsave(render_dir / f"billion_{tag}_crop.png", np.clip(image, 0.0, 1.0))
    image_t = walk_index_colors(ids, n).reshape(cs, cs, 3)
    plt.imsave(render_dir / f"billion_{tag}_crop_t.png", np.clip(image_t, 0.0, 1.0))
    print(f"crop rendered in {time.time() - t0:.1f}s", flush=True)
    print(f"wrote renders into {render_dir}", flush=True)


if __name__ == "__main__":
    main()
