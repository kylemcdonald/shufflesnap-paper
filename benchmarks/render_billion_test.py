"""Render a billion-point assignment for the appendix and exploratory views.

Renders two views of a saved assignment (shufflesnap or BSP-OT) of the
bench_billion meander instance:

  {tag}_interp{P}.png       every point splatted at (1-t)*initial + t*assigned
                            cell center (t = --interp), 1 px per point, plain
                            Lab position colors, averaged per pixel.
  {tag}_graphpaper{L}.png   the assigned grid block-averaged to --out-size,
                            colored like figure 6 but with --lines lines per
                            axis (thickness stays 20% of spacing).

Memory-lean: points are regenerated chunk by chunk and the accumulation is
bincount-based, so beyond the coordinates and the int32 assignment nothing
point-sized is held.

Example:
  python render_billion_test.py --side 31623                 # shufflesnap result
  python render_billion_test.py --side 31623 --tag bspot     # BSP-OT result

The default ShuffleSnap interpolation render is copied to
paper/figures/billion_interp75.png for the appendix.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_billion import CHUNK, generate_meander
from bench_common import MARGIN, results_dir
from build_diagrams import lab_to_srgb


def graph_paper_colors_spaced(points: np.ndarray, spacing: float) -> np.ndarray:
    """build_figures.graph_paper_colors with adjustable line spacing
    (thickness stays at 20% of spacing, half brightness)."""
    colors = lab_to_srgb(points)
    half_thickness = 0.1 * spacing
    offset = np.abs(points - np.round(points / spacing) * spacing)
    colors[(offset <= half_thickness).any(axis=1)] *= 0.5
    return colors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="shufflesnap",
                        help="assignment tag: 'shufflesnap' reads billion_meander_{side}_seed{seed}.npy, "
                             "anything else reads billion_{tag}_{side}_seed{seed}.npy")
    parser.add_argument("--interp", type=float, default=0.75,
                        help="interpolation fraction toward the assigned cell center")
    parser.add_argument("--lines", type=int, default=100, help="graph-paper lines per axis")
    parser.add_argument("--out-size", type=int, default=10000)
    args = parser.parse_args()

    side, out = args.side, args.out_size
    n = side * side
    if args.tag == "shufflesnap":
        asn_name = f"billion_meander_{side}_seed{args.seed}.npy"
    else:
        asn_name = f"billion_{args.tag}_{side}_seed{args.seed}.npy"
    prefix = f"billion_{args.tag}_{side}_seed{args.seed}_test"
    render_dir = results_dir() / "renders"
    render_dir.mkdir(exist_ok=True)

    t0 = time.time()
    pts = generate_meander(n, seed=args.seed)
    print(f"points regenerated in {time.time() - t0:.1f}s", flush=True)
    assignment = np.load(results_dir() / "assignments" / asn_name)
    assert assignment.shape[0] == n

    axis = np.linspace(MARGIN, 1.0 - MARGIN, side)

    def accumulate(flat_fn, color_fn):
        acc = np.zeros((3, out * out), dtype=np.float64)
        count = np.zeros(out * out, dtype=np.int64)
        for i0 in range(0, n, CHUNK):
            i1 = min(i0 + CHUNK, n)
            block = pts[i0:i1]
            cells = assignment[i0:i1].astype(np.int64)
            flat = flat_fn(block, cells)
            colors = color_fn(block)
            for ch in range(3):
                acc[ch] += np.bincount(flat, weights=colors[:, ch], minlength=out * out)
            count += np.bincount(flat, minlength=out * out)
        image = np.ones((out * out, 3), dtype=np.float64)  # empty pixels stay white
        covered = count > 0
        image[covered] = (acc[:, covered] / count[covered]).T
        return image.reshape(out, out, 3)

    def save(image, name):
        path = render_dir / name
        Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)
        print(f"wrote {path}", flush=True)

    t = args.interp

    def interp_flat(block, cells):
        row = cells // side
        col = cells - row * side
        x = (1.0 - t) * block[:, 0] + t * axis[col]
        y = (1.0 - t) * block[:, 1] + t * axis[row]
        ix = np.clip((x * out).astype(np.int64), 0, out - 1)
        iy = np.clip((y * out).astype(np.int64), 0, out - 1)
        return iy * out + ix

    t0 = time.time()
    save(accumulate(interp_flat, lab_to_srgb),
         f"{prefix}_interp{int(round(100 * t))}.png")
    print(f"interp render in {time.time() - t0:.1f}s", flush=True)

    def grid_flat(block, cells):
        row = cells // side
        col = cells - row * side
        return (row * out // side) * out + (col * out // side)

    spacing = 1.0 / args.lines
    t0 = time.time()
    save(accumulate(grid_flat, lambda block: graph_paper_colors_spaced(block, spacing)),
         f"{prefix}_graphpaper{args.lines}.png")
    print(f"graph-paper render in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
