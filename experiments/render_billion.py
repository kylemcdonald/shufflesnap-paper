"""Render the billion-point assignment partly interpolated from the points to their cells.

Port of the earlier work's visualizer (../../shufflesnap-paper/benchmarks/render_billion_test.py):
every point is splatted at (1 - t) * source + t * assigned cell center, in unit-square
coordinates (cell centers at linspace(0, 1, side)), 1 pixel per point, colored by its source
position (Lab plane) and averaged per pixel; empty pixels stay white.  Where neighboring cells
hold points from distant sources, the partial collapse opens white seams.  The points are
regenerated chunk by chunk; beyond them and the int32 assignment nothing point-sized is held.

    python experiments/render_billion.py                       # results/billion assignment
    python experiments/render_billion.py --assignment X.npy --name old
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from billion import CHUNK, OUT, generate_meander


def lab_to_srgb(points: np.ndarray) -> np.ndarray:
    """Unit-square positions onto the Lab (a, b) plane at L = 74, then sRGB (as the earlier figures)."""
    lab_l = np.full(points.shape[0], 74.0)
    lab_a = (points[:, 0] - 0.5) * 160.0
    lab_b = (points[:, 1] - 0.5) * -160.0

    def invf(t):
        delta = 6.0 / 29.0
        return np.where(t > delta, t ** 3, 3.0 * delta * delta * (t - 4.0 / 29.0))

    fy = (lab_l + 16.0) / 116.0
    x = 0.95047 * invf(fy + lab_a / 500.0)
    y = invf(fy)
    z = 1.08883 * invf(fy - lab_b / 200.0)
    rgb = np.stack([3.2406 * x - 1.5372 * y - 0.4986 * z,
                    -0.9689 * x + 1.8758 * y + 0.0415 * z,
                    0.0557 * x - 0.2040 * y + 1.0570 * z], axis=1)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=31623)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assignment", help="default: results/billion/billion_assignment_{side}_seed{seed}.npy")
    ap.add_argument("--name", default="shufflesnap")
    ap.add_argument("--interp", type=float, default=0.75)
    ap.add_argument("--out-size", type=int, default=10000)
    args = ap.parse_args()
    side, out, t = args.side, args.out_size, args.interp
    n = side * side
    path = args.assignment or OUT / f"billion_assignment_{side}_seed{args.seed}.npy"

    t0 = time.time()
    pts = generate_meander(n, args.seed)
    cell = np.load(path, mmap_mode="r")
    assert cell.shape == (n,)
    print(f"points regenerated in {time.time() - t0:.1f}s", flush=True)
    axis = np.linspace(0.0, 1.0, side)
    acc = np.zeros((3, out * out), dtype=np.float64)
    count = np.zeros(out * out, dtype=np.int64)
    t0 = time.time()
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        block = pts[i0:i1]
        c = np.asarray(cell[i0:i1], dtype=np.int64)
        row = c // side
        col = c - row * side
        x = (1.0 - t) * block[:, 0] + t * axis[col]
        y = (1.0 - t) * block[:, 1] + t * axis[row]
        flat = (np.clip((y * out).astype(np.int64), 0, out - 1) * out
                + np.clip((x * out).astype(np.int64), 0, out - 1))
        colors = lab_to_srgb(block)
        for ch in range(3):
            acc[ch] += np.bincount(flat, weights=colors[:, ch], minlength=out * out)
        count += np.bincount(flat, minlength=out * out)
    image = np.ones((out * out, 3), dtype=np.float64)
    covered = count > 0
    image[covered] = (acc[:, covered] / count[covered]).T
    dest = OUT / f"billion_{args.name}_interp{int(round(100 * t))}_{out}.png"
    Image.fromarray((np.clip(image.reshape(out, out, 3), 0, 1) * 255 + 0.5).astype(np.uint8)).save(dest)
    print(f"wrote {dest} in {time.time() - t0:.1f}s; empty pixels {1 - covered.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
