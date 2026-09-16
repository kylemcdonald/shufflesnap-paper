"""MNIST atlas: end-to-end example with quality metrics.

Uses the cached UMAP embedding of all 70,000 MNIST digits (see
paper/data/mnist_showcase/) and assigns them to a 250x280 grid with the
multiscale solver. Records the convergence trace, k-nearest-neighbor
preservation, and label agreement, and renders the atlas.

To regenerate the embedding from scratch: pip install umap-learn, delete
mnist_umap_embedding.npz, and run with --rebuild-embedding.

Outputs:
  results/mnist_trace.csv
  results/mnist_summary.json
  ../paper/figures/mnist_atlas_full_with_crop_box.png   (downscaled for print)
  ../paper/figures/mnist_atlas_detail.png               (native resolution)
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import shufflesnap
from bench_common import (
    assignment_cost,
    build_target_grid,
    results_dir,
    rasterfairy_assignment,
    seed_default,
)

GRID_W, GRID_H = 250, 280
MARGIN = 0.03
CROP_CELLS = 24
TILE = 28
DOWNSCALE = 4  # full-atlas print figure is reduced by this factor
DATA = REPO / "paper" / "data" / "mnist_showcase"
FIGURES = REPO / "paper" / "figures"


# ------------------------------------------------------------------- render

def png_chunk(tag: bytes, payload: bytes) -> bytes:
    chunk = tag + payload
    return struct.pack(">I", len(payload)) + chunk + struct.pack(">I", zlib.crc32(chunk))


def write_png(path: Path, rgb: np.ndarray) -> None:
    height, width = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    payload = zlib.compress(raw, 9)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    blob = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header) + png_chunk(b"IDAT", payload) + png_chunk(b"IEND", b"")
    path.write_bytes(blob)


def render_digit_atlas(images: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    canvas = np.full((GRID_H * TILE, GRID_W * TILE), 255, dtype=np.uint8)
    for point_id in range(images.shape[0]):
        target_id = int(assignment[point_id])
        row, col = divmod(target_id, GRID_W)
        canvas[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE] = 255 - images[point_id]
    return np.repeat(canvas[:, :, None], 3, axis=2)


def downscale(rgb: np.ndarray, factor: int) -> np.ndarray:
    h = rgb.shape[0] // factor * factor
    w = rgb.shape[1] // factor * factor
    blocks = rgb[:h, :w].reshape(h // factor, factor, w // factor, factor, 3)
    return blocks.mean(axis=(1, 3)).astype(np.uint8)


def build_label_grid(labels: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), -1, dtype=np.int16)
    for point_id, target_id in enumerate(np.asarray(assignment, dtype=np.int64)):
        row, col = divmod(int(target_id), GRID_W)
        grid[row, col] = int(labels[point_id])
    return grid


def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return -1.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-np.sum(probs * np.log2(probs)))


def choose_detail_crop(label_grid: np.ndarray) -> tuple[int, int]:
    rows, cols = label_grid.shape
    window = min(CROP_CELLS, rows, cols)
    best, best_score = (0, 0), -1.0
    for row0 in range(0, rows - window + 1, 2):
        for col0 in range(0, cols - window + 1, 2):
            patch = label_grid[row0:row0 + window, col0:col0 + window]
            real = patch[patch >= 0]
            if real.size < max(16, window * window // 3):
                continue
            score = entropy_from_counts(np.bincount(real.astype(np.int64), minlength=10))
            if score > best_score:
                best_score, best = score, (row0, col0)
    return best


def draw_highlight_box(rgb: np.ndarray, row0: int, col0: int) -> np.ndarray:
    out = rgb.copy()
    y0, x0 = row0 * TILE, col0 * TILE
    y1 = min(out.shape[0], (row0 + CROP_CELLS) * TILE)
    x1 = min(out.shape[1], (col0 + CROP_CELLS) * TILE)
    color = np.array([220, 36, 36], dtype=np.uint8)
    t = 14
    out[y0:y0 + t, x0:x1] = color
    out[y1 - t:y1, x0:x1] = color
    out[y0:y1, x0:x0 + t] = color
    out[y0:y1, x1 - t:x1] = color
    return out


# ------------------------------------------------------------------ metrics

def knn_preservation(embedded: np.ndarray, grid_positions: np.ndarray, k: int = 10) -> float:
    """Mean fraction of each point's k nearest embedding neighbors that are
    also among its k nearest grid neighbors."""
    from sklearn.neighbors import NearestNeighbors

    idx_embed = NearestNeighbors(n_neighbors=k + 1).fit(embedded).kneighbors(embedded)[1]
    idx_grid = NearestNeighbors(n_neighbors=k + 1).fit(grid_positions).kneighbors(grid_positions)[1]
    overlap = np.empty(len(embedded))
    for i in range(len(embedded)):
        overlap[i] = len(set(idx_embed[i, 1:]) & set(idx_grid[i, 1:])) / k
    return float(overlap.mean())


def label_agreement(labels: np.ndarray, positions: np.ndarray, k: int = 10) -> float:
    """Mean fraction of each point's k nearest neighbors sharing its label."""
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(positions).kneighbors(positions)[1]
    return float((labels[idx[:, 1:]] == labels[:, None]).mean())


# --------------------------------------------------------------------- main

def rebuild_embedding() -> None:
    from sklearn.datasets import fetch_openml
    import umap

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    images = np.asarray(mnist.data, dtype=np.uint8).reshape(-1, 28, 28)
    labels = np.asarray(mnist.target, dtype=np.int16)
    np.savez_compressed(DATA / "mnist_openml.npz", images=images, labels=labels)
    flat = images.reshape(len(images), -1).astype(np.float32) / 255.0
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.05,
                        metric="euclidean", random_state=0, low_memory=True, verbose=True)
    embedding = reducer.fit_transform(flat)
    np.savez_compressed(DATA / "mnist_umap_embedding.npz",
                        embedding=np.asarray(embedding, dtype=np.float64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-embedding", action="store_true")
    args = parser.parse_args()
    if args.rebuild_embedding or not (DATA / "mnist_umap_embedding.npz").exists():
        rebuild_embedding()

    embedding = np.asarray(np.load(DATA / "mnist_umap_embedding.npz")["embedding"], dtype=np.float64)
    cached = np.load(DATA / "mnist_openml.npz")
    images = np.asarray(cached["images"], dtype=np.uint8)
    labels = np.asarray(cached["labels"], dtype=np.int64)
    n = embedding.shape[0]
    assert n == GRID_W * GRID_H

    from bench_common import normalize
    points = normalize(embedding)
    targets = build_target_grid(GRID_W, GRID_H, MARGIN)

    t0 = time.time()
    seed = seed_default(points, GRID_W * GRID_H)
    seed_time = time.time() - t0
    seed_cost = assignment_cost(points, targets, seed)

    result = shufflesnap.window_cleanup(
        points, seed, rows=GRID_H, cols=GRID_W, budget_seconds=60.0, trace_rounds=True,
    )
    assignment = result["assignment"]

    # construction-only baseline: RasterFairy (Klingemann 2016)
    bisect, bisect_time = rasterfairy_assignment(points, GRID_W, GRID_H)
    bisect_cost = assignment_cost(points, targets, bisect)

    metrics_t0 = time.time()
    summary = {
        "n": int(n),
        "grid": [GRID_W, GRID_H],
        "seed_time_s": seed_time,
        "cleanup_elapsed_s": float(result["elapsed_s"]),
        "rounds": int(result["rounds_completed"]),
        "converged": bool(result["converged"]),
        "seed_cost": seed_cost,
        "final_cost": float(result["final_cost"]),
        "rasterfairy_time_s": bisect_time,
        "rasterfairy_cost": bisect_cost,
        "knn10_preservation_seed": knn_preservation(points, targets[seed]),
        "knn10_preservation_final": knn_preservation(points, targets[assignment]),
        "knn10_preservation_rasterfairy": knn_preservation(points, targets[bisect]),
        "label10_agreement_embedding": label_agreement(labels, points),
        "label10_agreement_seed": label_agreement(labels, targets[seed]),
        "label10_agreement_final": label_agreement(labels, targets[assignment]),
        "label10_agreement_rasterfairy": label_agreement(labels, targets[bisect]),
    }
    summary["metrics_time_s"] = time.time() - metrics_t0
    print(json.dumps(summary, indent=1))
    (results_dir() / "mnist_summary.json").write_text(json.dumps(summary, indent=1))

    with (results_dir() / "mnist_trace.csv").open("w") as f:
        f.write("round,stride,elapsed_s,cost\n")
        for i, (e, c, s) in enumerate(zip(
            result["round_elapsed_s"], result["round_costs"], result["round_strides"]
        )):
            f.write(f"{i+1},{int(s)},{float(e)},{float(c)}\n")

    FIGURES.mkdir(parents=True, exist_ok=True)
    atlas = render_digit_atlas(images, assignment)
    label_grid = build_label_grid(labels, assignment)
    row0, col0 = choose_detail_crop(label_grid)
    outlined = draw_highlight_box(atlas, row0, col0)
    write_png(FIGURES / "mnist_atlas_full_with_crop_box.png", downscale(outlined, DOWNSCALE))
    crop = atlas[row0 * TILE:(row0 + CROP_CELLS) * TILE, col0 * TILE:(col0 + CROP_CELLS) * TILE]
    write_png(FIGURES / "mnist_atlas_detail.png", crop)
    print("rendered atlas figures")


if __name__ == "__main__":
    main()
