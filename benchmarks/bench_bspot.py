"""Evaluate BSP-OT (Genest et al., SIGGRAPH Asia 2025) on the paper's
point-to-grid instances.

BSP-OT constructs bijective matchings between discrete measures in loglinear
time by recursive binary space partitioning, optionally merging several
randomized plans cost-decreasingly. It optimizes exactly our objective, so
gaps are measured against the same cached exact references. Requires the
driver solvers/cpp/bspot_grid (see its header for build instructions).

Outputs:
  results/bspot_quality.csv   six families at 200x200, plans in {1, 16, 64}
  results/bspot_scaling.csv   meander, sides 71..2048, plans=16
  results/bspot_seed.csv      BSP-OT construction as the start of descent
  results/bspot_mnist.json    MNIST 70k embedding-to-grid comparison
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import megalap
from bench_common import (
    DISTRIBUTIONS,
    REPO,
    assignment_cost,
    build_target_grid,
    exact_instance,
    gap_pct,
    make_points,
    normalize,
    results_dir,
    seed_raster,
)

DRIVER = REPO / "solvers" / "cpp" / "bspot_grid"


def run_bspot(points: np.ndarray, rows: int, cols: int, plans: int,
              seed: int = 0, want_assignment: bool = False):
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        pts_path = Path(f.name)
        f.write(np.asarray(points, dtype=np.float64).tobytes(order="C"))
    out_path = pts_path.with_suffix(".out") if want_assignment else None
    cmd = [str(DRIVER), "--points", str(pts_path), "--rows", str(rows),
           "--cols", str(cols), "--plans", str(plans), "--seed", str(seed)]
    if out_path:
        cmd += ["--assignment-out", str(out_path)]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    finally:
        pts_path.unlink()
    parsed = dict(line.split("=", 1) for line in out.strip().splitlines())
    result = {
        "time_s": float(parsed["solve_time_s"]),
        "cost": float(parsed["assignment_cost"]),
        "bijective": parsed["bijective"] == "1",
    }
    assert result["bijective"]
    if out_path:
        result["assignment"] = np.fromfile(out_path, dtype=np.int64)
        out_path.unlink()
    return result


def quality() -> None:
    rows_out = []
    side = 200
    for dist in DISTRIBUTIONS:
        for instance_seed in [0, 1, 2]:
            points = make_points(dist, side * side, seed=instance_seed)
            optimum = exact_instance(dist, side, instance_seed)["cost"]
            for plans in [1, 16, 64]:
                r = run_bspot(points, side, side, plans, seed=instance_seed)
                rows_out.append({
                    "distribution": dist,
                    "instance_seed": instance_seed,
                    "n": side * side,
                    "plans": plans,
                    "time_s": r["time_s"],
                    "gap_pct": gap_pct(r["cost"], optimum),
                })
                print(rows_out[-1], flush=True)
    path = results_dir() / "bspot_quality.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {path}")


def scaling() -> None:
    rows_out = []
    for side in [71, 100, 141, 200, 283, 400, 566, 800, 1131, 1600, 2048]:
        n = side * side
        points = make_points("meander", n, seed=0)
        r = run_bspot(points, side, side, plans=16)
        optimum = exact_instance("meander", side, 0)["cost"] if side <= 200 else None
        rows_out.append({
            "side": side,
            "n": n,
            "plans": 16,
            "time_s": r["time_s"],
            "cost": r["cost"],
            "gap_pct": gap_pct(r["cost"], optimum) if optimum else None,
        })
        print(rows_out[-1], flush=True)
    path = results_dir() / "bspot_scaling.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {path}")


def as_seed() -> None:
    """BSP-OT construction as the starting point of multiscale descent."""
    rows_out = []
    side = 200
    for instance_seed in [0, 1, 2]:
        points = make_points("meander", side * side, seed=instance_seed)
        optimum = exact_instance("meander", side, instance_seed)["cost"]
        bspot = run_bspot(points, side, side, plans=16, seed=instance_seed,
                          want_assignment=True)
        result = megalap.window_cleanup(points, bspot["assignment"],
                                        rows=side, cols=side)
        rows_out.append({
            "instance_seed": instance_seed,
            "bspot_time_s": bspot["time_s"],
            "bspot_gap_pct": gap_pct(bspot["cost"], optimum),
            "descent_time_s": float(result["elapsed_s"]),
            "descent_rounds": int(result["rounds_completed"]),
            "final_gap_pct": gap_pct(float(result["final_cost"]), optimum),
            "converged": bool(result["converged"]),
        })
        print(rows_out[-1], flush=True)
    path = results_dir() / "bspot_seed.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {path}")


def mnist() -> None:
    from bench_mnist import GRID_H, GRID_W, knn_preservation, label_agreement, DATA

    embedding = np.asarray(np.load(DATA / "mnist_umap_embedding.npz")["embedding"], dtype=np.float64)
    labels = np.asarray(np.load(DATA / "mnist_openml.npz")["labels"], dtype=np.int64)
    points = normalize(embedding)
    targets = build_target_grid(GRID_W, GRID_H)

    r = run_bspot(points, GRID_H, GRID_W, plans=16, want_assignment=True)
    summary = {
        "time_s": r["time_s"],
        "cost": r["cost"],
        "knn10_preservation": knn_preservation(points, targets[r["assignment"]]),
        "label10_agreement": label_agreement(labels, targets[r["assignment"]]),
    }
    print(json.dumps(summary, indent=1))
    (results_dir() / "bspot_mnist.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    quality()
    scaling()
    as_seed()
    mnist()
