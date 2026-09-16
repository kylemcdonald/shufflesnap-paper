"""Evaluate BSP-OT (Genest et al., SIGGRAPH Asia 2025) on the paper's
point-to-grid instances.

BSP-OT constructs bijective matchings between discrete measures in loglinear
time by recursive binary space partitioning, optionally merging several
randomized plans cost-decreasingly. It optimizes exactly our objective, so
gaps are measured against the same cached exact references. Requires the
driver solvers/cpp/bspot_grid (see its header for build instructions).

Outputs:
  results/bspot_quality.csv   six families at 200x200, plans in {1, 16, 64}
  results/bspot_time_matched.csv
                              ShuffleSnap at each 16-plan BSP-OT time budget
  results/bspot_scaling.csv   meander, sides 71..2048, plans=16
  results/bspot_seed.csv      BSP-OT construction as the start of descent
  results/bspot_mnist.json    MNIST 70k embedding-to-grid comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shufflesnap
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
    seed_default,
)

DRIVER = REPO / "solvers" / "cpp" / "bspot_grid"
TIME_MATCHED_REPEATS = 5


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


def time_matched() -> None:
    """Measure ShuffleSnap at the per-instance 16-plan BSP-OT time budgets.

    ``window_cleanup`` only checks its budget between complete rounds, so a
    direct budgeted call can overrun a sub-second budget by almost one round.
    We instead trace a run requested with each instance's individual BSP-OT
    time and retain the last complete ShuffleSnap round at or before that budget.
    The seed itself is the legal t=0 checkpoint.
    This is a conservative, strict time-matched comparison of cleanup time;
    like the existing table, it excludes input construction and seeding.
    """
    bspot_path = results_dir() / "bspot_quality.csv"
    with bspot_path.open(newline="") as f:
        bspot_rows = [
            row for row in csv.DictReader(f)
            if int(row["plans"]) == 16
        ]
    if not bspot_rows:
        raise RuntimeError(f"no 16-plan BSP-OT rows in {bspot_path}")

    rows_out = []
    for bspot in bspot_rows:
        dist = bspot["distribution"]
        instance_seed = int(bspot["instance_seed"])
        n = int(bspot["n"])
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(f"time-matched benchmark requires a square grid, got n={n}")
        bspot_time_s = float(bspot["time_s"])
        bspot_gap_pct = float(bspot["gap_pct"])

        points = make_points(dist, n, seed=instance_seed)
        targets = build_target_grid(side, side)
        optimum = exact_instance(dist, side, instance_seed)["cost"]
        for rep in range(TIME_MATCHED_REPEATS):
            t0 = time.perf_counter()
            seed_assignment = seed_default(points, n)
            seed_time_s = time.perf_counter() - t0
            seed_cost = assignment_cost(points, targets, seed_assignment)
            result = shufflesnap.window_cleanup(
                points,
                seed_assignment,
                rows=side,
                cols=side,
                budget_seconds=bspot_time_s,
                trace_rounds=True,
            )

            checkpoints = [(0, 0.0, seed_cost)]
            checkpoints.extend(
                (round_idx, float(elapsed), float(cost))
                for round_idx, (elapsed, cost) in enumerate(zip(
                    result["round_elapsed_s"], result["round_costs"]
                ), start=1)
            )
            round_idx, matched_elapsed_s, matched_cost = max(
                (checkpoint for checkpoint in checkpoints if checkpoint[1] <= bspot_time_s),
                key=lambda checkpoint: checkpoint[1],
            )
            row = {
                "distribution": dist,
                "instance_seed": instance_seed,
                "rep": rep,
                "n": n,
                "bspot_time_s": bspot_time_s,
                "bspot_gap_pct": bspot_gap_pct,
                "shufflesnap_budget_s": bspot_time_s,
                "shufflesnap_gap_pct": gap_pct(matched_cost, optimum),
                "shufflesnap_elapsed_s": matched_elapsed_s,
                "rounds_within_budget": round_idx,
                "seed_time_s": seed_time_s,
                "trace_run_elapsed_s": float(result["elapsed_s"]),
            }
            rows_out.append(row)
            print(row, flush=True)

    path = results_dir() / "bspot_time_matched.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows_out[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows_out)
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
        result = shufflesnap.window_cleanup(points, bspot["assignment"],
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time-matched", action="store_true",
        help="run only the per-instance ShuffleSnap-at-BSP-OT-time experiment",
    )
    args = parser.parse_args()
    if args.time_matched:
        time_matched()
    else:
        quality()
        time_matched()
        scaling()
        as_seed()
        mnist()
