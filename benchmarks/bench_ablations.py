"""Ablations: window size, stride schedule, and thread scaling,
all from the default seed (a random permutation).

Writes results/ablation_window.csv, results/ablation_schedule.csv,
results/ablation_threads.csv.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shufflesnap
from bench_common import (
    exact_instance,
    gap_pct,
    make_points,
    results_dir,
    seed_default,
)

SIDE = 200
SEEDS = [0, 1, 2]
BUDGET_S = 20.0


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def run(points, seed_assignment, optimum, **kwargs):
    result = shufflesnap.window_cleanup(
        points, seed_assignment, rows=SIDE, cols=SIDE,
        budget_seconds=BUDGET_S, trace_rounds=True, **kwargs,
    )
    elapsed = [float(e) for e in result["round_elapsed_s"]]
    gaps = [gap_pct(float(c), optimum) for c in result["round_costs"]]

    def time_to(threshold):
        for e, g in zip(elapsed, gaps):
            if g <= threshold:
                return e
        return None

    return {
        "final_gap_pct": gaps[-1],
        "rounds": int(result["rounds_completed"]),
        "elapsed_s": float(result["elapsed_s"]),
        "converged": bool(result["converged"]),
        "time_to_1pct_s": time_to(1.0),
        "time_to_0p1pct_s": time_to(0.1),
    }


def window_ablation() -> None:
    rows = []
    for instance_seed in SEEDS:
        points = make_points("meander", SIDE * SIDE, seed=instance_seed)
        optimum = exact_instance("meander", SIDE, instance_seed)["cost"]
        seed_assignment = seed_default(points, SIDE * SIDE)
        for window in [2, 3, 4, 5, 6]:
            stats = run(points, seed_assignment, optimum, window_size=window)
            rows.append({"instance_seed": instance_seed, "window_size": window, **stats})
            print(rows[-1], flush=True)
    write_csv(results_dir() / "ablation_window.csv", rows)


def schedule_ablation() -> None:
    def descent(start: int) -> list[int]:
        strides = []
        s = start
        while s > 1:
            strides.append(s)
            s //= 2
        strides.append(1)
        return strides

    schedules = {
        "local-only (stride 1)": [1],
        "coarse-to-fine (default)": None,
        "half-depth descent": descent(8),
        "fine-to-coarse": list(reversed(descent(64))) + [1],
        "cycled descent": descent(64) * 4,
        "ratio-4 descent": [64, 16, 4, 1],
    }
    rows = []
    for instance_seed in SEEDS:
        points = make_points("meander", SIDE * SIDE, seed=instance_seed)
        optimum = exact_instance("meander", SIDE, instance_seed)["cost"]
        seed_assignment = seed_default(points, SIDE * SIDE)
        for label, strides in schedules.items():
            stats = run(points, seed_assignment, optimum, strides=strides)
            rows.append({"instance_seed": instance_seed, "schedule": label, **stats})
            print(rows[-1], flush=True)
    write_csv(results_dir() / "ablation_schedule.csv", rows)


def thread_ablation() -> None:
    side = 512
    points = make_points("meander", side * side, seed=0)
    seed_assignment = seed_default(points, side * side)
    rows = []
    for threads in [1, 2, 4, 8, 16, 24]:
        t0 = time.time()
        result = shufflesnap.window_cleanup(
            points, seed_assignment, rows=side, cols=side,
            budget_seconds=5.0, num_threads=threads,
        )
        rows.append({
            "threads": threads,
            "rounds": int(result["rounds_completed"]),
            "elapsed_s": float(result["elapsed_s"]),
            "rounds_per_s": int(result["rounds_completed"]) / float(result["elapsed_s"]),
            "wall_s": time.time() - t0,
        })
        print(rows[-1], flush=True)
    write_csv(results_dir() / "ablation_threads.csv", rows)


if __name__ == "__main__":
    window_ablation()
    schedule_ablation()
    thread_ablation()
