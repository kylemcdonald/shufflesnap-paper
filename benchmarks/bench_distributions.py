"""Robustness across point distributions at 200x200 with exact references.

For each distribution and instance seed, runs multiscale cleanup to
convergence from the default seed (a random permutation) and records the
trace. Writes results/distributions.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import megalap
from bench_common import (
    DISTRIBUTIONS,
    assignment_cost,
    build_target_grid,
    exact_instance,
    gap_pct,
    make_points,
    results_dir,
    seed_default,
)

SIDE = 200
SEEDS = [0, 1, 2]
BUDGET_S = 30.0


def main() -> None:
    out_path = results_dir() / "distributions.csv"
    rows: list[dict] = []
    targets = build_target_grid(SIDE, SIDE)

    for dist in DISTRIBUTIONS:
        for instance_seed in SEEDS:
            points = make_points(dist, SIDE * SIDE, seed=instance_seed)
            optimum = exact_instance(dist, SIDE, instance_seed)["cost"]
            seed_assignment = seed_default(points, SIDE * SIDE)
            result = megalap.window_cleanup(
                points, seed_assignment, rows=SIDE, cols=SIDE,
                budget_seconds=BUDGET_S, trace_rounds=True,
            )
            seed_cost = assignment_cost(points, targets, seed_assignment)
            gaps = [gap_pct(float(c), optimum) for c in result["round_costs"]]
            elapsed = [float(e) for e in result["round_elapsed_s"]]

            def time_to(threshold: float) -> float | None:
                for e, g in zip(elapsed, gaps):
                    if g <= threshold:
                        return e
                return None

            rows.append({
                "distribution": dist,
                "instance_seed": instance_seed,
                "n": SIDE * SIDE,
                "optimum": optimum,
                "seed_gap_pct": gap_pct(seed_cost, optimum),
                "final_gap_pct": gaps[-1],
                "rounds": int(result["rounds_completed"]),
                "elapsed_s": float(result["elapsed_s"]),
                "converged": bool(result["converged"]),
                "time_to_1pct_s": time_to(1.0),
                "time_to_0p1pct_s": time_to(0.1),
            })
            print(rows[-1], flush=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
