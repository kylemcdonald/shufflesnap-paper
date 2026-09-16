"""Scaling: convergence time and quality from 5k to 4.2M points.

For each grid side, runs multiscale cleanup from the default seed (a random
permutation) with a budget cap and records the full trace. Where an exact
reference exists (side <= 200) the gap is reported against it, and the
oracle's own solve time gives the exact-solver crossover. Writes
results/scaling.csv (one row per side/seed) and results/scaling_traces.csv.
"""

from __future__ import annotations

import csv
import resource
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

SIDES = [71, 100, 141, 200, 283, 400, 566, 800, 1131, 1600, 2048]
EXACT_SIDES = {71, 100, 141, 200}
SEEDS = [0, 1, 2]
BUDGET_S = 180.0
STRESS_BUDGET_S = 3600.0  # let even the largest runs converge outright


def main() -> None:
    summary_path = results_dir() / "scaling.csv"
    trace_path = results_dir() / "scaling_traces.csv"
    summary_rows: list[dict] = []
    trace_rows: list[dict] = []

    for side in SIDES:
        n = side * side
        for instance_seed in SEEDS:
            if side > 800 and instance_seed != 0:
                continue  # largest cases are single-seed stress runs
            points = make_points("meander", n, seed=instance_seed)
            t0 = time.time()
            seed_assignment = seed_default(points, n)
            seed_time = time.time() - t0

            budget = STRESS_BUDGET_S if side > 800 else BUDGET_S
            result = shufflesnap.window_cleanup(
                points, seed_assignment, rows=side, cols=side,
                budget_seconds=budget, trace_rounds=True,
            )
            optimum = None
            oracle_time = None
            if side in EXACT_SIDES:
                entry = exact_instance("meander", side, instance_seed)
                optimum = entry["cost"]
                oracle_time = entry.get("solve_time_s")

            elapsed = [float(e) for e in result["round_elapsed_s"]]
            costs = [float(c) for c in result["round_costs"]]
            final = float(result["final_cost"])

            def time_within_of_final(ratio: float) -> float | None:
                target = final * (1.0 + ratio)
                for e, c in zip(elapsed, costs):
                    if c <= target:
                        return e
                return None

            def time_to_gap(threshold_pct: float) -> float | None:
                if optimum is None:
                    return None
                for e, c in zip(elapsed, costs):
                    if gap_pct(c, optimum) <= threshold_pct:
                        return e
                return None

            peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            summary_rows.append({
                "side": side,
                "n": n,
                "instance_seed": instance_seed,
                "seed_time_s": seed_time,
                "rounds": int(result["rounds_completed"]),
                "elapsed_s": float(result["elapsed_s"]),
                "converged": bool(result["converged"]),
                "per_round_s": float(result["elapsed_s"]) / int(result["rounds_completed"]),
                "final_cost": final,
                "final_gap_pct": gap_pct(final, optimum) if optimum else None,
                "time_to_1pct_s": time_to_gap(1.0),
                "time_to_0p1pct_s": time_to_gap(0.1),
                "time_within_0p1pct_of_final_s": time_within_of_final(0.001),
                "oracle_time_s": oracle_time,
                "dense_matrix_gib": n * n * 8 / 1024**3,
                "peak_rss_gib": peak_rss_gib,
            })
            print(summary_rows[-1], flush=True)

            for i, (e, c) in enumerate(zip(elapsed, costs)):
                trace_rows.append({
                    "side": side,
                    "instance_seed": instance_seed,
                    "round": i + 1,
                    "stride": int(result["round_strides"][i]),
                    "elapsed_s": e,
                    "cost": c,
                })

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with trace_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)
    print(f"wrote {summary_path} and {trace_path}")


if __name__ == "__main__":
    main()
