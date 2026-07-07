"""Headline experiment: convergence of multiscale vs local-only refinement
from different seeds, on 200x200 meandering clouds with exact references.

Methods:
  random+multiscale   random permutation seed, default coarse-to-fine schedule
  raster+multiscale   spread raster seed, default schedule
  raster+local        spread raster seed, stride-1 only (component ablation)
  rasterfairy+local   RasterFairy's construction (Klingemann 2016),
                      then stride-1 only

Writes results/convergence_traces.csv with per-round traces.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import megalap
from bench_common import (
    assignment_cost,
    build_target_grid,
    exact_instance,
    gap_pct,
    make_points,
    results_dir,
    rasterfairy_assignment,
    seed_random,
    seed_raster,
    write_environment,
)

SIDE = 200
SEEDS = [0, 1, 2]
BUDGET_S = 20.0


def main() -> None:
    write_environment()
    out_path = results_dir() / "convergence_traces.csv"
    rows: list[dict] = []

    for instance_seed in SEEDS:
        n = SIDE * SIDE
        points = make_points("meander", n, seed=instance_seed)
        targets = build_target_grid(SIDE, SIDE)
        optimum = exact_instance("meander", SIDE, instance_seed)["cost"]

        methods: dict[str, tuple[np.ndarray, float, object]] = {}
        t0 = time.time()
        raster = seed_raster(points)
        raster_time = time.time() - t0
        t0 = time.time()
        rand = seed_random(n, rng_seed=instance_seed)
        random_time = time.time() - t0
        bisect, bisect_time = rasterfairy_assignment(points, SIDE, SIDE)

        methods["random+multiscale"] = (rand, random_time, None)
        methods["raster+multiscale"] = (raster, raster_time, None)
        methods["raster+local"] = (raster, raster_time, [1])
        methods["rasterfairy+local"] = (bisect, bisect_time, [1])

        for method, (seed_assignment, seed_time, strides) in methods.items():
            result = megalap.window_cleanup(
                points,
                seed_assignment,
                rows=SIDE,
                cols=SIDE,
                budget_seconds=BUDGET_S,
                strides=strides,
                trace_rounds=True,
            )
            seed_cost = assignment_cost(points, targets, seed_assignment)
            rows.append({
                "instance_seed": instance_seed,
                "method": method,
                "round": 0,
                "stride": "",
                "seed_time_s": seed_time,
                "elapsed_s": 0.0,
                "cost": seed_cost,
                "gap_pct": gap_pct(seed_cost, optimum),
                "converged": "",
            })
            for i, (elapsed, cost, stride) in enumerate(zip(
                result["round_elapsed_s"], result["round_costs"], result["round_strides"]
            )):
                rows.append({
                    "instance_seed": instance_seed,
                    "method": method,
                    "round": i + 1,
                    "stride": int(stride),
                    "seed_time_s": seed_time,
                    "elapsed_s": float(elapsed),
                    "cost": float(cost),
                    "gap_pct": gap_pct(float(cost), optimum),
                    "converged": bool(result["converged"]),
                })
            print(
                f"seed={instance_seed} {method:22s} seed_gap={gap_pct(seed_cost, optimum):9.2f}% "
                f"final={gap_pct(result['final_cost'], optimum):8.4f}% "
                f"rounds={result['rounds_completed']} converged={result['converged']}",
                flush=True,
            )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
