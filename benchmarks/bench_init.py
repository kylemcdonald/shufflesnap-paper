"""Initialization ablation: raster-sorted spread seed vs random permutation.

Motivated by the convergence experiment, where the random start landed a few
times lower than the raster start on all three meander instances. Three
sweeps, every run to strict convergence (budget_seconds=None):

  families   all six point families x {96x96, 200x200} x 3 instance seeds;
             raster seed vs 5 random permutations; gaps vs cached exact refs
  scaling    meander at sides {100, 141, 200, 283, 400, 566, 800};
             raster vs 3 random permutations; gaps where an exact reference
             is cached, raw final costs everywhere (paired comparison)
Writes results/init_families.csv and init_scaling.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shufflesnap
from bench_common import (
    DISTRIBUTIONS,
    assignment_cost,
    build_target_grid,
    exact_instance,
    gap_pct,
    make_points,
    results_dir,
    seed_random_cells,
    seed_raster,
)

FAMILY_SIDES = [96, 200]
SCALING_SIDES = [100, 141, 200, 283, 400, 566, 800]
SEEDS = [0, 1, 2]
FAMILY_REPS = 5
SCALING_REPS = 3


def run_to_convergence(points, seed_assignment, rows, cols, targets, optimum):
    """One converged cleanup; gap fields are blank when optimum is None."""
    seed_cost = assignment_cost(points, targets, seed_assignment)
    result = shufflesnap.window_cleanup(
        points, seed_assignment, rows=rows, cols=cols,
        budget_seconds=None, trace_rounds=True,
    )
    final_cost = float(result["final_cost"])
    stats = {
        "seed_cost": seed_cost,
        "final_cost": final_cost,
        "rounds": int(result["rounds_completed"]),
        "elapsed_s": float(result["elapsed_s"]),
        "converged": bool(result["converged"]),
        "seed_gap_pct": "",
        "final_gap_pct": "",
        "time_to_0p1pct_s": "",
    }
    if optimum is not None:
        stats["seed_gap_pct"] = gap_pct(seed_cost, optimum)
        stats["final_gap_pct"] = gap_pct(final_cost, optimum)
        for elapsed, cost in zip(result["round_elapsed_s"], result["round_costs"]):
            if gap_pct(float(cost), optimum) <= 0.1:
                stats["time_to_0p1pct_s"] = float(elapsed)
                break
    return stats


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}", flush=True)


def family_sweep() -> None:
    rows: list[dict] = []
    for side in FAMILY_SIDES:
        n = side * side
        targets = build_target_grid(side, side)
        for family in DISTRIBUTIONS:
            for instance_seed in SEEDS:
                points = make_points(family, n, seed=instance_seed)
                optimum = exact_instance(family, side, instance_seed)["cost"]
                runs = [("raster", "", seed_raster(points))]
                for rep in range(FAMILY_REPS):
                    rng_seed = 1000 * instance_seed + rep
                    runs.append(("random", rep, seed_random_cells(n, n, rng_seed)))
                for init, rep, seed_assignment in runs:
                    stats = run_to_convergence(
                        points, seed_assignment, side, side, targets, optimum)
                    rows.append({
                        "family": family, "side": side,
                        "instance_seed": instance_seed,
                        "init": init, "rep": rep, **stats,
                    })
                    print(
                        f"families {family:9s} side={side} seed={instance_seed} "
                        f"{init}{rep} final={stats['final_gap_pct']:.4f}% "
                        f"rounds={stats['rounds']}", flush=True)
    write_csv(results_dir() / "init_families.csv", rows)


def scaling_sweep() -> None:
    cached_sides = {100, 141, 200}
    rows: list[dict] = []
    for side in SCALING_SIDES:
        n = side * side
        targets = build_target_grid(side, side)
        for instance_seed in SEEDS:
            points = make_points("meander", n, seed=instance_seed)
            optimum = (exact_instance("meander", side, instance_seed)["cost"]
                       if side in cached_sides else None)
            runs = [("raster", "", seed_raster(points))]
            for rep in range(SCALING_REPS):
                rng_seed = 1000 * instance_seed + rep
                runs.append(("random", rep, seed_random_cells(n, n, rng_seed)))
            for init, rep, seed_assignment in runs:
                stats = run_to_convergence(
                    points, seed_assignment, side, side, targets, optimum)
                rows.append({
                    "side": side, "n": n, "instance_seed": instance_seed,
                    "init": init, "rep": rep, **stats,
                })
                print(
                    f"scaling side={side} seed={instance_seed} {init}{rep} "
                    f"final_cost={stats['final_cost']:.6f} "
                    f"rounds={stats['rounds']} elapsed={stats['elapsed_s']:.1f}s",
                    flush=True)
    write_csv(results_dir() / "init_scaling.csv", rows)


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"families", "scaling"}
    if "families" in which:
        family_sweep()
    if "scaling" in which:
        scaling_sweep()
