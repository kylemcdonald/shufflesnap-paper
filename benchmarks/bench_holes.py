"""Holes: quality of rectangular (free-hole) assignment as fill ratio varies.

Compares two strategies against the exact rectangular optimum at 96x96:
  free-holes     default seed (random cell subset, sorted point order) +
                 rectangular multiscale cleanup: the solver decides where
                 the empty cells end up
  predetermined  the empty cells are fixed in advance to the trailing cells
                 (as a partial-last-rows layout does), via a cell mask

Writes results/holes.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import megalap
from bench_common import (
    assignment_cost,
    build_target_grid,
    exact_rect_cost,
    gap_pct,
    make_points,
    results_dir,
    seed_default,
    seed_raster,
)

SIDE = 96
FILL_RATIOS = [0.5, 0.7, 0.85, 0.95, 1.0]
SEEDS = [0, 1, 2]


def main() -> None:
    rows: list[dict] = []
    m = SIDE * SIDE
    targets = build_target_grid(SIDE, SIDE)

    for fill in FILL_RATIOS:
        n = int(round(fill * m))
        for instance_seed in SEEDS:
            points = make_points("meander", n, seed=instance_seed)
            key = f"rect-meander:{SIDE}x{SIDE}:n{n}:seed{instance_seed}"
            optimum = exact_rect_cost(points, SIDE, SIDE, key=key)

            default_seed = seed_default(points, m)
            seed_cost = assignment_cost(points, targets, default_seed)

            free = megalap.window_cleanup(points, default_seed, rows=SIDE, cols=SIDE)
            free_cost = float(free["final_cost"])

            # predetermined holes: only the first n cells are usable, so the
            # empty cells sit at the end of the grid no matter what
            mask = np.zeros(m, dtype=bool)
            mask[:n] = True
            packed = seed_raster(points, total_cells=n)
            fixed = megalap.window_cleanup(
                points, packed, rows=SIDE, cols=SIDE, cell_mask=mask,
            )
            fixed_cost = float(fixed["final_cost"])

            rows.append({
                "fill_ratio": fill,
                "n": n,
                "cells": m,
                "instance_seed": instance_seed,
                "optimum": optimum,
                "seed_gap_pct": gap_pct(seed_cost, optimum),
                "free_gap_pct": gap_pct(free_cost, optimum),
                "pinned_gap_pct": gap_pct(fixed_cost, optimum),
                "free_elapsed_s": float(free["elapsed_s"]),
                "free_converged": bool(free["converged"]),
            })
            print(rows[-1], flush=True)

    out_path = results_dir() / "holes.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
