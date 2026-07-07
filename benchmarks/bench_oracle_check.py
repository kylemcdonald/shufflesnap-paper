"""Validate the reference solvers against each other on small instances.

Checks that the on-demand auction oracle, the native dense JV solver, and
SciPy's linear_sum_assignment agree. Writes results/oracle_check.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import megalap
import numpy as np
from scipy.optimize import linear_sum_assignment as scipy_lsa

from bench_common import (
    build_target_grid,
    exact_square,
    make_points,
    results_dir,
)


def main() -> None:
    rows: list[dict] = []
    for side in [20, 32, 48, 64]:
        for seed in [0, 1, 2]:
            n = side * side
            points = make_points("meander", n, seed=seed)
            targets = build_target_grid(side, side)
            cost = ((points[:, None, :] - targets[None, :, :]) ** 2).sum(-1)

            ri, ci = scipy_lsa(cost)
            scipy_cost = float(cost[ri, ci].sum())
            _, _, jv_cost = megalap.linear_sum_assignment(cost)
            auction_cost = exact_square(points, side, side)["cost"]

            rows.append({
                "side": side,
                "seed": seed,
                "scipy": scipy_cost,
                "megalap_jv": jv_cost,
                "auction_oracle": auction_cost,
                "jv_vs_scipy_rel": abs(jv_cost - scipy_cost) / scipy_cost,
                "auction_vs_scipy_rel": abs(auction_cost - scipy_cost) / scipy_cost,
            })
            print(rows[-1], flush=True)

    out_path = results_dir() / "oracle_check.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
