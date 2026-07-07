"""Precompute and cache all exact reference costs used by the benchmarks."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_common import DISTRIBUTIONS, exact_instance

SEEDS = [0, 1, 2]


def main() -> None:
    jobs: list[tuple[str, int, int]] = []
    for seed in SEEDS:
        for side in [71, 100, 141, 200]:
            jobs.append(("meander", side, seed))
    for dist in DISTRIBUTIONS:
        for seed in SEEDS:
            jobs.append((dist, 96, seed))
            jobs.append((dist, 200, seed))

    for dist, side, seed in jobs:
        t0 = time.time()
        entry = exact_instance(dist, side, seed)
        print(
            f"{dist} {side}x{side} seed={seed}: cost={entry['cost']:.6f} "
            f"solve={entry.get('solve_time_s')} wall={time.time()-t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
