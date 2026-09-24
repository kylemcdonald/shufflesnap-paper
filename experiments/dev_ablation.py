"""Development-set configuration search (instance seed 0 only; 5 start seeds each).

Used only to choose the improved default configuration.  Final claims use held-out
instance seeds (1-5) in the benchmark scripts.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import shufflesnap as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
import methods  # noqa: E402
import reference  # noqa: E402

R2 = 2 ** 0.5
CONFIGS = {
    "w6-r2": dict(window=6, ratio=2.0),
    "w6-r1.41": dict(window=6, ratio=R2),
    "w6-r2-R2": dict(window=6, ratio=2.0, rounds=2),
    "w8-r2": dict(window=8, ratio=2.0),
    "w8-r1.68": dict(window=8, ratio=2 ** 0.75),
    "w8-r1.41": dict(window=8, ratio=R2),
    "w8-r1.19": dict(window=8, ratio=2 ** 0.25),
    "w8-r1.41-R2": dict(window=8, ratio=R2, rounds=2),
    "w10-r1.41": dict(window=10, ratio=R2),
    "w12-r1.41": dict(window=12, ratio=R2),
}

INSTANCES = [(d, 10000, "square") for d in ["uniform", "gaussian", "gmm", "spiral", "jitter", "unequal2", "mnist"]]
INSTANCES += [(d, 10000, g) for d in ["uniform", "gmm", "spiral"] for g in ["free10", "logo", "disk"]]
INSTANCES += [(d, 100000, "square") for d in ["uniform", "gaussian", "gmm", "spiral", "jitter", "unequal2"]]
INSTANCES += [("mnist", 70000, "square")]


def main(out, seeds=range(5)):
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["dist"], r["n"], r["grid"], r["config"], r["start_seed"]))
    f = open(out, "a")
    for dist, n, grid in INSTANCES:
        ref = reference.get(dist, n, 0, grid, compute=False)
        if ref is None:
            print("no reference yet:", dist, n, grid, flush=True)
            continue
        Pn, g, _, _ = datasets.instance(dist, n, 0, grid)
        for name, cfg in CONFIGS.items():
            for s in seeds:
                if (dist, n, grid, name, s) in done:
                    continue
                t = time.perf_counter()
                cell, info = methods.m_shufflesnap(Pn, g, seed=s, **cfg)
                dt = time.perf_counter() - t
                c = ss.assignment_cost(Pn, g, cell)
                rec = dict(dist=dist, n=n, grid=grid, config=name, start_seed=s, cost=c, opt=ref["cost"],
                           excess_pct=100 * (c / ref["cost"] - 1), excess_per_pt=(c - ref["cost"]) / n, time_s=dt,
                           windows_solved=info["windows_solved"])
                f.write(json.dumps(rec) + "\n")
                f.flush()
        print("done", dist, n, grid, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results", "raw",
                                                             "dev_ablation.jsonl"))
