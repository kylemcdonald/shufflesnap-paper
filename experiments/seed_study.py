"""Spread of ShuffleSnap's excess over the optimum across point clouds and random starts.

    python experiments/seed_study.py [--n 10000] [--clouds 50] [--inits 50] [--workers 20]

Uniform points, square grid, bbox normalization (as in the benchmark).  Cloud seeds start at
100, clear of the development seed 0 and the held-out benchmark seeds 1-3.  For every cloud
the dense SciPy solver gives the optimum once.  For every (cloud, init) pair the default
preset runs from the random start ``seed=init`` on one thread, and the certified solver,
warm-started from that ShuffleSnap layout, solves the problem again; its cost must equal the
dense optimum whatever the start.  Writes results/seed_study/n{N}/{clouds,runs}.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import shufflesnap as ss
from shufflesnap import exact
from shufflesnap.metrics import assignment_cost

import datasets
import methods

ROOT = Path(__file__).resolve().parents[1]
CLOUD_SEED0 = 100


def solve_cloud(args):
    cloud, n, inits = args
    seed = CLOUD_SEED0 + cloud
    Pn, g, _, _ = datasets.instance("uniform", n, seed)
    t = time.perf_counter()
    opt_cell, _ = methods.m_dense(Pn, g)
    t_dense = time.perf_counter() - t
    opt = assignment_cost(Pn, g, opt_cell)
    runs = []
    for init in range(inits):
        t = time.perf_counter()
        cell = ss.assign(Pn, g, normalize="none", preset="balanced", init="random", seed=init, threads=1).cell
        t_ss = time.perf_counter() - t
        cost = assignment_cost(Pn, g, cell)
        t = time.perf_counter()
        r = exact.solve_certified(Pn, g, cell, threads=1)
        t_cert = time.perf_counter() - t
        cert = assignment_cost(Pn, g, r["cell"])
        runs.append(dict(cloud_seed=seed, init_seed=init, ss_cost=cost, opt_cost=opt,
                         excess_pct=100 * (cost / opt - 1), ss_time=t_ss,
                         on_optimal_cell=float(np.mean(cell == opt_cell)),
                         certified_cost=cert, certified=bool(r["certified"]),
                         certified_matches=bool(abs(cert - opt) <= 1e-9 * opt), certified_time=t_cert))
    return dict(cloud_seed=seed, opt_cost=opt, opt_per_point=opt / n, dense_time=t_dense), runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--clouds", type=int, default=50)
    ap.add_argument("--inits", type=int, default=50)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    out = ROOT / "results" / "seed_study" / f"n{a.n}"
    out.mkdir(parents=True, exist_ok=True)
    clouds, runs = [], []
    with ProcessPoolExecutor(a.workers) as pool:
        for c, r in pool.map(solve_cloud, [(i, a.n, a.inits) for i in range(a.clouds)]):
            clouds.append(c)
            runs += r
            ex = [x["excess_pct"] for x in r]
            print(f"cloud {c['cloud_seed']}: opt {c['opt_per_point']:.4f}/pt, excess mean {np.mean(ex):.3f}% "
                  f"max {np.max(ex):.3f}%, certified ok {all(x['certified_matches'] for x in r)}", flush=True)
    for name, rows in (("clouds", clouds), ("runs", runs)):
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out}")
    if os.environ.get("AGENT_FILES_URL"):
        from urllib.parse import quote
        print(f"{os.environ['AGENT_FILES_URL']}/#{quote(str(out))}")


if __name__ == "__main__":
    main()
