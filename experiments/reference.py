"""Certified optimal assignments, cached per instance.

The certificate is the Lagrangian lower bound computed over *all* cells (never the
candidate subset), see ``shufflesnap.exact``.  ``gap = cost - lower_bound`` is stored;
an instance is "certified" when gap <= max(1e-9 * cost, N * 4 / S).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import shufflesnap as ss
from shufflesnap import exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "reference")


def get(dist, n, seed, grid="square", normalize="bbox", compute=True, verbose=False, time_limit=None):
    os.makedirs(REF, exist_ok=True)
    key = datasets.instance_key(dist, n, seed, grid, normalize)
    path = os.path.join(REF, key + ".npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return dict(cell=z["cell"], **meta)
    if not compute:
        return None
    Pn, g, _, _ = datasets.instance(dist, n, seed, grid, normalize)
    t0 = time.perf_counter()
    warm = ss.assign(Pn, g, normalize="none", preset="balanced", seed=12345)
    t_warm = time.perf_counter() - t0
    ex = exact.solve_certified(Pn, g, warm.cell, verbose=verbose, time_limit=time_limit)
    ss.validate_assignment(ex["cell"], n, g)
    cost = ss.assignment_cost(Pn, g, ex["cell"])
    meta = dict(key=key, cost=float(cost), lower_bound=ex["lower_bound"], gap=ex["gap"],
                certified=bool(ex["certified"]), tolerance=ex["tolerance"], rounds=len(ex["rounds"]),
                edges=ex["rounds"][-1]["edges"], time_exact_s=ex["time_s"], time_warm_s=t_warm,
                n_cells=g.n_cells, width=g.width, height=g.height)
    np.savez_compressed(path, cell=ex["cell"].astype(np.int32), meta=json.dumps(meta))
    return dict(cell=ex["cell"], **meta)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", nargs="+", required=True)
    ap.add_argument("--n", nargs="+", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--grid", nargs="+", default=["square"])
    args = ap.parse_args()
    for d in args.dist:
        for n in args.n:
            for g in args.grid:
                for s in args.seeds:
                    t = time.time()
                    r = get(d, n, s, g, verbose=False)
                    print(json.dumps({k: v for k, v in r.items() if k != "cell"}), f"{time.time() - t:.1f}s",
                          flush=True)
