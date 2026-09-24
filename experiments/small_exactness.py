"""Small instances: how often does each method reach the exact optimum?

For every instance the optimum comes from the dense SciPy solver *and* from the
certified sparse solver; both must agree (an independent check of the certifier).
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
import methods  # noqa: E402

CONFIGS = {"SS-baseline": dict(window=6, ratio=2.0), "SS-fast": dict(window=6, ratio=2 ** 0.5),
           "SS-balanced": dict(window=8, ratio=2 ** 0.5), "SS-quality": dict(window=12, ratio=2 ** 0.5)}


def main(out):
    f = open(out, "w")
    for n in (100, 400, 900, 2500):
        for dist in ("uniform", "gaussian", "gmm", "spiral", "jitter", "duplicates"):
            for grid in ("square", "free", "disk"):
                for seed in range(100, 110):
                    Pn, g, _, _ = datasets.instance(dist, n, seed, grid)
                    t = time.perf_counter()
                    dense = exact.solve_dense(Pn, g)
                    t_dense = time.perf_counter() - t
                    opt = ss.assignment_cost(Pn, g, dense)
                    t = time.perf_counter()
                    cert = exact.solve_certified(Pn, g, ss.assign(Pn, g, normalize="none", seed=seed).cell)
                    t_cert = time.perf_counter() - t
                    agree = bool(np.isclose(cert["cost"], opt, rtol=1e-9, atol=1e-9))
                    rec = dict(n=n, dist=dist, grid=grid, seed=seed, opt=opt, t_dense=t_dense, t_cert=t_cert,
                               cert_agrees=agree, cert_certified=bool(cert["certified"]), results={})
                    for name, cfg in CONFIGS.items():
                        cell, _ = methods.m_shufflesnap(Pn, g, seed=seed, **cfg)
                        c = ss.assignment_cost(Pn, g, cell)
                        rec["results"][name] = c
                    for k in (16, 64):
                        try:
                            cell, _ = methods.m_bspot(Pn, g, num_plans=k,
                                                      variant="gaussian" if g.n_cells == n else "default")
                            rec["results"][f"BSP-OT-{k}"] = ss.assignment_cost(Pn, g, cell)
                        except Exception as e:  # noqa: BLE001
                            rec["results"][f"BSP-OT-{k}"] = None
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            print("done", n, dist, flush=True)


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw", "small_exactness.jsonl"))
