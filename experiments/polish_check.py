"""Held-out check of two post-convergence polishing ideas (quality only, instance seeds 1-3).

(a) V-cycles: after the baseline converges, run [stride 2 (1 round), stride 1 (to convergence)]
    three more times.
(b) Merging: combine 4 independent baseline runs (start seeds 1-4) per connected component of
    their union, keeping the cheaper side (the BSP-OT merge rule).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import shufflesnap as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
import methods  # noqa: E402
import reference  # noqa: E402


def merge(P, g, a, b):
    N, M = len(a), g.n_cells
    occ_b = np.full(M, -1)
    occ_b[b] = np.arange(N)
    j = occ_b[a]
    m = j >= 0
    G = coo_matrix((np.ones(m.sum()), (np.nonzero(m)[0], j[m])), shape=(N, N))
    ncomp, lab = connected_components(G, directed=False)
    ca = ((P - g.centers[a]) ** 2).sum(1)
    cb = ((P - g.centers[b]) ** 2).sum(1)
    useb = np.bincount(lab, cb, ncomp) < np.bincount(lab, ca, ncomp)
    return np.where(useb[lab], b, a).astype(np.int32)


def main(out):
    f = open(out, "w")
    for dist, n in (("uniform", 100000), ("gmm", 100000), ("mnist", 70000)):
        for seed in (1, 2, 3):
            Pn, g, _, _ = datasets.instance(dist, n, seed)
            opt = reference.get(dist, n, seed, compute=False)["cost"]
            base, _ = methods.m_shufflesnap(Pn, g, window=6, ratio=2.0, seed=seed)
            c0 = ss.assignment_cost(Pn, g, base)
            pos = base.copy()
            t = time.perf_counter()
            vc = np.array([[2, 2, 1, 0], [1, 1, -1, 1]], dtype=np.int32)
            for _ in range(3):
                ss.run_schedule(Pn, g, pos, vc, window=6)
            t_vc = time.perf_counter() - t
            c_vc = ss.assignment_cost(Pn, g, pos)
            runs = [base] + [methods.m_shufflesnap(Pn, g, window=6, ratio=2.0, seed=seed + 100 * k)[0] for k in (1, 2, 3)]
            cur = runs[0]
            for r in runs[1:]:
                cur = merge(Pn, g, cur, r)
            ss.validate_assignment(cur, n, g)
            c_m = ss.assignment_cost(Pn, g, cur)
            rec = dict(dist=dist, n=n, seed=seed, base_excess=100 * (c0 / opt - 1),
                       vcycle_excess=100 * (c_vc / opt - 1), vcycle_changed_cost=bool(c_vc < c0), vcycle_s=t_vc,
                       merge4_excess=100 * (c_m / opt - 1))
            print(json.dumps(rec), flush=True)
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw", "polish_check.jsonl"))
