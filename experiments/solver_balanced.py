"""Window-solver comparison at the default (balanced) preset, one and 24 threads.

    python experiments/solver_balanced.py --threads 24
    python experiments/solver_balanced.py --threads 1 --cpus 8

Results are appended (resumably) to results/raw/solver_balanced_t<threads>.jsonl (or $SS_RESULTS_RAW).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench  # noqa: E402
import specs as _specs  # noqa: E402

SOLVERS = ("hungarian", "hungarian_greedy", "jv", "auction", "geometric")


def specs(threads):
    out = []
    for d in ("uniform", "gmm", "mnist"):
        n = 70000 if d == "mnist" else 100000
        for s in (1, 2, 3):
            for sv in SOLVERS:
                p = dict(_specs.DEFAULT, solver=sv, seed=s, threads=threads)
                out.append(dict(method="shufflesnap", label="abl-solver-balanced", dist=d, n=n, seed=s,
                                grid="square", params=p))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--cpus", default=None)
    args = ap.parse_args()
    out = os.path.join(bench.RAW_DIR, f"solver_balanced_t{args.threads}.jsonl")
    bench.run_specs(specs(args.threads), out, threads=args.threads, cpus=args.cpus)
