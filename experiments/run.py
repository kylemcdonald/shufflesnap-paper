"""Run a named experiment through the subprocess harness.

    python experiments/run.py main --threads 1 --cpus 8
    python experiments/run.py main --threads 24 --only-parallel

Results are appended (resumably) to results/raw/<name>_t<threads>.jsonl.
Single-thread runs are pinned to one performance core (CPU 8 on the benchmark host);
multi-thread runs use all logical CPUs without pinning.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench  # noqa: E402
import specs  # noqa: E402

EXPERIMENTS = {
    "main": specs.main_quality,
    "real": specs.real_embeddings,
    "masks": specs.masks,
    "primes": specs.primes,
    "scaling": specs.scaling,
    "anytime": specs.anytime,
    "ablations": specs.ablations,
    "memory": specs.memory,
}

PARALLEL_METHODS = {"shufflesnap", "bspot"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(EXPERIMENTS) + ["threads"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cpus", default=None, help="taskset CPU list, e.g. 8")
    ap.add_argument("--only-parallel", action="store_true", help="only methods that use threads")
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "..", "results", "raw")
    os.makedirs(out_dir, exist_ok=True)
    if args.name == "threads":
        out = os.path.join(out_dir, "threads.jsonl")
        for T, sp in specs.threads_scaling():
            bench.run_specs([sp], out, threads=T, cpus=None, timeout=args.timeout)
        return
    sp = EXPERIMENTS[args.name](threads=args.threads)
    if args.only_parallel:
        sp = [s for s in sp if s["method"] in PARALLEL_METHODS]
    if args.max_n:
        sp = [s for s in sp if s["n"] <= args.max_n]
    out = os.path.join(out_dir, f"{args.name}_t{args.threads}.jsonl")
    print(len(sp), "runs ->", out, flush=True)
    bench.run_specs(sp, out, threads=args.threads, cpus=args.cpus, timeout=args.timeout)


if __name__ == "__main__":
    main()
