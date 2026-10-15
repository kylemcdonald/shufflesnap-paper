"""The 1e7-iteration swap-optimizer runs on seeds 2 and 3 (seed 1: run.py swap_long), spread
over five idle physical performance cores, one run per core at a time (2026-10-08)."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import specs  # noqa: E402

CPUS = ["0", "2", "4", "6", "10"]  # physical cores 0-3 and 5 (8 and 12 hold the other runs)
todo = [sp for s in (2, 3) for sp in specs.swap_optimizer_long(seed=s)]


def work(k):
    out = os.path.join(bench.RAW_DIR, f"swap_long_seeds_cpu{CPUS[k]}_t1.jsonl")
    bench.run_specs(todo[k::len(CPUS)], out, threads=1, cpus=CPUS[k])


with ThreadPoolExecutor(len(CPUS)) as ex:
    list(ex.map(work, range(len(CPUS))))
