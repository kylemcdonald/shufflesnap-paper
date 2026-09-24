"""Compute certified references for every benchmark instance (parallel workers).

Run this BEFORE the timing benchmarks; it saturates the machine.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def work(inst):
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import reference

    d, n, s, g = inst
    t = time.time()
    try:
        r = reference.get(d, n, s, g)
        return dict(inst=inst, ok=True, certified=r["certified"], gap=r["gap"], cost=r["cost"],
                    rounds=r["rounds"], time_exact_s=r["time_exact_s"], wall=time.time() - t)
    except Exception as e:
        return dict(inst=inst, ok=False, error=repr(e), wall=time.time() - t)


if __name__ == "__main__":
    import specs

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    todo = specs.all_instances()
    print(len(todo), "instances", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for f in as_completed(futs):
            print(json.dumps(f.result()), flush=True)
