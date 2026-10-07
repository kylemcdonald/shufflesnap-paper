"""The billion-point meander instance, solved with the current kernel under a time budget.

The instance is the one of the earlier ShuffleSnap work (../../shufflesnap-paper/benchmarks/
bench_billion.py): side = 31623 (1,000,014,129 points), a random walk
normalize(cumsum(rng.normal(size=(n, 2)))) drawn with numpy's default_rng(seed) and
generated chunk by chunk.  The start is the package default, default_rng(0).permutation(n),
which is also the start of the earlier run, so the two runs differ only in the solver.

``assign()`` would hold several point-sized temporaries (the Grid's lattice and the
nonzero() indices, a normalized copy of the points, the duplicate-cell check).  This
driver builds the same inputs without them and calls the kernel directly with the
balanced preset (8x8 windows, ratio sqrt 2, geometric window solver).  The mapping is
the "bbox" normalization: the points span [0, 1] on both axes, the cell centers
[0, side - 1], so P is scaled by side - 1 in place.  The budget counts from that
scaling, as ``assign(time_budget=...)`` does; generating the instance is not counted.

Peak memory is about 58 GiB: points and cell centers (16 GB each), grid index, positions
and occupancy (4 GB each), change stamps (8 GB) and the window lists.

    python experiments/billion.py --budget 3600
    python experiments/billion.py --side 300 --check   # compare with assign() at small size

Writes results/billion/billion_{side}_seed{seed}.json, the per-phase trace as CSV, and
the assignment (int32 cell id per point, row-major cells) as .npy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import threading
import time
from pathlib import Path

import numpy as np

import shufflesnap as ss
from shufflesnap import _core
from shufflesnap.api import PRESETS, SOLVERS, TRACE_COLUMNS, build_schedule, default_offsets

# driver-only preset: 16x16 windows, to test whether larger windows still improve the final assignment
PRESETS = {**PRESETS, "w16": dict(window=16, schedule="geometric", ratio=2 ** 0.5)}

CHUNK = 20_000_000
OUT = Path(__file__).resolve().parent.parent / "results" / "billion"


def generate_meander(n: int, seed: int = 0, chunk: int = CHUNK) -> np.ndarray:
    """Chunked normalize(cumsum(rng.normal(size=(n, 2)))), bit-identical to the earlier
    bench_billion.generate_meander (same RNG stream, same add order)."""
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 2), dtype=np.float64)
    last = np.zeros(2, dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        block = rng.normal(size=(i1 - i0, 2))
        block[0] += last
        np.cumsum(block, axis=0, out=block)
        pts[i0:i1] = block
        last = pts[i1 - 1].copy()
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.maximum(maxs - mins, np.finfo(np.float64).eps)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        block = pts[i0:i1]
        block -= mins
        block /= span
    return pts


def rss_gib() -> tuple[float, float]:
    rss = hwm = 0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS"):
            rss = int(line.split()[1])
        elif line.startswith("VmHWM"):
            hwm = int(line.split()[1])
    return rss / 2**20, hwm / 2**20


def cell_centers(side: int, path: str | None = None, chunk: int = CHUNK) -> np.ndarray:
    """Row-major cell centers (x = column, y = row).  With ``path`` they live in a file-backed
    memmap: they are read-only during the solve, so under memory pressure the kernel can drop
    their pages (and reread them) instead of swapping out working memory."""
    shape = (side * side, 2)
    C = np.empty(shape, dtype=np.float64) if path is None else np.memmap(path, np.float64, "w+", shape=shape)
    for i0 in range(0, side * side, chunk):
        ids = np.arange(i0, min(i0 + chunk, side * side), dtype=np.int64)
        C[i0:i0 + len(ids), 0] = ids % side
        C[i0:i0 + len(ids), 1] = ids // side
    return C


def solve(P: np.ndarray, side: int, budget: float | None, threads: int, preset: str, heartbeat: float,
          centers_file: str | None = None, resume: str | None = None, resume_stride: int | None = None,
          one_round: bool = False):
    """Scale P in place to the grid frame and run the preset's schedule on it.  As in assign(),
    the grid is built before the budget starts.

    With ``resume`` (a saved assignment), start from it instead of the random permutation and
    run the schedule from the first level with stride ``resume_stride`` on.  Restarting at a
    level boundary loses only the skip bookkeeping of an interrupted level, which is then
    re-run from its first tiling."""
    n = side * side
    cfg = PRESETS[preset]
    t_grid0 = time.perf_counter()
    index = np.arange(n, dtype=np.int32).reshape(side, side)
    C = cell_centers(side, centers_file)
    grid_s = time.perf_counter() - t_grid0
    t_start = time.perf_counter()
    P *= side - 1  # "bbox": [0, 1]^2 onto the cell-center box [0, side - 1]^2
    t_norm = time.perf_counter()
    if resume is None:
        pos = np.random.default_rng(0).permutation(n).astype(np.int32)  # initial_assignment("random", seed=0)
    else:
        pos = np.ascontiguousarray(np.load(resume), dtype=np.int32)
        assert pos.shape == (n,)
    t_init = time.perf_counter()
    occ = np.empty(n, dtype=np.int32)
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        occ[pos[i0:i1]] = np.arange(i0, i1, dtype=np.int32)
    t_grid = time.perf_counter()
    stages = build_schedule(side, side, cfg["window"], cfg["schedule"], ratio=cfg["ratio"])
    if resume_stride is not None:
        first = int(np.nonzero(stages[:, 0] == resume_stride)[0][0])
        stages = stages[first:]
    if one_round:  # only the first level, for a single round (one phase per offset)
        stages = stages[:1].copy()
        stages[0, 2:] = (1, 0)
    offsets = default_offsets(cfg["window"])
    remaining = None if budget is None else max(0.0, budget - (t_grid - t_start))
    print(f"setup {t_grid - t_start:.1f}s (scale {t_norm - t_start:.1f}, start {t_init - t_norm:.1f}, "
          f"occupancy {t_grid - t_init:.1f}; grid built before, {grid_s:.1f}); rss {rss_gib()[0]:.1f} GiB; stages {stages[:, :2].tolist()}", flush=True)

    stop = threading.Event()

    def beat():
        while not stop.wait(heartbeat):
            r, h = rss_gib()
            swap = [l.split()[1] for l in Path("/proc/self/status").read_text().splitlines() if l.startswith("VmSwap")]
            print(f"[hb] t={time.perf_counter() - t_start:7.0f}s rss={r:5.1f}GiB peak={h:5.1f}GiB "
                  f"swap={int(swap[0]) / 2**20 if swap else 0:.2f}GiB", flush=True)

    hb = threading.Thread(target=beat, daemon=True)
    hb.start()
    flat, finished = _core.run_schedule(P, index, C, occ, pos, stages, offsets, cfg["window"],
                                        -1.0 if remaining is None else remaining, threads, 1e-12, True,
                                        SOLVERS["geometric"])
    stop.set()
    t_end = time.perf_counter()
    trace = np.asarray(flat, dtype=np.float64).reshape(-1, len(TRACE_COLUMNS))
    trace[:, 0] += t_grid - t_start  # seconds since the budget started, as in assign()
    cost = float(_core.assignment_cost(P, C, pos, threads))
    times = dict(scale_s=t_norm - t_start, init_s=t_init - t_norm, occupancy_s=t_grid - t_init, grid_build_s=grid_s,
                 solve_s=t_end - t_grid, total_s=t_end - t_start)
    return pos, cost, finished, trace, times, C, stages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=31623)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=3600.0, help="seconds; <0 means none")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--heartbeat", type=float, default=60.0)
    ap.add_argument("--check", action="store_true", help="compare with shufflesnap.assign (small sides)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--centers-file", help="keep the cell centers in a file-backed memmap at this path")
    ap.add_argument("--resume", help="continue from this saved assignment (.npy) instead of the random start")
    ap.add_argument("--resume-stride", type=int, help="with --resume: first schedule level to run")
    ap.add_argument("--time-offset", type=float, default=0.0,
                    help="with --resume: seconds already spent, added to the trace's elapsed times")
    ap.add_argument("--one-round", action="store_true", help="run a single round of the first level only")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="earlier assignments (.npy, int cell ids, row-major) to price on the same instance")
    args = ap.parse_args()
    side, n = args.side, args.side * args.side
    budget = None if args.budget < 0 else args.budget

    t0 = time.perf_counter()
    P = generate_meander(n, args.seed)
    gen_s = time.perf_counter() - t0
    print(f"side={side} n={n:,} generated in {gen_s:.1f}s rss={rss_gib()[0]:.1f}GiB", flush=True)
    raw = P.copy() if args.check else None

    pos, cost, finished, trace, times, C, stages = solve(P, side, budget, args.threads, args.preset, args.heartbeat,
                                                           args.centers_file, args.resume, args.resume_stride,
                                                           args.one_round)
    trace[:, 0] += args.time_offset
    print(f"cost {cost:.6f} finished={finished} {times}", flush=True)

    if args.check:
        res = ss.assign(raw, ss.Grid.rectangle(side, side), preset=args.preset, time_budget=budget,
                        threads=args.threads)
        print(f"assign(): cost {res.cost:.6f} finished={res.finished}; identical cells: "
              f"{np.array_equal(res.cell, pos)}", flush=True)

    compare = {}
    for path in args.compare:
        other = np.load(path, mmap_mode="r")
        assert other.shape == (n,)
        other = np.ascontiguousarray(other, dtype=np.int32)
        compare[Path(path).name] = float(_core.assignment_cost(P, C, other, args.threads))
        del other
        print(f"{path}: cost {compare[Path(path).name]:.6f}", flush=True)

    elapsed, costs = trace[:, 0], trace[:, 1]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    def first_within(ratio):
        hit = np.nonzero(costs <= cost * (1 + ratio))[0]
        return float(elapsed[hit[0]]) if len(hit) else None

    fine = trace[(trace[:, 5] == 1) & (trace[:, 6] == 1)]
    summary = dict(
        side=side, n=n, instance_seed=args.seed, init="random permutation, default_rng(0)",
        preset=args.preset, window=PRESETS[args.preset]["window"], ratio=PRESETS[args.preset]["ratio"],
        solver="geometric", threads=args.threads or _core.max_threads(), budget_s=budget,
        shufflesnap_version=ss.__version__ if hasattr(ss, "__version__") else None,
        gen_s=gen_s, **times, finished=finished, final_cost=cost, mean_sq_displacement=cost / n,
        start_cost=float(costs[0]), stages=stages.tolist(),
        phases=int(len(trace) - 1), stride1_rounds=int(len(fine) // 4),
        time_to_stride1_s=float(fine[0, 0]) if len(fine) else None,
        time_within_1pct_s=first_within(1e-2), time_within_0p1pct_s=first_within(1e-3),
        time_within_0p01pct_s=first_within(1e-4),
        peak_rss_gib=peak, compare_costs=compare,
        resume=args.resume, resume_stride=args.resume_stride, time_offset_s=args.time_offset,
    )
    if args.no_save:
        print(json.dumps(summary, indent=1))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{side}_seed{args.seed}{args.tag}"
    (OUT / f"billion_{tag}.json").write_text(json.dumps(summary, indent=1) + "\n")
    with open(OUT / f"billion_trace_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(TRACE_COLUMNS)
        for row in trace:
            w.writerow([f"{row[0]:.3f}", repr(float(row[1]))] + [int(v) for v in row[2:]])
    np.save(OUT / f"billion_assignment_{tag}.npy", pos)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
