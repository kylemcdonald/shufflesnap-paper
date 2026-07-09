"""Billion-scale meander: convergence timing on side*side points.

Generates a meander (random-walk) point set chunk by chunk with the same RNG
stream as make_points("meander", n, seed), seeds it with a quantized raster
ordering (a memory-lean stand-in for seed_raster's lexsort), then runs
multiscale window cleanup with a wall-clock budget and full round tracing.

Writes results/billion_{side}_seed{seed}.json, billion_traces_{side}_seed{seed}.csv,
and the final assignment as int32 into results/assignments/.

Peak memory at side=31623 (1.00e9 points) is ~48 GiB, so this needs the
memory-lean kernel (megalap >= 0.2.0 with axis-band window enumeration).
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import megalap
from bench_common import MARGIN, results_dir

CHUNK = 20_000_000


def generate_meander(n: int, seed: int = 0, chunk: int = CHUNK) -> np.ndarray:
    """Chunked equivalent of normalize(cumsum(rng.normal(size=(n, 2)))).

    Draws the identical RNG stream as make_points("meander", n, seed) and
    applies the identical normalization arithmetic, but never holds more than
    one chunk of temporaries (verified bit-identical at small n).
    """
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 2), dtype=np.float64)
    last = np.zeros(2, dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        block = rng.normal(size=(i1 - i0, 2))
        block[0] += last  # carry in before cumsum: same add order as one long cumsum
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
        block *= 1.0 - 2.0 * MARGIN
        block += MARGIN
    return pts


def quantized_raster_seed(pts: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """seed_raster with a packed uint64 (y, x) sort key.

    seed_raster lexsorts two float64 keys, which buffers both key columns at
    full length; packing 32-bit quantized coordinates into one uint64 key
    halves that. Ordering matches seed_raster up to 2^-32 quantization ties,
    which is irrelevant here: the multiscale schedule erases seed differences
    within a few rounds.
    """
    n = len(pts)
    key = np.empty(n, dtype=np.uint64)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        xq = (pts[i0:i1, 0] * 4294967296.0).astype(np.uint64)
        yq = (pts[i0:i1, 1] * 4294967296.0).astype(np.uint64)
        key[i0:i1] = (yq << np.uint64(32)) | xq
    order = np.argsort(key, kind="stable")
    del key
    assignment = np.empty(n, dtype=np.int64)
    assignment[order] = np.arange(n, dtype=np.int64)
    return assignment


def read_rss_gib() -> tuple[float, float]:
    rss = hwm = 0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS"):
            rss = int(line.split()[1])
        elif line.startswith("VmHWM"):
            hwm = int(line.split()[1])
    return rss / 2**20, hwm / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, required=True, help="cleanup budget in seconds")
    parser.add_argument("--heartbeat", type=float, default=60.0)
    parser.add_argument("--verify-gen", action="store_true",
                        help="check chunked generation against make_points (small sides only)")
    parser.add_argument("--no-save-assignment", action="store_true")
    args = parser.parse_args()

    side = args.side
    n = side * side
    tag = f"{side}_seed{args.seed}"
    print(f"side={side} n={n:,} budget={args.budget}s", flush=True)

    t0 = time.time()
    pts = generate_meander(n, seed=args.seed)
    gen_time = time.time() - t0
    print(f"generated in {gen_time:.1f}s rss={read_rss_gib()[0]:.1f}GiB", flush=True)

    if args.verify_gen:
        from bench_common import make_points
        ref = make_points("meander", n, seed=args.seed)
        assert np.array_equal(pts, ref), "chunked generation diverged from make_points"
        del ref
        print("generation verified bit-identical to make_points", flush=True)

    t0 = time.time()
    seed_assignment = quantized_raster_seed(pts)
    seed_time = time.time() - t0
    print(f"seeded in {seed_time:.1f}s rss={read_rss_gib()[0]:.1f}GiB", flush=True)

    stop = threading.Event()
    t_start = time.time()

    def heartbeat() -> None:
        while not stop.wait(args.heartbeat):
            rss, hwm = read_rss_gib()
            print(f"[hb] t={time.time() - t_start:7.0f}s rss={rss:5.1f}GiB peak={hwm:5.1f}GiB",
                  flush=True)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    result = megalap.window_cleanup(
        pts, seed_assignment, rows=side, cols=side,
        budget_seconds=args.budget, trace_rounds=True,
    )
    stop.set()
    hb.join()
    del seed_assignment
    print(f"cleanup done rounds={result['rounds_completed']} "
          f"elapsed={result['elapsed_s']:.1f}s converged={result['converged']}", flush=True)

    elapsed = [float(e) for e in result["round_elapsed_s"]]
    costs = [float(c) for c in result["round_costs"]]
    final = float(result["final_cost"])

    def time_within_of_final(ratio: float) -> float | None:
        target = final * (1.0 + ratio)
        for e, c in zip(elapsed, costs):
            if c <= target:
                return e
        return None

    rounds = int(result["rounds_completed"])
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    summary = {
        "side": side,
        "n": n,
        "instance_seed": args.seed,
        "gen_time_s": gen_time,
        "seed_time_s": seed_time,
        "budget_s": args.budget,
        "rounds": rounds,
        "elapsed_s": float(result["elapsed_s"]),
        "converged": bool(result["converged"]),
        "per_round_s": float(result["elapsed_s"]) / rounds,
        "final_cost": final,
        "time_within_1pct_of_final_s": time_within_of_final(0.01),
        "time_within_0p1pct_of_final_s": time_within_of_final(0.001),
        "time_within_0p01pct_of_final_s": time_within_of_final(0.0001),
        "time_within_0p001pct_of_final_s": time_within_of_final(0.00001),
        "peak_rss_gib": peak_gib,
        "dense_matrix_gib": n * n * 8 / 1024**3,
    }
    out_json = results_dir() / f"billion_{tag}.json"
    out_json.write_text(json.dumps(summary, indent=1))

    trace_path = results_dir() / f"billion_traces_{tag}.csv"
    with trace_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "stride_r", "stride_c", "elapsed_s", "cost"])
        writer.writeheader()
        for i, (e, c) in enumerate(zip(elapsed, costs)):
            writer.writerow({
                "round": i + 1,
                "stride_r": int(result["round_strides"][i]),
                "stride_c": int(result["round_strides_col"][i]),
                "elapsed_s": e,
                "cost": c,
            })

    if not args.no_save_assignment:
        asn_dir = results_dir() / "assignments"
        asn_dir.mkdir(exist_ok=True)
        np.save(asn_dir / f"billion_meander_{tag}.npy",
                result["assignment"].astype(np.int32))

    print(json.dumps(summary, indent=1), flush=True)
    print(f"wrote {out_json} and {trace_path}", flush=True)


if __name__ == "__main__":
    main()
