"""Development-set exploration of ShuffleSnap variants (seed 0 only; not used for final claims).

A variant is a list of passes; each pass = (window, stage table, offsets or None), run in
sequence on the same assignment.  Excess is measured against the certified optimum.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import shufflesnap as ss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
import reference  # noqa: E402


def offsets_grid(w, k):
    """k x k phase offsets evenly spaced in [0, w)."""
    vals = sorted({int(round(i * w / k)) for i in range(k)})
    return np.array([(a, b) for a in vals for b in vals], dtype=np.int32)


def run_variant(Pn, g, passes, init="random", seed=0, threads=0):
    t0 = time.perf_counter()
    if isinstance(init, str) and init.startswith("strat"):
        pos = stratified_start(Pn, g, int(init[5:]), seed)
    else:
        pos = ss.initial_assignment(Pn, g, init, seed)
    t_init = time.perf_counter() - t0
    total_solved = 0
    for window, stages, offsets in passes:
        tr, fin = ss.run_schedule(Pn, g, pos, stages, window=window, offsets=offsets, threads=threads)
        total_solved += tr[:, 8].sum()
    t = time.perf_counter() - t0
    return pos, t, t_init, total_solved


def halving(g, window=6, start=None, rounds=1, final=-1):
    return ss.build_schedule(g.width, g.height, window, "halving", start, rounds, None if final < 0 else final)


VARIANTS = {
    # name: (init, passes-builder)
    "baseline": ("random", lambda g: [(6, halving(g), None)]),
    "bisect": ("bisect", lambda g: [(6, halving(g), None)]),
    "rowsort": ("rowsort", lambda g: [(6, halving(g), None)]),
    "bisect-s8": ("bisect", lambda g: [(6, halving(g, start=8), None)]),
    "bisect-s4": ("bisect", lambda g: [(6, halving(g, start=4), None)]),
    "bisect-flat": ("bisect", lambda g: [(6, np.array([[1, 1, -1, 1]], np.int32), None)]),
    "random-2rounds": ("random", lambda g: [(6, halving(g, rounds=2), None)]),
    "w4": ("random", lambda g: [(4, halving(g, window=4), None)]),
    "w8": ("random", lambda g: [(8, halving(g, window=8), None)]),
    "w10": ("random", lambda g: [(10, halving(g, window=10), None)]),
    "w12": ("random", lambda g: [(12, halving(g, window=12), None)]),
    "9phase": ("random", lambda g: [(6, halving(g), offsets_grid(6, 3))]),
    "polish-w12": ("random", lambda g: [(6, halving(g), None), (12, np.array([[1, 1, -1, 1]], np.int32), None)]),
    "polish-vcycle": ("random", lambda g: [(6, halving(g), None)] + [
        (6, np.array([[2, 2, 1, 0], [1, 1, -1, 1]], np.int32), None)] * 3),
    "bisect-polish-w12": ("bisect", lambda g: [(6, halving(g), None), (12, np.array([[1, 1, -1, 1]], np.int32), None)]),
    "bisect-w12": ("bisect", lambda g: [(12, halving(g, window=12), None)]),
    # mixed window sizes: small windows at coarse strides, large at fine strides
    "w6c-w12f": ("random", lambda g: [(6, coarse(g, 6, 2), None), (12, np.array([[1, 1, -1, 1]], np.int32), None)]),
    "w6c-w12f2": ("random", lambda g: [(6, coarse(g, 6, 4), None), (12, np.array([[2, 2, 1, 0], [1, 1, -1, 1]], np.int32), None)]),
    "w6c-w8f": ("random", lambda g: [(6, coarse(g, 6, 2), None), (8, np.array([[1, 1, -1, 1]], np.int32), None)]),
    "w6c-w10f": ("random", lambda g: [(6, coarse(g, 6, 2), None), (10, np.array([[1, 1, -1, 1]], np.int32), None)]),
    "w6c-w16f": ("random", lambda g: [(6, coarse(g, 6, 2), None), (16, np.array([[1, 1, -1, 1]], np.int32), None)]),
    "w12c-w6f": ("random", lambda g: [(12, coarse(g, 12, 2), None), (6, np.array([[1, 1, -1, 1]], np.int32), None)]),
    "w8-2rounds": ("random", lambda g: [(8, halving(g, window=8, rounds=2), None)]),
    "ratio1.5": ("random", lambda g: [(6, ratio_schedule(g, 6, 1.5), None)]),
    "ratio1.41": ("random", lambda g: [(6, ratio_schedule(g, 6, 2 ** 0.5), None)]),
    "ratio3": ("random", lambda g: [(6, ratio_schedule(g, 6, 3.0), None)]),
    "ratio1.25": ("random", lambda g: [(6, ratio_schedule(g, 6, 1.25), None)]),
    "ratio1.19": ("random", lambda g: [(6, ratio_schedule(g, 6, 2 ** 0.25), None)]),
    "ratio1.1": ("random", lambda g: [(6, ratio_schedule(g, 6, 1.1), None)]),
    "linear": ("random", lambda g: [(6, linear_schedule(g, 6), None)]),
    "r1.41-w8": ("random", lambda g: [(8, ratio_schedule(g, 8, 2 ** 0.5), None)]),
    "r1.41-w10": ("random", lambda g: [(10, ratio_schedule(g, 10, 2 ** 0.5), None)]),
    "r1.41-w12": ("random", lambda g: [(12, ratio_schedule(g, 12, 2 ** 0.5), None)]),
    "r1.19-w8": ("random", lambda g: [(8, ratio_schedule(g, 8, 2 ** 0.25), None)]),
    "r1.19-w12": ("random", lambda g: [(12, ratio_schedule(g, 12, 2 ** 0.25), None)]),
    "strat2": ("strat2", lambda g: [(6, halving(g), None)]),
    "strat4": ("strat4", lambda g: [(6, halving(g), None)]),
    "strat8": ("strat8", lambda g: [(6, halving(g), None)]),
    "strat16": ("strat16", lambda g: [(6, halving(g), None)]),
    "strat32": ("strat32", lambda g: [(6, halving(g), None)]),
    "strat64": ("strat64", lambda g: [(6, halving(g), None)]),
}


def stratified_start(Pn, g, block, seed=0):
    """Bisection start, then a random permutation of occupants inside each block x block tile."""
    pos = ss.initial_assignment(Pn, g, "bisect")
    rng = np.random.default_rng(seed)
    lat = g.lattice[pos]
    tile = (lat[:, 1] // block) * ((g.width + block - 1) // block) + lat[:, 0] // block
    order = np.argsort(tile, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(tile, minlength=tile.max() + 1))]
    for t in range(len(bounds) - 1):
        idx = order[bounds[t]:bounds[t + 1]]
        if len(idx) > 1:
            pos[idx] = pos[rng.permutation(idx)]
    return pos


def ratio_schedule(g, window=6, ratio=2.0):
    """Strides s0, s0/r, s0/r^2, ... (rounded, deduplicated) down to 1; per-axis rule for s0."""
    def seq(L):
        s = 1
        while window * s < L:
            s *= 2
        out, x = [], float(s)
        while x > 1.0:
            out.append(max(1, int(round(x))))
            x /= ratio
        return out
    sx, sy = seq(g.width), seq(g.height)
    n = max(len(sx), len(sy))
    sx = sx + [1] * (n - len(sx))
    sy = sy + [1] * (n - len(sy))
    st = []
    for a, b in zip(sx, sy):
        if (a, b) != (1, 1) and (not st or st[-1][:2] != [a, b]):
            st.append([a, b, 1, 0])
    st.append([1, 1, -1, 1])
    return np.array(st, dtype=np.int32)


def linear_schedule(g, window=6):
    """Every integer stride from the start stride down to 1 (same stride on both axes = max)."""
    s = 1
    while window * s < max(g.width, g.height):
        s *= 2
    st = [[k, k, 1, 0] for k in range(s, 1, -1)]
    st.append([1, 1, -1, 1])
    return np.array(st, dtype=np.int32)


def coarse(g, window, smallest):
    """Halving stages of the baseline rule, stopping at stride `smallest` (1 round each)."""
    st = halving(g, window=window)
    keep = [r for r in st if max(r[0], r[1]) >= smallest and not (r[0] == 1 and r[1] == 1)]
    return np.array(keep, dtype=np.int32).reshape(-1, 4)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", nargs="+", default=["uniform", "gaussian", "gmm", "jitter"])
    ap.add_argument("--n", nargs="+", type=int, default=[10000])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = open(args.out, "a") if args.out else None
    for d in args.dist:
        for n in args.n:
            Pn, g, _, _ = datasets.instance(d, n, 0)
            ref = reference.get(d, n, 0)
            opt = ref["cost"]
            for name in args.variants:
                init, builder = VARIANTS[name]
                pos, t, t_init, solved = run_variant(Pn, g, builder(g), init, 0, args.threads)
                ss.validate_assignment(pos, n, g)
                c = ss.assignment_cost(Pn, g, pos)
                rec = dict(dist=d, n=n, variant=name, cost=c, opt=opt, excess_pct=100 * (c / opt - 1) if opt > 0 else None,
                           excess_abs_per_pt=(c - opt) / n, time_s=t, init_s=t_init, windows_solved=int(solved))
                print(json.dumps(rec), flush=True)
                if out:
                    out.write(json.dumps(rec) + "\n")
                    out.flush()


if __name__ == "__main__":
    main()
