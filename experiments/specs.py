"""Experiment definitions (lists of run specs) for the benchmark harness.

Instance seeds 1-3 are held out from development (seed 0).  Method seeds (random starts,
BSP-OT directions) are set per run where the method exposes one.
"""
from __future__ import annotations

R2 = 2 ** 0.5

SS = {
    "SS-baseline": dict(window=6, ratio=2.0),
    "SS-fast": dict(window=6, ratio=R2),
    "SS-balanced": dict(window=8, ratio=R2),
    "SS-quality": dict(window=12, ratio=R2),
}

MAIN_DISTS = ["uniform", "gaussian", "gmm", "unequal2", "ring", "spiral", "checker", "heavy", "duplicates", "line",
              "jitter"]


def ss_spec(label, dist, n, seed, grid="square", threads=1, method_seed=None, **extra):
    p = dict(SS[label])
    p.update(seed=seed if method_seed is None else method_seed, threads=threads)
    p.update(extra)
    return dict(method="shufflesnap", label=label, dist=dist, n=n, seed=seed, grid=grid, params=p)


def comparators(dist, n, seed, grid="square", threads=1, bsp_plans=(16, 64), rf=("master", "pypi"), exact=True,
                dense=False):
    out = []
    bij = grid in ("square", "wide", "disk", "ring", "logo")  # exactly n cells
    for k in bsp_plans:
        out.append(dict(method="bspot", label=f"BSP-OT-{k}", dist=dist, n=n, seed=seed, grid=grid,
                        params=dict(num_plans=k, variant="gaussian" if bij else "default", threads=threads)))
    for v in rf:
        out.append(dict(method="rasterfairy", label=f"RasterFairy-{v}", dist=dist, n=n, seed=seed, grid=grid,
                        params=dict(version=v)))
    if exact:
        out.append(dict(method="exact", label="Exact-certified", dist=dist, n=n, seed=seed, grid=grid,
                        params=dict(threads=threads)))
        if grid == "square":
            # same solver, seeded by the O(N log N) bisection start instead of ShuffleSnap
            out.append(dict(method="exact", label="Exact-certified-bisect", dist=dist, n=n, seed=seed, grid=grid,
                            params=dict(threads=threads, warm="bisect")))
    if dense:
        out.append(dict(method="dense", label="Exact-dense-scipy", dist=dist, n=n, seed=seed, grid=grid,
                        params=dict(solver="scipy")))
    out.append(dict(method="init", label="Bisection", dist=dist, n=n, seed=seed, grid=grid,
                    params=dict(kind="bisect")))
    return out


def main_quality(sizes=(10000, 50000, 100000), seeds=(1, 2, 3), dists=MAIN_DISTS, threads=1):
    """Core table: all methods, square grids, single thread."""
    specs = []
    for n in sizes:
        for d in dists:
            for s in seeds:
                for lab in SS:
                    specs.append(ss_spec(lab, d, n, s, threads=threads))
                # timed exact runs on the first held-out seed only (references cover the rest)
                specs += comparators(d, n, s, threads=threads, exact=(s == seeds[0]), dense=(n <= 10000 and s == seeds[0]))
    return specs


def real_embeddings(seeds=(1, 2, 3), threads=1):
    specs = []
    for d in ("mnist", "fashion"):
        for s in seeds:
            for lab in SS:
                specs.append(dict(ss_spec(lab, d, 70000, s, threads=threads), neighborhood=True, save_cell=True))
            for sp in comparators(d, 70000, s, threads=threads, exact=(s == seeds[0])):
                sp["neighborhood"] = True
                sp["save_cell"] = True
                specs.append(sp)
    return specs


def masks(n=20000, seeds=(1, 2, 3), dists=("uniform", "gmm", "spiral", "mnist"),
          grids=("free", "free10", "wide", "disk", "ring", "logo"), threads=1):
    specs = []
    for g in grids:
        for d in dists:
            for s in seeds:
                for lab in SS:
                    specs.append(ss_spec(lab, d, n, s, grid=g, threads=threads))
                specs += comparators(d, n, s, grid=g, threads=threads, bsp_plans=(16, 64), exact=(s == seeds[0]))
    return specs


def primes(seeds=(1, 2, 3), threads=1):
    """Prime point counts: exact-fit partial last row (square) vs surplus cells (free)."""
    specs = []
    for n in (49999, 99991):
        for g in ("square", "free"):
            for d in ("uniform", "gmm"):
                for s in seeds:
                    for lab in SS:
                        specs.append(ss_spec(lab, d, n, s, grid=g, threads=threads))
                    specs += comparators(d, n, s, grid=g, threads=threads, bsp_plans=(16,), exact=(s == seeds[0]))
    return specs


def scaling(dists=("uniform", "gmm"), threads=1, seed=1):
    sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    specs = []
    for d in dists:
        for n in sizes:
            for lab in SS:
                if lab == "SS-quality" and n > 200000 and threads == 1:
                    continue
                specs.append(ss_spec(lab, d, n, seed, threads=threads))
            # find the practical limits instead of assuming them (runs are capped at 1 h)
            exact_ok = n <= (1000000 if d == "uniform" else 200000)
            dense_ok = n <= (50000 if d == "uniform" else 20000)
            specs += comparators(d, n, seed, threads=threads, bsp_plans=(16,), rf=("master",),
                                 exact=exact_ok, dense=dense_ok)
    specs = [s for s in specs if s["label"] != "Exact-certified-bisect"]
    return specs


def threads_scaling(n=100000, seed=1):
    specs = []
    for T in (1, 2, 4, 8, 16, 24):
        for d in ("uniform", "gmm"):
            for lab in ("SS-baseline", "SS-balanced"):
                specs.append((T, ss_spec(lab, d, n, seed, threads=T)))
            specs.append((T, dict(method="bspot", label="BSP-OT-16", dist=d, n=n, seed=seed, grid="square",
                                  params=dict(num_plans=16, variant="gaussian", threads=T))))
    return specs


def anytime(dists=("uniform", "gmm", "mnist"), threads=1, seed=1):
    """Quality-versus-time: ShuffleSnap traces and budgets, BSP-OT plan sweep."""
    specs = []
    for d in dists:
        n = 70000 if d == "mnist" else 100000
        for lab in SS:
            specs.append(dict(ss_spec(lab, d, n, seed, threads=threads), save_trace=True))
        for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            specs.append(dict(method="bspot", label=f"BSP-OT-{k}", dist=d, n=n, seed=seed, grid="square",
                              params=dict(num_plans=k, variant="gaussian", threads=threads)))
        for thr in (1, 50, 200, 1000):
            specs.append(dict(method="rasterfairy", label=f"RasterFairy-master-h{thr}", dist=d, n=n, seed=seed,
                              grid="square", params=dict(version="master", hungarian_threshold=thr)))
    return specs


ABLATIONS = {
    "window": [dict(window=w, ratio=2.0) for w in (3, 4, 5, 6, 7, 8, 10, 12, 16)],
    "window-r1.41": [dict(window=w, ratio=R2) for w in (4, 6, 8, 10, 12, 16)],
    "ratio": [dict(window=6, ratio=r) for r in (3.0, 2.0, 2 ** 0.75, R2, 2 ** 0.25, 1.1)],
    "start": [dict(window=6, ratio=2.0, init=i) for i in ("random", "bisect", "rowsort")],
    "rounds": [dict(window=6, ratio=2.0, rounds=r) for r in (1, 2, 3)],
    "offsets": [dict(window=6, ratio=2.0, offsets=o) for o in ([[0, 0]], [[0, 0], [3, 3]],
                                                               [[0, 0], [0, 3], [3, 0], [3, 3]],
                                                               [[a, b] for a in (0, 2, 4) for b in (0, 2, 4)])],
    "solver": [dict(window=w, ratio=2.0, solver=sv) for w in (6, 12) for sv in ("hungarian", "hungarian_greedy", "jv")],
    "skip_clean": [dict(window=6, ratio=2.0, skip_clean=b) for b in (True, False)],
    "coarse": [dict(window=6, ratio=2.0, start_stride=c) for c in (None, 16, 8, 4, 2, 1)],
}


def ablations(dists=("uniform", "gmm", "mnist"), seeds=(1, 2, 3), threads=1):
    specs = []
    for d in dists:
        n = 70000 if d == "mnist" else 100000
        for s in seeds:
            for group, cfgs in ABLATIONS.items():
                for cfg in cfgs:
                    p = dict(cfg)
                    p.update(seed=s, threads=threads)
                    specs.append(dict(method="shufflesnap", label=f"abl-{group}", dist=d, n=n, seed=s, grid="square",
                                      params=p))
    return specs


def all_instances():
    """Unique (dist, n, seed, grid) needed by any experiment, for the reference cache."""
    specs = main_quality() + real_embeddings() + masks() + primes() + scaling() + anytime() + ablations()
    specs += [sp for _, sp in threads_scaling()]
    inst = {(sp["dist"], sp["n"], sp["seed"], sp.get("grid", "square")) for sp in specs}
    # certified references only where tractable (<= 200k points)
    return sorted((i for i in inst if i[1] <= 200000), key=lambda i: (i[1], i[0], i[3], i[2]))
