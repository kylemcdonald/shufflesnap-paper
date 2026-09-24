"""Method adapters with a uniform interface.

Every adapter receives the *same* normalized float64 points ``Pn`` (grid units) and the
target :class:`shufflesnap.Grid`, and returns one cell id per point.  Timing (done by
the caller) covers the adapter call: all method-specific preprocessing (cost matrices,
trees, starts, internal normalization, format conversion) is included; building the
instance and the final validation are excluded.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

import shufflesnap as ss

HERE = os.path.dirname(os.path.abspath(__file__))
THIRD = os.path.abspath(os.path.join(HERE, "..", "..", "third_party"))


# --------------------------------------------------------------------------- ShuffleSnap
def ratio_schedule(width, height, window=6, ratio=2.0, rounds=1, start_stride=None):
    """Strides s0, s0/r, s0/r^2, ... (rounded, deduplicated) down to 1, per axis; s0 is the
    smallest power of two with window * s0 >= L, optionally capped at start_stride.
    ratio=2 reproduces the baseline rule."""
    def seq(L):
        s = 1
        while window * s < L:
            s *= 2
        if start_stride is not None:
            s = min(s, int(start_stride))
        out, x = [], float(s)
        while x > 1.0 + 1e-9:
            out.append(max(1, int(round(x))))
            x /= ratio
        return out
    sx, sy = seq(width), seq(height)
    n = max(len(sx), len(sy))
    sx = sx + [1] * (n - len(sx))
    sy = sy + [1] * (n - len(sy))
    st = []
    for a, b in zip(sx, sy):
        if (a, b) != (1, 1) and (not st or st[-1][:2] != [a, b]):
            st.append([a, b, rounds, 0])
    st.append([1, 1, -1, 1])
    return np.array(st, dtype=np.int32)


def m_shufflesnap(Pn, g, window=6, ratio=2.0, rounds=1, init="random", seed=0, threads=0, solver="hungarian_greedy",
                  time_budget=None, final_rounds=None, offsets=None, skip_clean=True, start_stride=None):
    stages = ratio_schedule(g.width, g.height, window, ratio, rounds, start_stride)
    if final_rounds is not None:
        stages[-1, 2] = int(final_rounds)
    res = ss.assign(Pn, g, normalize="none", init=init, seed=seed, window=window, schedule=stages,
                    time_budget=time_budget, threads=threads, solver=solver, offsets=offsets, skip_clean=skip_clean)
    return res.cell, {"finished": res.finished, "phases": int(len(res.trace)), "trace": res.trace[:, :2].tolist(),
                      "windows_solved": int(res.trace[:, 8].sum())}


def m_init(Pn, g, kind="bisect", seed=0):
    return ss.initial_assignment(Pn, g, kind, seed), {}


# --------------------------------------------------------------------------- exact
def m_exact(Pn, g, warm="shufflesnap", threads=0, k=12, radius=2):
    from shufflesnap import exact

    if warm == "shufflesnap":
        # the package default preset ("balanced"); its time is part of the exact run
        cell0 = ss.assign(Pn, g, normalize="none", preset="balanced", seed=0, threads=threads).cell
    else:
        cell0 = ss.initial_assignment(Pn, g, warm, 0)
    r = exact.solve_certified(Pn, g, cell0, k=k, radius=radius, threads=threads)
    return r["cell"], {"lower_bound": r["lower_bound"], "gap": r["gap"], "certified": r["certified"],
                       "rounds": len(r["rounds"]), "edges": r["rounds"][-1]["edges"]}


def m_dense(Pn, g, solver="scipy"):
    C = g.centers
    # float64 dense cost matrix, N x M
    D = np.empty((len(Pn), len(C)), dtype=np.float64)
    for a in range(0, len(Pn), 4096):
        d = Pn[a:a + 4096, None, :] - C[None, :, :]
        np.einsum("ijk,ijk->ij", d, d, out=D[a:a + 4096])
    if solver == "scipy":
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(D)
        cell = np.empty(len(Pn), dtype=np.int32)
        cell[r] = c
    elif solver == "lap":
        import lap

        _, x, _ = lap.lapjv(D, extend_cost=D.shape[0] != D.shape[1])
        cell = x.astype(np.int32)
    elif solver == "lapjv":
        import lapjv

        if D.shape[0] != D.shape[1]:
            raise ValueError("lapjv (src-d) needs a square matrix")
        row_ind, _, _ = lapjv.lapjv(D)
        cell = np.asarray(row_ind, dtype=np.int32)
    else:
        raise ValueError(solver)
    return cell, {}


# --------------------------------------------------------------------------- RasterFairy
def _rf_target(g):
    full = g.mask.all()
    if full and g.n_cells == g.width * g.height:
        return (g.width, g.height)
    return {"width": g.width, "height": g.height, "mask": (~g.mask).astype(int), "count": g.n_cells, "hex": False}


def m_rasterfairy(Pn, g, version="master", prescale=1000.0, hungarian_threshold=None, empty_strategy=None,
                  quiet=True):
    """version="pypi": rasterfairy 1.0.6 from PyPI (needs the np.float alias; sorts on
    integer-truncated coordinates, so inputs are multiplied by `prescale`).
    version="master": upstream GitHub master (normalizes internally)."""
    if version == "master":
        sys.path.insert(0, os.path.join(THIRD, "RasterFairy"))
    import rasterfairy

    if version == "master":
        assert os.path.join(THIRD, "RasterFairy") in rasterfairy.__file__, rasterfairy.__file__
    else:
        assert "site-packages" in rasterfairy.__file__
        if not hasattr(np, "float"):
            np.float = float  # removed alias used by rasterfairy 1.0.6
    target = _rf_target(g)
    kw = {}
    if hungarian_threshold is not None:
        kw["hungarian_threshold"] = hungarian_threshold
    if empty_strategy is not None:
        kw["emptyStrategy"] = empty_strategy
    X = Pn * prescale if version == "pypi" else Pn
    n = len(Pn)
    if g.n_cells > n:
        if isinstance(target, tuple):
            target = {"width": g.width, "height": g.height, "mask": np.zeros((g.height, g.width), dtype=int),
                      "count": g.n_cells, "hex": False}
    sink = io.StringIO()
    with (contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()):
        out = rasterfairy.transformPointCloud2D(X, target=target, **kw)
    if out is False:
        raise RuntimeError("rasterfairy returned False: " + sink.getvalue()[-500:])
    xy, (w, h) = out
    xy = np.rint(np.asarray(xy)).astype(np.int64)
    if (w, h) != (g.width, g.height):
        raise RuntimeError(f"rasterfairy used a {w}x{h} grid instead of {g.width}x{g.height}")
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < g.width) & (xy[:, 1] >= 0) & (xy[:, 1] < g.height)
    if not ok.all():
        raise RuntimeError("rasterfairy produced out-of-range cells")
    cell = g.index[xy[:, 1], xy[:, 0]].astype(np.int32)
    return cell, {"stdout_lines": sink.getvalue().count("\n")}


# --------------------------------------------------------------------------- BSP-OT
def m_bspot(Pn, g, num_plans=16, variant="gaussian", threads=0, precision="f64"):
    import bspot

    mod = bspot
    if precision == "f32":
        import bspot.f32 as mod  # noqa: F811
    mod.set_num_threads(int(threads))
    A = np.ascontiguousarray(Pn.T)
    B = np.ascontiguousarray(g.centers.T.astype(np.float64))
    if len(Pn) == g.n_cells:
        T = mod.compute_matching(A, B, num_plans=int(num_plans), orthogonal=variant == "orthogonal",
                                 gaussian=variant == "gaussian")
    else:
        if variant == "gaussian":
            raise ValueError("bspot partial matching has no gaussian option")
        T = mod.compute_partial_matching(A, B, num_plans=int(num_plans), orthogonal=variant == "orthogonal")
    return np.asarray(T, dtype=np.int32), {}


METHODS = {
    "shufflesnap": m_shufflesnap,
    "init": m_init,
    "exact": m_exact,
    "dense": m_dense,
    "rasterfairy": m_rasterfairy,
    "bspot": m_bspot,
}
