"""Benchmark harness: one method run per fresh subprocess.

Timing boundary: ``time_s`` covers exactly one call of the method adapter (see
methods.py): method-internal preprocessing, cost matrices, starts, and conversions are
included; instance generation, coordinate normalization (identical for all methods),
validation and quality metrics are excluded.  Before the timed call, every method is
run once on a tiny instance (``warmup_s``) so that imports, lazy initialization and
thread-pool start-up are not attributed to the measured run.  ShuffleSnap has no JIT;
its compile cost is a one-time build step recorded separately (see README).

Peak memory: the kernel's peak-RSS counter is reset (``/proc/self/clear_refs`` = 5)
right before the timed call; ``peak_rss_mb`` = VmHWM after the call minus VmRSS before
it, i.e. the extra resident memory the method needed on top of its inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _status():
    d = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmHWM", "VmRSS")):
                k, v = line.split(":")
                d[k] = int(v.split()[0]) / 1024.0
    return d


def _reset_peak():
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def spec_id(spec):
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def run_child(spec):
    sys.path.insert(0, HERE)
    import numpy as np

    import datasets
    import methods
    import reference
    import shufflesnap as ss

    fn = methods.METHODS[spec["method"]]
    params = dict(spec.get("params", {}))
    rec = dict(spec=spec, id=spec_id(spec))
    # warm-up on a tiny instance of the same grid type
    tw = time.perf_counter()
    try:
        Pw, gw, _, _ = datasets.instance("uniform", 256, 99, spec.get("grid", "square"))
        wparams = dict(params)
        if spec["method"] == "shufflesnap":
            wparams.pop("time_budget", None)
        fn(Pw, gw, **wparams)
    except Exception as e:  # warm-up failures are reported, the real run may still work
        rec["warmup_error"] = repr(e)
    rec["warmup_s"] = time.perf_counter() - tw
    Pn, g, labels, _ = datasets.instance(spec["dist"], spec["n"], spec["seed"], spec.get("grid", "square"))
    n = len(Pn)
    st0 = _status()
    _reset_peak()
    t0 = time.perf_counter()
    try:
        cell, info = fn(Pn, g, **params)
        err = None
    except Exception as e:
        cell, info, err = None, {}, repr(e)
    t1 = time.perf_counter()
    st1 = _status()
    rec["time_s"] = t1 - t0
    rec["peak_rss_mb"] = st1["VmHWM"] - st0["VmRSS"]
    rec["base_rss_mb"] = st0["VmRSS"]
    if err is not None:
        rec["error"] = err
        return rec
    cell = np.asarray(cell)
    try:
        ss.validate_assignment(cell, n, g)
        rec["valid"] = True
    except ValueError as e:
        rec["valid"] = False
        rec["error"] = "invalid assignment: " + str(e)
        return rec
    cost = ss.assignment_cost(Pn, g, cell)
    rec["cost"] = cost
    rec["mean_sq"] = cost / n
    rec["n_cells"] = g.n_cells
    rec["grid_wh"] = [g.width, g.height]
    rec["disp"] = ss.displacement_stats(Pn, g, cell)
    ref = reference.get(spec["dist"], spec["n"], spec["seed"], spec.get("grid", "square"), compute=False)
    if ref is not None:
        rec["ref_cost"] = ref["cost"]
        rec["ref_certified"] = ref["certified"]
        rec["ref_gap"] = ref["gap"]
        rec["excess_pct"] = 100.0 * (cost / ref["cost"] - 1.0) if ref["cost"] > 0 else None
        rec["excess_per_pt"] = (cost - ref["cost"]) / n
    info = dict(info)
    trace = info.pop("trace", None)
    if trace is not None and len(trace) > 1:
        tc = np.asarray(trace, dtype=float)[:, 1]
        rec["trace_monotone"] = bool(np.all(np.diff(tc) <= 1e-9 * np.maximum(tc[:-1], 1.0)))
        rec["trace_final_matches"] = bool(np.isclose(tc[-1], cost, rtol=1e-9))
    rec["info"] = info
    if spec.get("save_trace") and trace is not None:
        rec["trace"] = trace
    if spec.get("neighborhood"):
        from shufflesnap.metrics import knn_preservation, grid_neighbor_preservation

        rec["knn10"] = knn_preservation(Pn, g, cell, k=10, sample=min(n, 5000), seed=0)
        rec.update(grid_neighbor_preservation(Pn, g, cell, sample=min(n, 5000), seed=0))
    if spec.get("save_cell"):
        d = os.path.join(ROOT, "results", "assignments")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, rec["id"] + ".npy"), cell.astype(np.int32))
    return rec


def environment():
    import numpy
    import scipy

    info = dict(python=platform.python_version(), platform=platform.platform(), numpy=numpy.__version__,
                scipy=scipy.__version__)
    try:
        info["cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    for mod in ("shufflesnap", "bspot", "ortools", "lap", "lapjv", "rasterfairy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            info[mod] = "unavailable: " + repr(e)
    for repo in ("RasterFairy", "BSP-OT", "bspot-python"):
        try:
            info[repo + "_commit"] = subprocess.check_output(
                ["git", "-C", os.path.join(ROOT, "..", "third_party", repo), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            pass
    frozen = os.environ.get("SHUFFLESNAP_SRC")
    if frozen and os.path.exists(os.path.join(frozen, "..", "COMMIT")):
        info["shufflesnap_commit"] = open(os.path.join(frozen, "..", "COMMIT")).read().strip()
        info["shufflesnap_src"] = "frozen snapshot"
    else:
        try:
            info["shufflesnap_commit"] = subprocess.check_output(
                ["git", "-C", os.path.join(ROOT, "..", "shufflesnap"), "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL).strip()
        except Exception:
            info["shufflesnap_commit"] = "uncommitted"
    return info


def run_specs(specs, out, threads=1, cpus=None, timeout=3600, rerun=False, verbose=True):
    """Run each spec in a fresh subprocess, append JSON lines to `out` (resumable)."""
    done = set()
    if os.path.exists(out) and not rerun:
        with open(out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["id"])
                except Exception:
                    pass
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[v] = str(threads)
    # SHUFFLESNAP_SRC may point at a frozen snapshot of one commit (see run_all.sh), so that
    # editing the working tree during a long benchmark cannot change later runs.
    src = os.environ.get("SHUFFLESNAP_SRC", os.path.join(ROOT, "..", "shufflesnap", "src"))
    env["PYTHONPATH"] = os.pathsep.join([src, HERE, env.get("PYTHONPATH", "")])
    envinfo = environment()
    with open(out, "a") as f:
        for spec in specs:
            sid = spec_id(spec)
            if sid in done:
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--child", json.dumps(spec)]
            if cpus:
                cmd = ["taskset", "-c", cpus] + cmd
            t = time.time()
            try:
                p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
                line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                if p.returncode != 0 or not line:
                    rec = dict(spec=spec, id=sid, error=f"exit {p.returncode}: {p.stderr[-800:]}")
                else:
                    rec = json.loads(line[-1])
            except subprocess.TimeoutExpired:
                rec = dict(spec=spec, id=sid, error=f"timeout after {timeout}s", time_s=None)
            rec["threads"] = threads
            rec["cpus"] = cpus
            rec["env"] = envinfo
            rec["wall_s"] = time.time() - t
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if verbose:
                short = {k: rec.get(k) for k in ("time_s", "mean_sq", "excess_pct", "peak_rss_mb", "error")}
                print(spec["method"], json.dumps(spec.get("params", {})), spec["dist"], spec["n"], spec.get("grid"),
                      spec["seed"], json.dumps(short), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None)
    args = ap.parse_args()
    if args.child:
        rec = run_child(json.loads(args.child))
        print(json.dumps(rec, default=float))
