"""Shared utilities for the ShuffleSnap paper benchmarks.

All benchmarks write CSV files into benchmarks/results/. Exact reference costs
are cached in benchmarks/results/exact_cache.json so repeated runs are cheap.
"""

from __future__ import annotations

import json
import hashlib
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
SOLVER_SOURCE = REPO / "solvers" / "cpp" / "exact_grid_lap.cpp"
SOLVER_BINARY = REPO / "solvers" / "cpp" / "exact_grid_lap"
EXACT_CACHE = RESULTS / "exact_cache.json"



def results_dir() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


def ensure_exact_solver() -> Path:
    if not SOLVER_BINARY.exists() or SOLVER_BINARY.stat().st_mtime < SOLVER_SOURCE.stat().st_mtime:
        subprocess.run(
            ["c++", "-O3", "-std=c++20", "-pthread", str(SOLVER_SOURCE), "-o", str(SOLVER_BINARY)],
            check=True,
        )
    return SOLVER_BINARY


# ---------------------------------------------------------------- point sets

def normalize(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    mins, maxs = points.min(axis=0), points.max(axis=0)
    span = np.maximum(maxs - mins, np.finfo(np.float64).eps)
    return (points - mins) / span


def make_points(name: str, n: int, seed: int = 0) -> np.ndarray:
    """Synthetic point families. All normalized into the unit square."""
    rng = np.random.default_rng(seed)
    if name == "meander":
        steps = rng.normal(size=(n, 2))
        return normalize(np.cumsum(steps, axis=0))
    if name == "uniform":
        return normalize(rng.random((n, 2)))
    if name == "two_blobs":
        half = n // 2
        a = rng.normal([0.2, 0.2], 0.05, size=(half, 2))
        b = rng.normal([0.8, 0.8], 0.05, size=(n - half, 2))
        return normalize(np.vstack([a, b]))
    if name == "ring":
        theta = rng.uniform(0, 2 * np.pi, n)
        r = 0.35 + rng.normal(0, 0.02, n)
        return normalize(np.column_stack([0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)]))
    if name == "spiral":
        t = np.sqrt(rng.uniform(0, 1, n)) * 3 * 2 * np.pi
        r = 0.05 + 0.4 * t / (3 * 2 * np.pi)
        x = r * np.cos(t) + rng.normal(0, 0.01, n)
        y = r * np.sin(t) + rng.normal(0, 0.01, n)
        return normalize(np.column_stack([x, y]))
    if name == "mixture":
        centers = rng.random((5, 2))
        weights = rng.dirichlet(np.ones(5) * 2)
        counts = np.round(weights * n).astype(int)
        counts[-1] = n - counts[:-1].sum()
        parts = [
            rng.normal(c, 0.04 + 0.08 * rng.random(), size=(k, 2))
            for c, k in zip(centers, counts)
        ]
        return normalize(np.vstack(parts))
    raise ValueError(f"unknown distribution: {name}")


DISTRIBUTIONS = ["meander", "two_blobs", "ring", "spiral", "mixture"]


# ------------------------------------------------------------------- targets

def build_target_grid(width: int, height: int) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, width) if width > 1 else np.array([0.5])
    ys = np.linspace(0.0, 1.0, height) if height > 1 else np.array([0.5])
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack((gx.ravel(), gy.ravel()))


def assignment_cost(points: np.ndarray, targets: np.ndarray, assignment: np.ndarray) -> float:
    d = points - targets[np.asarray(assignment, dtype=np.int64)]
    return float(np.einsum("ij,ij->", d, d))


# --------------------------------------------------------------------- seeds

def seed_raster(points: np.ndarray, total_cells: int | None = None) -> np.ndarray:
    """Sort by (y, x), place on evenly spread cells in row-major order."""
    n = len(points)
    m = n if total_cells is None else int(total_cells)
    order = np.lexsort((points[:, 0], points[:, 1]))
    cells = np.floor(np.arange(n, dtype=np.float64) * (m / n)).astype(np.int64)
    assignment = np.empty(n, dtype=np.int64)
    assignment[order] = cells
    return assignment


def seed_random(n: int, rng_seed: int = 0) -> np.ndarray:
    return np.random.default_rng(rng_seed).permutation(n).astype(np.int64)


def seed_random_cells(n: int, total_cells: int, rng_seed: int = 0) -> np.ndarray:
    """Random subset of cells in random order; a full permutation when
    n == total_cells."""
    rng = np.random.default_rng(rng_seed)
    return rng.permutation(total_cells)[:n].astype(np.int64)


def seed_default(points: np.ndarray, total_cells: int) -> np.ndarray:
    """shufflesnap's default deterministic random assignment."""
    return seed_random_cells(len(points), total_cells, 0)


def rasterfairy_assignment(points: np.ndarray, width: int, height: int):
    """Assignment via the RasterFairy package (Klingemann 2016), timed.

    RasterFairy sorts integer-cast coordinates (it expects pixel-scale input
    such as t-SNE screen positions), so the unit-square cloud is scaled up
    before the call; the returned grid coordinates are unaffected. Returns
    (assignment, solve_time_s).
    """
    import time as time_mod

    np.float = float  # rasterfairy predates numpy 2 (np.bool still exists)
    np.int = int
    import rasterfairy

    pts = np.asarray(points, dtype=np.float64)
    t0 = time_mod.time()
    grid_xy, (w, h) = rasterfairy.transformPointCloud2D(
        pts * 1e6, target=(int(width), int(height)))
    elapsed = time_mod.time() - t0
    if (int(w), int(h)) != (int(width), int(height)):
        raise RuntimeError(f"rasterfairy returned grid {w}x{h}, wanted {width}x{height}")
    cols_idx = np.asarray(np.round(grid_xy[:, 0]), dtype=np.int64)
    rows_idx = np.asarray(np.round(grid_xy[:, 1]), dtype=np.int64)
    return rows_idx * int(width) + cols_idx, elapsed


# ------------------------------------------------------------- exact oracle

def _load_cache() -> dict:
    if EXACT_CACHE.exists():
        return json.loads(EXACT_CACHE.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    results_dir()
    EXACT_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))


def exact_square(points: np.ndarray, width: int, height: int,
                 key: str | None = None) -> dict:
    """Exact optimal cost via the on-demand auction oracle (square case).

    Returns {"cost": float, "solve_time_s": float}. Cached by key.
    """
    fingerprint = hashlib.sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest()
    cache = _load_cache()
    if key is not None and key in cache:
        entry = cache[key]
        if (isinstance(entry, dict) and entry.get("points_sha256") == fingerprint
                and entry.get("grid") == [width, height, "unit-square"]):
            return entry
    binary = ensure_exact_solver()
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = Path(f.name)
        f.write(np.asarray(points, dtype=np.float64).tobytes(order="C"))
    assignment_path = path.with_suffix(".assignment")
    try:
        out = subprocess.run(
            [str(binary), "--points", str(path), "--width", str(width),
             "--height", str(height), "--solver", "auction", "--cost-mode", "jit",
             "--assignment-out", str(assignment_path)],
            check=True, capture_output=True, text=True,
        ).stdout
        assignment = np.fromfile(assignment_path, dtype=np.int64)
    finally:
        path.unlink()
        assignment_path.unlink(missing_ok=True)
    cost = None
    solve_time = None
    for line in out.splitlines():
        if line.startswith("assignment_cost="):
            cost = float(line.split("=", 1)[1])
        elif line.startswith("solve_time_s="):
            solve_time = float(line.split("=", 1)[1])
    if cost is None or solve_time is None:
        raise RuntimeError(f"oracle omitted cost or solve time:\n{out}")
    entry = {"cost": cost, "solve_time_s": solve_time,
             "points_sha256": fingerprint, "grid": [width, height, "unit-square"]}
    # The distribution gallery uses the 200x200 seed-0 cases and solver-only timing.
    if key is not None and width == height == 200 and key.endswith(":seed0"):
        figure_cache = results_dir() / "assignments" / f"exact_{key.split(':')[0]}_200_seed0.npz"
        figure_cache.parent.mkdir(exist_ok=True)
        np.savez(figure_cache, assignment=assignment, time_s=solve_time)
    if key is not None:
        cache = _load_cache()
        cache[key] = entry
        _save_cache(cache)
    return entry


def exact_instance(dist: str, n_side: int, seed: int) -> dict:
    key = f"{dist}:{n_side}x{n_side}:seed{seed}"
    points = make_points(dist, n_side * n_side, seed=seed)
    return exact_square(points, n_side, n_side, key=key)


def exact_square_assignment(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Exact optimal assignment (cell ids) via the on-demand auction oracle."""
    binary = ensure_exact_solver()
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        pts_path = Path(f.name)
        f.write(np.asarray(points, dtype=np.float64).tobytes(order="C"))
    out_path = pts_path.with_suffix(".assignment")
    try:
        subprocess.run(
            [str(binary), "--points", str(pts_path), "--width", str(width),
             "--height", str(height), "--solver", "auction", "--cost-mode", "jit",
             "--assignment-out", str(out_path)],
            check=True, capture_output=True, text=True,
        )
        assignment = np.fromfile(out_path, dtype=np.int64)
    finally:
        pts_path.unlink()
        out_path.unlink(missing_ok=True)
    if assignment.shape[0] != points.shape[0]:
        raise RuntimeError("oracle returned a truncated assignment")
    return assignment


# -------------------------------------------------------------- environment

def write_environment() -> None:
    import shufflesnap  # noqa: F401

    info = {
        "platform": platform.platform(),
        "processor": "",
        "cpu_count": None,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import os
        info["cpu_count"] = os.cpu_count()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["processor"] = line.split(":", 1)[1].strip()
                break
        mem_kb = 0
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem_kb = int(line.split()[1])
                break
        info["memory_gib"] = round(mem_kb / (1024 * 1024), 1)
    except OSError:
        pass
    (results_dir() / "environment.json").write_text(json.dumps(info, indent=1))


def gap_pct(cost: float, optimum: float) -> float:
    return 100.0 * (cost - optimum) / optimum
