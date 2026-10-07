"""Seeded benchmark instances: point distributions x target cell sets.

Every instance is identified by (dist, n, seed, grid).  Points are generated in their
own coordinates and then mapped into the grid frame with the *same* normalization for
every method (``shufflesnap.normalize_points(..., "bbox")``), so all methods see
identical float64 inputs and identical cell centers.
"""
from __future__ import annotations

import os
import zlib
from functools import lru_cache

import numpy as np

import shufflesnap as ss

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

DISTS = ["uniform", "gaussian", "gmm", "unequal2", "ring", "spiral", "heavy", "duplicates", "line",
         "checker", "lattice", "jitter", "mnist", "fashion"]
GRIDS = ["square", "free", "wide", "disk", "ring", "logo", "free10"]


def _rng(*keys):
    ints = [k if isinstance(k, int) else zlib.crc32(str(k).encode()) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(ints))


@lru_cache(maxsize=4)
def _embedding(name):
    z = np.load(os.path.join(DATA, f"{name}_umap.npz"))
    return z["xy"], z["labels"]


def points(dist: str, n: int, seed: int, grid: ss.Grid | None = None):
    """Return (P, labels) with P of shape (n, 2); labels may be None."""
    rng = _rng(dist, n, seed)
    labels = None
    if dist == "uniform":
        P = rng.random((n, 2))
    elif dist == "gaussian":
        P = rng.normal(size=(n, 2))
    elif dist == "gmm":
        # 10 anisotropic clusters with uneven weights (UMAP-like cluster structure)
        crng = _rng("gmm-components", seed)
        k = 10
        means = crng.uniform(-10, 10, size=(k, 2))
        w = crng.dirichlet(np.full(k, 2.0))
        comp = rng.choice(k, size=n, p=w)
        ang = crng.uniform(0, np.pi, size=k)
        sc = crng.uniform(0.3, 1.5, size=(k, 2))
        Z = rng.normal(size=(n, 2)) * sc[comp]
        c, s = np.cos(ang[comp]), np.sin(ang[comp])
        P = np.c_[c * Z[:, 0] - s * Z[:, 1], s * Z[:, 0] + c * Z[:, 1]] + means[comp]
        labels = comp
    elif dist == "unequal2":
        m = int(round(0.9 * n))
        P = np.r_[rng.normal(size=(m, 2)), rng.normal(size=(n - m, 2)) * 0.5 + [8.0, 0.0]]
        labels = np.r_[np.zeros(m, int), np.ones(n - m, int)]
        perm = rng.permutation(n)
        P, labels = P[perm], labels[perm]
    elif dist == "ring":
        t = rng.uniform(0, 2 * np.pi, n)
        r = 1 + 0.05 * rng.normal(size=n)
        P = np.c_[r * np.cos(t), r * np.sin(t)]
    elif dist == "spiral":
        t = np.sqrt(rng.random(n)) * 4 * np.pi
        P = np.c_[t * np.cos(t), t * np.sin(t)] + 0.15 * rng.normal(size=(n, 2))
    elif dist == "heavy":
        P = rng.standard_t(1.5, size=(n, 2))
    elif dist == "duplicates":
        locs = _rng("dup-locs", seed).random((16, 2))
        P = locs[rng.integers(0, 16, n)]
    elif dist == "line":
        t = rng.random(n)
        P = np.c_[t, t] + 1e-3 * rng.normal(size=(n, 2))
    elif dist == "checker":
        # 8x8 checkerboard, dark squares 4x denser than light squares
        k = 8
        cells = np.array([(i, j) for i in range(k) for j in range(k)])
        dens = np.where((cells[:, 0] + cells[:, 1]) % 2 == 0, 4.0, 1.0)
        c = rng.choice(len(cells), size=n, p=dens / dens.sum())
        P = (cells[c] + rng.random((n, 2))) / k
    elif dist in ("lattice", "jitter"):
        if grid is None:
            raise ValueError("lattice/jitter need the target grid")
        if grid.n_cells != n:
            # choose n distinct cells at random for the points to sit on
            sel = np.sort(rng.choice(grid.n_cells, n, replace=False))
        else:
            sel = np.arange(n)
        P = grid.centers[sel].astype(np.float64).copy()
        if dist == "jitter":
            P += 0.25 * rng.normal(size=P.shape)
        perm = rng.permutation(n)
        P = P[perm]
    elif dist in ("mnist", "fashion"):
        xy, lab = _embedding(dist)
        if n > len(xy):
            raise ValueError(f"{dist} has only {len(xy)} points")
        idx = np.sort(rng.choice(len(xy), n, replace=False)) if n < len(xy) else np.arange(len(xy))
        P, labels = xy[idx], lab[idx]
    else:
        raise ValueError(dist)
    return np.ascontiguousarray(P, dtype=np.float64), labels


def _glyph_mask(n: int, text: str = "S"):
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    font = ImageFont.truetype(path, 400)
    img = Image.new("L", (360 * len(text), 480), 255)
    ImageDraw.Draw(img).text((10, 0), text, font=font, fill=0)
    return ss.Grid.from_image(np.asarray(img), n)


def s_mask(n: int, res: int = 600) -> ss.Grid:
    """The blocky rectilinear S of the earlier ShuffleSnap work (three horizontal bars joined
    by a left bar in the upper half and a right bar in the lower half, all 18-20% thick),
    rasterized to exactly n cells; any surplus is trimmed from the end of the bottom bar."""
    y, x = np.mgrid[:res, :res] / (res - 1)  # y grows downwards (row order)
    horizontal = (y < 0.18) | ((y > 0.41) & (y < 0.59)) | (y > 0.82)
    vertical = ((x < 0.2) & (y < 0.5)) | ((x > 0.8) & (y >= 0.5))
    img = np.where(horizontal | vertical, 0, 255).astype(np.uint8)
    return ss.Grid.from_image(img, n)


def ring_mask(n: int, inner: float = 0.5) -> ss.Grid:
    """Annulus with inner/outer radius ratio `inner` and exactly n cells."""
    area = np.pi * (1 - inner ** 2)
    R = np.sqrt(n / area) + 2
    size = int(2 * np.ceil(R) + 1)
    c = (size - 1) / 2
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    # rank cells by distance to the annulus mid-radius band, grow until n cells
    lo, hi = 0.0, R
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cnt = int(((d <= mid) & (d >= inner * mid)).sum())
        if cnt < n:
            lo = mid
        else:
            hi = mid
    Ro = hi
    inside = (d <= Ro) & (d >= inner * Ro)
    # trim to exactly n by removing outermost-by-angle cells
    score = np.where(inside, np.abs(d - 0.5 * (1 + inner) * Ro), np.inf).ravel()
    order = np.lexsort((np.arctan2(yy - c, xx - c).ravel(), score))
    m = np.zeros(size * size, bool)
    m[order[:n]] = True
    m = m.reshape(size, size)
    rows = np.nonzero(m.any(1))[0]
    cols = np.nonzero(m.any(0))[0]
    return ss.Grid.from_mask(m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1])


def grid_for(kind: str, n: int) -> ss.Grid:
    if kind == "square":
        return ss.Grid.for_count(n, partial="center")
    if kind == "free":
        return ss.Grid.for_count(n, partial="free")
    if kind == "free10":
        return ss.Grid.for_count(int(np.ceil(1.1 * n)), partial="free")
    if kind == "wide":
        return ss.Grid.for_count(n, aspect=4.0, partial="center")
    if kind == "disk":
        return ss.Grid.disk(n)
    if kind == "ring":
        return ring_mask(n)
    if kind == "logo":
        return _glyph_mask(n, "S")
    if kind == "srect":
        return s_mask(n)
    raise ValueError(kind)


def instance(dist: str, n: int, seed: int, grid: str = "square", normalize: str = "bbox"):
    """(P_normalized, grid, labels, raw_points) for one benchmark instance."""
    g = grid_for(grid, n)
    P, labels = points(dist, n, seed, g)
    if dist in ("lattice", "jitter"):
        Pn = P.copy() if dist == "lattice" else ss.normalize_points(P, g, normalize)
        if dist == "lattice":
            Pn = np.ascontiguousarray(Pn)
    else:
        Pn = ss.normalize_points(P, g, normalize)
    return Pn, g, labels, P


def instance_key(dist, n, seed, grid="square", normalize="bbox"):
    return f"{dist}-n{n}-s{seed}-{grid}-{normalize}"
