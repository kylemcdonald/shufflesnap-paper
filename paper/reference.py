"""ShuffleSnap in NumPy and SciPy, written to be read: serial, unit cells.

Without exact ties it returns the same cells as the compiled kernel,
``shufflesnap.assign(points, grid, normalize="none", ...)`` (tested).
``snap(P, mask)`` is the default preset; ``w=5`` and ``w=12`` give the
fast and quality presets.
"""
from collections import defaultdict
from itertools import zip_longest

import numpy as np
from scipy.optimize import linear_sum_assignment


def strides(length, w, ratio):
    """Strides on one axis: the first window spans it, then divide by ratio."""
    s = 1
    while w * s < length:
        s *= 2
    out = []
    while s > 1 + 1e-9:
        out.append(max(1, round(s)))
        s /= ratio
    return out


def schedule(width, height, w, ratio):
    """Stride pairs (sx, sy) before stride 1; the shorter axis waits at 1."""
    sx, sy = strides(width, w, ratio), strides(height, w, ratio)
    pairs = []
    for pair in zip_longest(sx, sy, fillvalue=1):
        if pair != (1, 1) and pair not in pairs[-1:]:
            pairs.append(pair)
    return pairs


def windows(cx, cy, sx, sy, w, ox, oy):
    """One tiling: cells in the same residue class and the same tile."""
    tx = (cx // sx + w - ox) // w  # tiles start at lattice column ox
    ty = (cy // sy + w - oy) // w
    groups = defaultdict(list)
    for cell, key in enumerate(zip(cx % sx, cy % sy, tx, ty)):
        groups[key].append(cell)
    return [np.array(cells) for cells in groups.values()]


def solve_window(P, C, cells, occ, pos, tol):
    """Reassign the window's points among its cells; keep it if better."""
    pts = occ[cells][occ[cells] >= 0]  # the points in these cells
    cost = ((P[pts, None] - C[None, cells]) ** 2).sum(-1)  # points x cells
    new = cells[linear_sum_assignment(cost)[1]]
    old_cost = ((P[pts] - C[pos[pts]]) ** 2).sum()
    new_cost = ((P[pts] - C[new]) ** 2).sum()
    if new_cost >= old_cost * (1 - tol):  # strict improvements only
        return False
    occ[cells] = -1
    occ[new] = pts
    pos[pts] = new
    return True


def snap(P, mask, w=8, ratio=2 ** 0.5, seed=0, tol=1e-12):
    """Cell of each point; P is (N, 2) in grid units, cells are row-major."""
    cy, cx = np.nonzero(np.asarray(mask, dtype=bool))
    C = np.stack([cx, cy], axis=1).astype(float)  # cell centers (x, y)
    rng = np.random.default_rng(seed)
    pos = rng.permutation(len(C))[:len(P)]  # random start: cell of point
    occ = np.full(len(C), -1)  # point in each cell, -1 if empty
    occ[pos] = np.arange(len(P))
    h = w // 2

    def one_round(sx, sy):  # four tilings, shifted by half a window
        moved = 0
        for ox, oy in [(0, 0), (0, h), (h, 0), (h, h)]:
            for cells in windows(cx, cy, sx, sy, w, ox, oy):
                moved += solve_window(P, C, cells, occ, pos, tol)
        return moved

    for sx, sy in schedule(mask.shape[1], mask.shape[0], w, ratio):
        one_round(sx, sy)
    while one_round(1, 1):  # stride 1 until nothing moves
        pass
    return pos
