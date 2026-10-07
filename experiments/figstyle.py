"""Shared figure conventions for the paper.

Color follows the method, never its rank: every figure uses the same slots.
ShuffleSnap presets form an ordered family (one blue ramp, light -> dark with window
size); comparators take the first categorical slots (validated all-pairs on white:
worst CVD dE 9.2, normal-vision dE 24.0; aqua is below 3:1 contrast, so every chart
also carries marker shapes, a legend, and a table in the paper).
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SS_RAMP = {"SS-fast": "#5b9ceb", "SS-balanced": "#1c5cab", "SS-quality": "#0d366b"}
COLORS = dict(SS_RAMP)
COLORS.update({
    "BSP-OT": "#eb6834",
    "RasterFairy-master": "#1baf7a",
    "RasterFairy-pypi": "#1baf7a",
    "Exact-certified": INK,
    "Exact-certified-bisect": INK2,
    "Exact-dense-scipy": INK2,
    "Bisection": MUTED,
})
MARKERS = {"SS-fast": "o", "SS-balanced": "o", "SS-quality": "o", "BSP-OT": "s",
           "RasterFairy-master": "^", "RasterFairy-pypi": "v", "Exact-certified": "D", "Exact-certified-bisect": "P", "Exact-dense-scipy": "d",
           "Bisection": "x"}
NAMES = {"SS-fast": "ShuffleSnap fast (5×5)", "SS-balanced": "ShuffleSnap balanced (8×8, default)",
         "SS-quality": "ShuffleSnap quality (12×12)",
         "BSP-OT": "BSP-OT", "RasterFairy-master": "RasterFairy (GitHub master)",
         "RasterFairy-pypi": "RasterFairy (PyPI 1.0.6)", "Exact-certified": "Exact (certified, sparse)",
         "Exact-certified-bisect": "Exact (certified, bisection start)",
         "Exact-dense-scipy": "Exact (dense SciPy LSA)", "Bisection": "Bisection start only"}


def family(label: str) -> str:
    if label.startswith("BSP-OT"):
        return "BSP-OT"
    return label


def color(label):
    return COLORS[family(label)]


def marker(label):
    return MARKERS[family(label)]


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.grid": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "figure.dpi": 100,
    })


def recess(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, which="major")
        ax.set_axisbelow(True)


# ------------------------------------------------------------------ source-position colors
def _lab_to_srgb(L, a, b):
    """CIELAB (D65) -> sRGB in [0, 1] (clipped)."""
    fy = (L + 16) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    eps = 216 / 24389
    kappa = 24389 / 27

    def finv(t):
        t3 = t ** 3
        return np.where(t3 > eps, t3, (116 * t - 16) / kappa)

    X = 0.95047 * finv(fx)
    Y = np.where(L > kappa * eps, fy ** 3, L / kappa)
    Z = 1.08883 * finv(fz)
    M = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    rgb = np.stack([X, Y, Z], -1) @ M.T
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)


def position_colors(P, L=62.0, chroma=58.0):
    """Color every point by its *input* position: the unit square of normalized
    coordinates is mapped onto a disc in the CIELAB a*/b* plane at constant lightness,
    so nearby inputs get similar colors and any tear or fold of the layout shows up as
    a color discontinuity.  Used identically for every method in a comparison."""
    P = np.asarray(P, dtype=float)
    u = (P - P.min(0)) / np.maximum(P.max(0) - P.min(0), 1e-12)
    x, y = 2 * u[:, 0] - 1, 2 * u[:, 1] - 1
    # square -> disc (keeps corners distinct, avoids clipping in the corners)
    xd = x * np.sqrt(np.maximum(0.0, 1 - y * y / 2))
    yd = y * np.sqrt(np.maximum(0.0, 1 - x * x / 2))
    return _lab_to_srgb(np.full(len(P), L), chroma * xd, chroma * yd)


def graph_paper(P, colors, n_lines=12, width=0.16, darken=0.5):
    """Draw graph paper on the *input* layout: points within width/2 line spacings of one
    of n_lines equally spaced lines per axis (over the point bounding box) are darkened.
    The lines travel with the points, so in an assignment they show how the input grid
    was bent, sheared or torn, which a smooth color field alone can hide."""
    P = np.asarray(P, dtype=float)
    u = (P - P.min(0)) / np.maximum(P.max(0) - P.min(0), 1e-12)
    f = u * n_lines
    on = (np.abs(f - np.round(f)) < width / 2).any(1)
    out = np.array(colors, dtype=float, copy=True)
    out[on] *= darken
    return out


def source_colors(P, **lines):
    """Input-position colors with graph-paper lines (the convention of every gallery)."""
    return graph_paper(P, position_colors(P), **lines)


def grid_image(grid, cell, values, fill=(1, 1, 1)):
    """H x W x 3 image with each point's RGB at its assigned cell (masked/empty = fill)."""
    img = np.empty((grid.height, grid.width, 3))
    img[:] = fill
    lat = grid.lattice[np.asarray(cell)]
    img[lat[:, 1], lat[:, 0]] = values
    return img
