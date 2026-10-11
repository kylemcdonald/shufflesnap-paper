"""Explainer diagram for the package README (../shufflesnap/docs/how-it-works.svg).

    python experiments/fig_readme.py [out.svg]

Combines Figures 1 and 2 of the short paper in one self-contained SVG: the window solve,
strided windows and shifted tilings are drawn as vectors, and the MNIST progress panels are
embedded PNGs.  The SVG has its own white background so it reads the same in GitHub's light
and dark themes and on PyPI.
"""
from __future__ import annotations

import base64
import io
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

import shufflesnap as ss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402
import figstyle as fs  # noqa: E402
import reference  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "shufflesnap", "docs", "how-it-works.svg")
BLUE = "#2a78d6"
ORANGE = "#eb6834"
DOT = "#c9c8c1"
LINE = "#d7d6cf"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
W_SVG = 880
WIN = 8  # the default window


def hexcol(rgb):
    r, g, b = (int(round(255 * float(v))) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def text(x, y, s, size=12, color=fs.INK2, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}"'
            f' font-weight="{weight}">{s}</text>')


def heading(x, y, num, title):
    return (f'<circle cx="{x + 9}" cy="{y - 5}" r="9" fill="{fs.INK}"/>'
            + text(x + 9, y - 0.8, num, 11.5, "#ffffff", "middle", 700)
            + text(x + 25, y, title, 15, fs.INK, weight=600))


# ------------------------------------------------------------------------------ (1) window solve
def panel_solve(x0, y0):
    """Nine points and a 3x3 window: a poor assignment, then the exact reassignment (same
    instance as the paper's Figure 1a)."""
    rng = np.random.default_rng(4)
    cells = np.array([(x, y) for y in range(3) for x in range(3)], dtype=float)
    pts = cells[rng.permutation(9)] + rng.uniform(-0.42, 0.42, size=(9, 2))
    before = rng.permutation(9)
    cost = ((pts[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    after = linear_sum_assignment(cost)[1]
    cols = [hexcol(c) for c in fs.position_colors(pts, L=58.0, chroma=62.0)]
    cs, gap = 32.0, 34.0  # cell size, gap between the two windows
    out = []
    for k, (assign, label) in enumerate(((before, "before"), (after, "after"))):
        ox = x0 + k * (3 * cs + gap)
        oy = y0 + 22
        tot = cost[np.arange(9), assign].sum()
        out.append(text(ox + 1.5 * cs, y0 + 8, f"{label}: cost {tot:.1f}", 12, anchor="middle"))
        out.append(f'<rect x="{ox}" y="{oy}" width="{3 * cs}" height="{3 * cs}" fill="#ffffff" stroke="{LINE}"/>')
        for i in (1, 2):
            out.append(f'<path d="M{ox + i * cs} {oy}v{3 * cs}M{ox} {oy + i * cs}h{3 * cs}" stroke="{LINE}"/>')
        X = lambda v: ox + (v + 0.5) * cs  # noqa: E731
        Y = lambda v: oy + (v + 0.5) * cs  # noqa: E731
        for i in range(9):
            c = cells[assign[i]]
            out.append(f'<line x1="{X(pts[i, 0]):.1f}" y1="{Y(pts[i, 1]):.1f}" x2="{X(c[0]):.1f}" y2="{Y(c[1]):.1f}"'
                       f' stroke="{cols[i]}" stroke-width="1.6"/>')
        for i in range(9):
            c = cells[assign[i]]
            out.append(f'<rect x="{X(c[0]) - 4:.1f}" y="{Y(c[1]) - 4:.1f}" width="8" height="8" fill="{cols[i]}"'
                       ' stroke="#ffffff" stroke-width="0.8"/>')
            out.append(f'<circle cx="{X(pts[i, 0]):.1f}" cy="{Y(pts[i, 1]):.1f}" r="3.4" fill="{cols[i]}"'
                       f' stroke="{fs.INK}" stroke-width="0.5"/>')
    ax = x0 + 3 * cs + gap / 2
    ay = y0 + 22 + 1.5 * cs
    out.append(f'<path d="M{ax - 9} {ay}h15m-5 -4.5l5 4.5l-5 4.5" fill="none" stroke="{fs.INK2}" stroke-width="1.4"'
               ' stroke-linecap="round" stroke-linejoin="round"/>')
    leg_y = y0 + 22 + 3 * cs + 20
    out.append(f'<circle cx="{x0 + 5}" cy="{leg_y - 4}" r="3.4" fill="{fs.MUTED}" stroke="{fs.INK}" stroke-width="0.5"/>')
    out.append(text(x0 + 13, leg_y, "point", 11.5))
    out.append(f'<rect x="{x0 + 55}" y="{leg_y - 8}" width="8" height="8" fill="{fs.MUTED}"/>')
    out.append(text(x0 + 68, leg_y, "its assigned cell", 11.5))
    return "\n".join(out)


# ------------------------------------------------------------------------------ (2) strided windows
def panel_stride(x0, y0, s, title, sub, size=118.0):
    n = 4 * (WIN - 1) + 3  # the lattice holds the whole stride-4 window with a margin
    p = size / n
    oy = y0 + 36
    out = [text(x0 + size / 2, y0 + 8, title, 12, fs.INK, "middle", 600),
           text(x0 + size / 2, y0 + 24, sub, 12, anchor="middle")]
    dots = "".join(f"M{x0 + (i + 0.5) * p:.2f} {oy + (j + 0.5) * p:.2f}h0" for j in range(n) for i in range(n))
    out.append(f'<path d="{dots}" stroke="{DOT}" stroke-width="1.6" stroke-linecap="round"/>')
    first = 1 if s > 1 else (n - WIN) // 2
    idx = first + s * np.arange(WIN)
    q = 3.4 if s > 1 else p * 0.82
    cells = "".join(f"M{x0 + (i + 0.5) * p - q / 2:.2f} {oy + (j + 0.5) * p - q / 2:.2f}h{q:.2f}v{q:.2f}h{-q:.2f}z"
                    for j in idx for i in idx)
    out.append(f'<path d="{cells}" fill="{BLUE}"/>')
    lo, hi = idx[0] * p - 0.15 * p, (idx[-1] + 1) * p + 0.15 * p
    out.append(f'<rect x="{x0 + lo:.2f}" y="{oy + lo:.2f}" width="{hi - lo:.2f}" height="{hi - lo:.2f}" fill="none"'
               f' stroke="{fs.INK}" stroke-width="1"/>')
    return "\n".join(out)


# ------------------------------------------------------------------------------ (3) shifted tilings
def panel_tilings(x0, y0, size=220.0):
    """One subgrid tiled twice: unshifted (solid) and shifted by half a window (dashed).
    Windows are clipped at every edge, never dropped."""
    wa, hb = 2 * WIN + 1, WIN + 3
    h = WIN // 2
    p = size / wa
    oy = y0 + 22
    out = [text(x0, y0 + 8, "one subgrid, two of its four tilings", 12)]
    dots = "".join(f"M{x0 + (i + 0.5) * p:.2f} {oy + (j + 0.5) * p:.2f}h0" for j in range(hb) for i in range(wa))
    out.append(f'<path d="{dots}" stroke="{fs.MUTED}" stroke-width="1.8" stroke-linecap="round"/>')
    for off, style in ((h, f'stroke="{ORANGE}" stroke-width="1.6" stroke-dasharray="5 3.5"'),
                       (0, f'stroke="{fs.INK}" stroke-width="1.6"')):
        d = ""
        for x in range(off, wa, WIN):
            if 0 < x < wa:
                d += f"M{x0 + x * p:.2f} {oy}v{hb * p:.2f}"
        for y in range(off, hb, WIN):
            if 0 < y < hb:
                d += f"M{x0} {oy + y * p:.2f}h{wa * p:.2f}"
        out.append(f'<path d="{d}" fill="none" {style}/>')
    out.append(f'<rect x="{x0}" y="{oy}" width="{wa * p:.2f}" height="{hb * p:.2f}" fill="none" stroke="{fs.INK}"'
               ' stroke-width="1.6"/>')
    ly = oy + hb * p + 20
    out.append(f'<path d="M{x0 + 4} {ly - 4}h22" stroke="{fs.INK}" stroke-width="1.6"/>')
    out.append(text(x0 + 31, ly, "shift (0, 0)", 11.5))
    out.append(f'<path d="M{x0 + 104} {ly - 4}h22" stroke="{ORANGE}" stroke-width="1.6" stroke-dasharray="5 3.5"/>')
    out.append(text(x0 + 131, ly, f"shift ({h}, {h})", 11.5))
    return "\n".join(out)


# ------------------------------------------------------------------------------ (4) progress
def png_uri(rgb, scale):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def input_image(Pn, g, cols, px=300):
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(Pn[:, 0], Pn[:, 1], s=0.8, c=cols, linewidths=0)
    ax.set_xlim(-1, g.width)
    ax.set_ylim(g.height, -1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=float) / 255


def panel_progress(x0, y0, keep=("input", "start", 16, 8, 4, 1)):
    n = 10000
    Pn, g, _, _ = datasets.instance("mnist", n, 0)
    cols = fs.source_colors(Pn)
    res = ss.assign(Pn, g, normalize="none", preset="balanced", seed=1, snapshots=True)
    frames = {"input": (input_image(Pn, g, cols), None)}
    for name, cell in res.snapshots:
        key = "start" if name == "start" else int(name.split("s=")[1].split("x")[0])
        frames[key] = (fs.grid_image(g, cell, cols), ss.assignment_cost(Pn, g, cell) / n)
    final = res.snapshots[-1][1]
    frames[1] = (fs.grid_image(g, final, cols), ss.assignment_cost(Pn, g, final) / n)
    opt = reference.get("mnist", n, 0, compute=False)
    labels = {"input": "input (UMAP)", "start": "random start", 1: "stride 1, converged"}
    k = len(keep)
    gap = 14.0
    size = (W_SVG - 2 * x0 - (k - 1) * gap) / k
    out = []
    for j, key in enumerate(keep):
        img, cost = frames[key]
        x = x0 + j * (size + gap)
        label = labels.get(key, f"stride {key}")
        out.append(text(x + size / 2, y0 + 8, label, 12, fs.INK, "middle", 600))
        scale = 1 if key == "input" else 3
        out.append(f'<image x="{x:.1f}" y="{y0 + 18}" width="{size:.1f}" height="{size:.1f}"'
                   f' preserveAspectRatio="none" style="image-rendering:pixelated" href="{png_uri(img, scale)}"/>')
        out.append(f'<rect x="{x:.1f}" y="{y0 + 18}" width="{size:.1f}" height="{size:.1f}" fill="none"'
                   f' stroke="{fs.AXIS}"/>')
        if cost is not None:
            out.append(text(x + size / 2, y0 + 18 + size + 17, f"{cost:.1f}", 12.5, fs.INK, "middle"))
        if j < k - 1:
            ax_, ay_ = x + size + gap / 2, y0 + 18 + size / 2
            out.append(f'<path d="M{ax_ - 3} {ay_ - 4.5}l4 4.5l-4 4.5" fill="none" stroke="{fs.MUTED}"'
                       ' stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>')
    return "\n".join(out), size, (None if opt is None else opt["cost"] / n)


def build(out_path=OUT):
    pad = 28
    body = []
    # row 1: the three ingredients
    y1 = 40
    body.append(heading(pad, y1, "1", "Exact solve in each window"))
    body.append(panel_solve(pad, y1 + 22))
    xb = 318
    body.append(heading(xb, y1, "2", "Strides reach far"))
    body.append(panel_stride(xb, y1 + 22, 4, "stride 4", "64 cells, span 29×29"))
    body.append(panel_stride(xb + 136, y1 + 22, 1, "stride 1", "the same 64 cells"))
    xc = W_SVG - pad - 220
    body.append(heading(xc, y1, "3", "Shifted tilings"))
    body.append(panel_tilings(xc, y1 + 22))
    # captions
    yc = y1 + 22 + 208
    for x, lines in ((pad, ["The points in a window are reassigned", "among its cells by an exact solver;",
                            "the total cost never increases."]),
                     (xb, ["A window takes every s-th row and", "column, so points can travel far. The",
                           "stride shrinks by √2 per round to 1."]),
                     (xc, ["Each subgrid is tiled four times,", "shifted by half a window, so points",
                           "can cross every window border."])):
        for i, s in enumerate(lines):
            body.append(text(x, yc + 16 * i, s, 12, fs.INK2))
    # divider, row 2: the coarse-to-fine run
    y2 = yc + 64
    body.append(f'<path d="M{pad} {y2}h{W_SVG - 2 * pad}" stroke="{fs.GRID}"/>')
    y3 = y2 + 36
    body.append(heading(pad, y3, "4", "Coarse to fine: 10,000 MNIST digits snapped to a 100×100 grid"))
    prog, size, opt = panel_progress(pad, y3 + 18)
    body.append(prog)
    yf = y3 + 18 + 18 + size + 44
    body.append(text(pad, yf, "Each point keeps the color of its input position; dark bands are graph-paper lines drawn"
                     " on the input. Numbers are the mean", 12))
    body.append(text(pad, yf + 16, "squared displacement in cells²"
                     + ("." if opt is None else f"; the certified optimum is {opt:.1f}."), 12))
    height = yf + 34
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_SVG} {height:.0f}" width="{W_SVG}"'
           f' height="{height:.0f}" font-family="{FONT}">\n'
           f'<rect x="0.5" y="0.5" width="{W_SVG - 1}" height="{height - 1:.0f}" rx="10" fill="#ffffff"'
           f' stroke="#d0d7de"/>\n' + "\n".join(body) + "\n</svg>\n")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(svg)
    print("wrote", out_path, f"({len(svg) / 1024:.0f} KiB)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else OUT)
