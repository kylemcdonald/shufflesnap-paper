"""The annotated reference implementation: one diagram per part of paper/reference.py, and
the LaTeX that sets each diagram next to its code on landscape pages.

    python experiments/fig_reference.py

Every diagram is computed with the reference's own functions (strides, schedule, windows,
solve_window), so the pictures cannot drift from the code.  Writes paper/figures/fig_ref_*
and paper/sections/reference_annotated.tex, whose line ranges are read from
paper/reference.py; rerun it whenever that file changes.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "reference.py"
FIGS = ROOT / "paper" / "figures"
TEX = ROOT / "paper" / "sections" / "reference_annotated.tex"

_spec = importlib.util.spec_from_file_location("reference_printed", SOURCE)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#dcdbd4"
WIDTH = 3.75  # inches: the diagram column of a landscape page


def _save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def _frame(ax, W, H):
    ax.set_xlim(-0.8, W - 0.2)
    ax.set_ylim(H - 0.2, -0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def _cells(W, H):
    cy, cx = np.nonzero(np.ones((H, W), dtype=bool))
    return cx, cy


def _square(ax, x, y, color, size=1.0, **kw):
    ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, fc=color, ec="none", **kw))


# ------------------------------------------------------------------------------ schedule
def fig_schedule(L=64, w=8, ratio=2 ** 0.5):
    """One window at every stride of the schedule on an L x L grid."""
    cx, cy = _cells(L, L)
    stages = [s for s, _ in ref.schedule(L, L, w, ratio)] + [1]
    fig, axes = plt.subplots(2, (len(stages) + 1) // 2, figsize=(WIDTH, 2.45))
    for ax in axes.flat[len(stages):]:
        ax.axis("off")
    for ax, s in zip(axes.flat, stages):
        win = next(c for c in ref.windows(cx, cy, s, s, w, 0, 0) if 0 in c)
        ax.add_patch(Rectangle((-0.5, -0.5), L, L, fc="#f3f2ee", ec=fs.AXIS, lw=0.5))
        span = (w - 1) * s + 1
        ax.add_patch(Rectangle((-0.5, -0.5), span, span, fill=False, ec=fs.INK2, lw=0.5, ls=(0, (2, 1.5))))
        for c in win:
            _square(ax, cx[c], cy[c], BLUE, size=min(1.8, 0.8 * s))
        ax.set_title(f"stride {s}", fontsize=7.5, color=fs.INK2, pad=2)
        _frame(ax, L, L)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.12, hspace=0.16)
    _save(fig, "fig_ref_schedule")


# ------------------------------------------------------------------------------ windows
def fig_windows(G=16, s=2, w=4):
    """(a) the residue classes at stride s; (b, c) the windows of one class in the
    unshifted tiling and in the tiling shifted by half a window."""
    cx, cy = _cells(G, G)
    h = w // 2
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.42))
    klass = [BLUE, ORANGE, AQUA, fs.INK2]
    ax = axes[0]
    for c in range(len(cx)):
        _square(ax, cx[c], cy[c], klass[(cx[c] % s) + s * (cy[c] % s)], size=0.82)
    ax.set_title("(a) residue classes", fontsize=7, color=fs.INK2, pad=2)
    _frame(ax, G, G)
    for ax, (ox, oy), title in ((axes[1], (0, 0), "(b) shift (0, 0)"), (axes[2], (h, h), f"(c) shift ({h}, {h})")):
        for c in range(len(cx)):
            _square(ax, cx[c], cy[c], GRAY, size=0.82)
        for cells in ref.windows(cx, cy, s, s, w, ox, oy):
            if cx[cells[0]] % s or cy[cells[0]] % s:
                continue  # draw the windows of residue class (0, 0) only
            for c in cells:
                _square(ax, cx[c], cy[c], BLUE, size=0.82)
            x0, x1 = cx[cells].min(), cx[cells].max()
            y0, y1 = cy[cells].min(), cy[cells].max()
            ax.add_patch(Rectangle((x0 - 0.6, y0 - 0.6), x1 - x0 + 1.2, y1 - y0 + 1.2, fill=False, ec=fs.INK,
                                   lw=0.6))
        ax.set_title(title, fontsize=7, color=fs.INK2, pad=2)
        _frame(ax, G, G)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.1)
    _save(fig, "fig_ref_windows")


# ------------------------------------------------------------------------------ one window
def fig_solve(seed=3):
    """A 3 x 3 window with seven points: the current cells, the exact reassignment, and
    the cost matrix with the chosen entries."""
    rng = np.random.default_rng(seed)
    cx, cy = _cells(3, 3)
    C = np.stack([cx, cy], axis=1).astype(float)
    cells = np.arange(9)
    P = C[rng.permutation(9)[:7]] + rng.uniform(-0.45, 0.45, size=(7, 2))
    pos = rng.permutation(9)[:7]
    occ = np.full(9, -1)
    occ[pos] = np.arange(7)
    before = pos.copy()
    pts = occ[cells][occ[cells] >= 0]
    cost = ((P[pts, None] - C[None, cells]) ** 2).sum(-1)
    assert ref.solve_window(P, C, cells, occ, pos, 1e-12)
    cols = fs.position_colors(P, L=58.0, chroma=62.0)
    fig = plt.figure(figsize=(WIDTH, 1.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], left=0.01, right=0.99, top=0.84, bottom=0.03,
                          wspace=0.18)
    for k, (cell_of, title) in enumerate(((before, "before"), (pos, "after"))):
        ax = fig.add_subplot(gs[k])
        for c in cells:
            ax.add_patch(Rectangle((cx[c] - 0.5, cy[c] - 0.5), 1, 1, fill=False, ec=fs.AXIS, lw=0.6))
        for i in range(7):
            x, y = C[cell_of[i]]
            ax.plot([P[i, 0], x], [P[i, 1], y], color=cols[i], lw=1.1, zorder=1)
            ax.plot(x, y, marker="s", ms=4.0, color=cols[i], mec="white", mew=0.4, zorder=2)
            ax.plot(*P[i], marker="o", ms=3.0, color=cols[i], mec=fs.INK, mew=0.3, zorder=3)
        total = ((P - C[cell_of]) ** 2).sum()
        ax.set_title(f"{title}: {total:.2f}", fontsize=7, color=fs.INK2, pad=2)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(2.6, -0.6)
        ax.set_aspect("equal")
        ax.axis("off")
    ax = fig.add_subplot(gs[2])
    ax.imshow(cost, cmap="Greys", vmin=0, vmax=cost.max() * 1.6, aspect="equal")
    chosen = np.searchsorted(cells, pos[pts])
    ax.scatter(chosen, np.arange(len(pts)), s=14, facecolors="none", edgecolors=BLUE, linewidths=1.0)
    ax.set_xticks(range(9))
    ax.set_yticks(range(len(pts)))
    ax.tick_params(length=0, labelsize=5.5, pad=1)
    ax.set_xlabel("cells", fontsize=6, labelpad=1)
    ax.set_ylabel("points", fontsize=6, labelpad=1)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    _save(fig, "fig_ref_solve")


# ------------------------------------------------------------------------------ snap
def fig_snap(G=32, w=8, ratio=2 ** 0.5, seed=0):
    """A run of snap() on G x G uniform points, after the start, each coarse stage and
    the end; colors encode each point's input position."""
    rng = np.random.default_rng(seed)
    P = rng.uniform(-0.5, G - 0.5, size=(G * G, 2))
    colors = fs.position_colors(P)
    cx, cy = _cells(G, G)
    C = np.stack([cx, cy], axis=1).astype(float)
    pos = np.random.default_rng(seed).permutation(len(C))[:len(P)]
    occ = np.full(len(C), -1)
    occ[pos] = np.arange(len(P))
    h = w // 2
    snaps = [("start", pos.copy())]
    trace = []  # (stride, mean squared displacement) after every tiling

    def one_round(sx, sy):
        moved = 0
        for ox, oy in [(0, 0), (0, h), (h, 0), (h, h)]:
            for cells in ref.windows(cx, cy, sx, sy, w, ox, oy):
                moved += ref.solve_window(P, C, cells, occ, pos, 1e-12)
            trace.append((sx, ((P - C[pos]) ** 2).sum(1).mean()))
        return moved

    for sx, sy in ref.schedule(G, G, w, ratio):
        one_round(sx, sy)
        snaps.append((f"stride {sx}", pos.copy()))
    while one_round(1, 1):
        pass
    snaps.append(("end", pos.copy()))
    assert np.array_equal(pos, ref.snap(P, np.ones((G, G), bool), w=w, ratio=ratio, seed=seed))
    fig = plt.figure(figsize=(WIDTH, 2.95))
    gs = fig.add_gridspec(2, len(snaps), height_ratios=[1, 1.55], left=0.11, right=0.99, top=0.9, bottom=0.12,
                          wspace=0.08, hspace=0.18)
    for k, (title, p) in enumerate(snaps):
        ax = fig.add_subplot(gs[0, k])
        img = np.ones((G, G, 3))
        img[cy[p], cx[p]] = colors
        ax.imshow(img, interpolation="nearest")
        cost = ((P - C[p]) ** 2).sum(1).mean()
        ax.set_title(f"{title}\n{cost:.2f}", fontsize=6.5, color=fs.INK2, pad=2)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    start_cost = ((P - C[snaps[0][1]]) ** 2).sum(1).mean()
    y = [start_cost] + [c for _, c in trace]
    ax.plot(range(len(y)), y, color=BLUE, lw=1.2, marker="o", ms=2.6, mec="white", mew=0.4)
    stride = [None] + [s for s, _ in trace]
    k = 1
    while k < len(stride):
        j = k
        while j < len(stride) and stride[j] == stride[k]:
            j += 1
        if (stride[k] - 1) % 2 == 0 or stride[k] == 1:
            ax.axvspan(k - 0.5, j - 0.5, color="#f3f2ee", lw=0, zorder=0)
        label = f"stride {stride[k]}" if stride[k] > 1 else "stride 1, rounds\nuntil none moves"
        ax.text((k + j - 1) / 2, 0.97, label, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6, color=fs.INK2)
        k = j
    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(y) - 0.5)
    ax.set_ylim(min(y) * 0.6, max(y) * 6)
    ax.set_xlabel("tilings solved (four per round)", fontsize=7)
    ax.set_ylabel("mean squared\ndisplacement", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_ref_snap")


# ------------------------------------------------------------------------------ LaTeX
def _ranges():
    """Line ranges of the five parts: header, schedule, windows, one window, snap."""
    tree = ast.parse(SOURCE.read_text())
    fn = {n.name: (n.lineno, n.end_lineno) for n in tree.body if isinstance(n, ast.FunctionDef)}
    imports = max(n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    return [(1, imports), (fn["strides"][0], fn["schedule"][1]), fn["windows"], fn["solve_window"], fn["snap"]]


INTRO = r"""This is the complete algorithm: the file \texttt{shufflesnap/reference.py} from the
package, split into its parts, each beside a diagram computed with its code. Without exact
ties it returns the same assignment as the compiled kernel; the package's tests check this
for every preset on six kinds of target, from an exact-fit square to surplus cells and a
random mask. The kernel adds threads, skips windows whose cells have not changed (which does
not change the result) and an optional time budget. Points are in grid units: cell $(x,y)$
has its center at $(x,y)$."""

PARTS = [
    ("fig_ref_schedule", "The schedule",
     r"""A window always holds $w\times w$ cells, but at stride $s$ they lie $s$ cells apart, so
it spans $(w-1)s+1$ cells per axis (dashed). The first stride is the smallest power of two
with $ws\ge L$, so the first windows reach across the axis; each later stride is the last one
divided by \texttt{ratio} and rounded. Each axis has its own strides, and the shorter axis
waits at 1. Drawn: one window of the default preset ($w=8$) on a $64\times64$ grid at every
stride."""),
    ("fig_ref_windows", "The windows of one tiling",
     r"""At stride $(s_x,s_y)$ the cells fall into $s_xs_y$ residue classes $(x \bmod s_x,\,
y \bmod s_y)$ (a). In its class, cell $x$ is lattice column $x /\!/ s_x$, and tiles of $w$
lattice columns start at column \texttt{ox}. A window is the set of cells that share a class
and a tile (b, boxes). The four tilings of a round shift the tiles by half a window (c). Edge
windows are smaller, and masked cells never appear. Drawn with $w=4$, stride 2, class
$(0,0)$."""),
    ("fig_ref_solve", "One window",
     r"""The window's points are reassigned among all of its cells, empty ones included, by an
exact linear assignment on squared distances. The result is kept only if it lowers the
window's cost by more than a relative \texttt{tol}, so the cost never rises and ties cannot
cycle. \texttt{pos} gives each point's cell and \texttt{occ} each cell's point. Drawn: seven
points in a $3\times3$ window before and after, with summed squared distances, and the cost
matrix with the chosen entries."""),
    ("fig_ref_snap", "The whole method",
     r"""The cells are the mask's usable entries in row-major order, and point $i$ starts in a
random cell \texttt{pos[i]}. A round solves the windows of the four tilings in turn. Each
coarse stage runs one round; then stride-1 rounds repeat until one moves nothing, a fixed
point of every window but not a proof of optimality. Drawn: 1{,}024 uniform points on a
$32\times32$ grid, colored by input position, after the start, each
coarse stage and the end (top), and the mean squared displacement in cells$^2$ after every
tiling (bottom): each stride halves it or better, and the last stride-1 round moves
nothing."""),
]


def _code(lines):
    a, b = lines
    return (rf"\lstinputlisting[firstline={a},lastline={b},firstnumber={a},numbers=left,"
            r"numberstyle=\tiny\color{black!45},numbersep=6pt,frame=none,xleftmargin=1.6em,"
            r"aboveskip=0pt,belowskip=0pt]{reference.py}")


def _row(left, lines):
    return rf"""\noindent\begin{{minipage}}[t]{{0.405\linewidth}}\vspace{{0pt}}\raggedright
{left}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.565\linewidth}}\vspace{{0pt}}
{_code(lines)}
\end{{minipage}}"""


def _part(fig, title, text):
    return rf"""\textbf{{{title}.}}\par\smallskip
\includegraphics[width=\linewidth]{{figures/{fig}}}\par\smallskip
{{\small {text}\par}}"""


def write_tex():
    header, *parts = _ranges()
    rows = [_row(_part(*p), lines) for p, lines in zip(PARTS, parts)]
    intro = _row(r"{\small " + INTRO + r"\par}", header)
    body = rf"""% Generated by experiments/fig_reference.py from paper/reference.py; do not edit.
\begin{{landscape}}
\section{{A serial reference implementation}}
\label{{app:reference}}

{intro}

\vspace{{1.4em}}
{rows[0]}

\clearpage
{rows[1]}

\vspace{{1.4em}}
{rows[2]}

\clearpage
{rows[3]}
\end{{landscape}}
"""
    TEX.write_text(body)
    print("wrote", TEX.relative_to(ROOT))


def main():
    fs.setup()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_schedule()
    fig_windows()
    fig_solve()
    fig_snap()
    write_tex()
    if os.environ.get("AGENT_FILES_URL"):
        from urllib.parse import quote
        print(f"{os.environ['AGENT_FILES_URL']}/#{quote(str(FIGS))}")


if __name__ == "__main__":
    main()
