"""Compact figures for the shorter editions (versions/long, medium, short).

    python experiments/fig_editions.py [schematic|progress|summary|gallery|failures|masks|short|all]

They reuse the benchmark data and the cached assignments of the full paper's figures;
nothing here reruns a benchmark.  All are drawn at 7.0 in wide and included at text width,
like every figure of the full paper, so their type matches it.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

import shufflesnap as ss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets  # noqa: E402
import figstyle as fs  # noqa: E402
import specs  # noqa: E402
import make_figures as mf  # noqa: E402
import reference  # noqa: E402

OUT = os.path.join(HERE, "..", "paper", "figures")
BLUE = "#2a78d6"
BLUE_LIGHT = "#b7d3f6"
ORANGE = "#eb6834"
GRAY = "#cfcec7"


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
    plt.close(fig)
    print("wrote", name)


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(fs.AXIS)
        sp.set_linewidth(0.5)


# ------------------------------------------------------------------------------ schematic
def _panel_solve(ax_before, ax_after):
    """Nine points and a 3x3 window: a poor assignment, then the exact reassignment."""
    rng = np.random.default_rng(4)
    cells = np.array([(x, y) for y in range(3) for x in range(3)], dtype=float)
    pts = cells[rng.permutation(9)] + rng.uniform(-0.42, 0.42, size=(9, 2))
    before = rng.permutation(9)
    cost = ((pts[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    after = linear_sum_assignment(cost)[1]
    cols = fs.position_colors(pts, L=58.0, chroma=62.0)
    for ax, assign, title in ((ax_before, before, "before"), (ax_after, after, "after")):
        for k in range(4):
            ax.plot([-0.5, 2.5], [k - 0.5, k - 0.5], color=fs.AXIS, lw=0.6, zorder=0)
            ax.plot([k - 0.5, k - 0.5], [-0.5, 2.5], color=fs.AXIS, lw=0.6, zorder=0)
        for i in range(9):
            c = cells[assign[i]]
            ax.plot([pts[i, 0], c[0]], [pts[i, 1], c[1]], color=cols[i], lw=1.1, zorder=1)
            ax.plot(*c, marker="s", ms=4.2, color=cols[i], mec="white", mew=0.4, zorder=2)
            ax.plot(*pts[i], marker="o", ms=3.2, color=cols[i], mec=fs.INK, mew=0.3, zorder=3)
        tot = cost[np.arange(9), assign].sum()
        ax.set_title(f"{title}: cost {tot:.1f}", fontsize=7, color=fs.INK2)
        ax.set_xlim(-0.8, 2.8)
        ax.set_ylim(2.8, -0.8)
        ax.set_aspect("equal")
        ax.axis("off")


def _panel_stride(ax, s, title, w=6):
    W = max(22, 4 * (w - 1) + 3)  # the lattice holds the whole stride-4 window
    xs, ys = np.meshgrid(np.arange(W), np.arange(W))
    ax.scatter(xs.ravel(), ys.ravel(), s=0.6 * (22 / W) ** 2, c=GRAY, linewidths=0)
    wx = s * np.arange(w) + (0 if s > 1 else (W - w) // 2)
    WX, WY = np.meshgrid(wx, wx)
    ax.scatter(WX.ravel(), WY.ravel(), s=(5 if s > 1 else 4) * (36 / w ** 2) ** 0.5, marker="s", c=BLUE,
               linewidths=0)
    ax.add_patch(Rectangle((wx[0] - 0.6, wx[0] - 0.6), wx[-1] - wx[0] + 1.2, wx[-1] - wx[0] + 1.2, fill=False,
                           ec=fs.INK, lw=0.6))
    ax.set_title(title, fontsize=7, color=fs.INK2)
    ax.set_xlim(-1, W)
    ax.set_ylim(W, -1)
    ax.set_aspect("equal")
    ax.axis("off")


def _panel_tilings(ax, w=6):
    """One subgrid: the unshifted tiling (solid) and the tiling shifted by half a window on
    both axes (dashed); windows are clipped at every edge.  Each window border is drawn as
    one line across the subgrid, so shared edges are never drawn twice (which would fill in
    the dashes)."""
    Wa, Hb = 2 * w + 1, 2 * w - 1
    h = w // 2
    xs, ys = np.meshgrid(np.arange(Wa), np.arange(Hb))
    ax.scatter(xs.ravel(), ys.ravel(), s=0.9 if w == 6 else 0.6, c=fs.INK2, linewidths=0, zorder=3)
    solid = dict(color=fs.INK, lw=0.9, ls="-")
    dashed = dict(color=ORANGE, lw=1.0, ls=(0, (3, 2)))
    for off, style in ((0, solid), (h, dashed)):
        for x in range(off, Wa, w):
            if 0 < x < Wa:
                ax.plot([x - 0.5, x - 0.5], [-0.5, Hb - 0.5], zorder=2, **style)
        for y in range(off, Hb, w):
            if 0 < y < Hb:
                ax.plot([-0.5, Wa - 0.5], [y - 0.5, y - 0.5], zorder=2, **style)
    # the subgrid's edge clips the windows of both tilings
    ax.add_patch(Rectangle((-0.5, -0.5), Wa, Hb, fill=False, ec=fs.INK, lw=0.9, zorder=2))
    ax.set_xlim(-1, Wa)
    ax.set_ylim(Hb, -1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([], [], label="shift (0, 0)", **solid)
    ax.plot([], [], label=f"shift ({h}, {h})", **dashed)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=6.3, handlelength=1.8,
              columnspacing=1.0)


def fig_schematic(w=specs.DEFAULT["window"], name="fig_schematic"):
    fig = plt.figure(figsize=(7.0, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.05, 2.05, 1.55], wspace=0.18, left=0.01, right=0.99, top=0.84,
                          bottom=0.07)
    ga = gs[0].subgridspec(1, 2, wspace=0.12)
    axa = fig.add_subplot(ga[0])
    _panel_solve(axa, fig.add_subplot(ga[1]))
    gb = gs[1].subgridspec(1, 2, wspace=0.12)
    axb = fig.add_subplot(gb[0])
    # coarse to fine: the stride-4 window first, then stride 1
    _panel_stride(axb, 4, f"stride 4: {w * w} cells\nspan {4 * (w - 1) + 1}×{4 * (w - 1) + 1}", w)
    _panel_stride(fig.add_subplot(gb[1]), 1, f"stride 1: the same\nnumber, {w}×{w}", w)
    axc = fig.add_subplot(gs[2])
    _panel_tilings(axc, w)
    y = 0.975
    for ax, text in ((axa, "(a) one exact window solve"), (axb, "(b) same size, longer reach"),
                     (axc, "(c) shifted tilings")):
        fig.text(ax.get_position().x0, y, text, fontsize=8.5, va="top")
    _save(fig, name)


# ------------------------------------------------------------------------------ progress
def fig_progress(preset="balanced", out_name="fig_progress", every=2):
    n = 10000
    Pn, g, _, _ = datasets.instance("mnist", n, 0)
    cols = fs.source_colors(Pn)
    res = ss.assign(Pn, g, normalize="none", preset=preset, seed=1, snapshots=True)
    opt = reference.get("mnist", n, 0, compute=False)
    panels = [("input", None, None)]
    for name, cell in res.snapshots:
        label = "random start" if name == "start" else "stride " + name.split("s=")[1].split("x")[0]
        panels.append((label, cell, ss.assignment_cost(Pn, g, cell) / n))
    panels[-1] = ("converged", panels[-1][1], panels[-1][2])
    if every > 1:  # keep the start, every `every`-th coarse level and the converged layout
        mid = panels[2:-1]
        panels = panels[:2] + mid[::every] + panels[-1:]
    if opt is not None:
        panels.append(("optimum", np.asarray(opt["cell"]), opt["cost"] / n))
    k = len(panels)
    fig, axes = plt.subplots(1, k, figsize=(7.0, 1.28), gridspec_kw=dict(wspace=0.06, left=0.005, right=0.995,
                                                                          top=0.8, bottom=0.12))
    for ax, (label, cell, c) in zip(axes, panels):
        if cell is None:
            ax.scatter(Pn[:, 0], Pn[:, 1], s=0.12, c=cols, linewidths=0, rasterized=True)
            ax.set_xlim(-1, g.width)
            ax.set_ylim(g.height, -1)
            ax.set_aspect("equal")
        else:
            ax.imshow(fs.grid_image(g, cell, cols), interpolation="nearest")
            ax.set_xlabel(f"{c:.1f}", fontsize=6.8, labelpad=1.5)
        ax.set_title(label, fontsize=6.8, pad=2.5)
        _clean(ax)
    _save(fig, out_name)
    return [(p[0], p[2]) for p in panels]


# ------------------------------------------------------------------------------ summary
def fig_summary(name="fig_summary"):
    import analysis

    df = analysis.load()
    main = mf._ok(df[(df["n"] == 100000) & (df["grid"] == "square") & (df["threads"] == 1)
                     & df["file"].str.startswith("main")])
    real = mf._ok(df[(df["threads"] == 1) & df["file"].str.startswith("real")])
    keep = ["SS-fast", "SS-balanced", "SS-quality", "BSP-OT-16", "BSP-OT-64", "RasterFairy-master", "Exact-certified"]
    panels = [("uniform (100k)", main[main["dist"] == "uniform"]),
              ("Gaussian mixture (100k)", main[main["dist"] == "gmm"]),
              ("MNIST UMAP (70k)", real[real["dist"] == "mnist"]),
              ("spiral (100k)", main[main["dist"] == "spiral"])]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.35), layout="constrained")
    for ax, (title, sub) in zip(axes, panels):
        mf._family_points(ax, sub[sub["label"].isin(keep)])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=7.5)
        for axis, get in ((ax.xaxis, ax.get_xlim), (ax.yaxis, ax.get_ylim)):
            lo, hi = get()
            if hi / lo < 30:  # an axis can span less than a decade: label 1-2-5 ticks as plain numbers
                axis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
                axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("time (s, one thread)")
        fs.recess(ax, grid_axis="both")
    axes[0].set_ylabel("excess over the optimum (%)")
    handles = []
    entries = [("SS-balanced", "ShuffleSnap fast, balanced (default), quality"),
               ("BSP-OT", "BSP-OT (16, 64 plans)"), ("RasterFairy-master", "RasterFairy")]
    for lab, label in entries:
        filled = fs.marker(lab) not in ("x", "+")
        color = fs.color(lab)
        handles.append(plt.Line2D([], [], ls="-" if lab in ("SS-balanced", "BSP-OT") else "", marker=fs.marker(lab),
                                  color=color, ms=4.5, lw=0.8, mec="white" if filled else None,
                                  mew=0.5 if filled else 1.2, label=label))
    handles.append(plt.Line2D([], [], color=fs.INK, lw=0.8, label="certified exact solver (time)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=6.3, frameon=False, handlelength=1.6,
               columnspacing=1.2)
    _save(fig, name)


# ------------------------------------------------------------------------------ gallery
def fig_gallery():
    """MNIST, all 70,000 points: whole grid, zoom, and lattice precision (three rows)."""
    import fig_visual as fv

    Pn, g, _, cells = fv.assignments("mnist")
    n = len(Pn)
    cols = fs.source_colors(Pn)
    cx, cy, half = int(g.width * 0.52), int(g.height * 0.45), 32
    labs = [lab for lab, _ in fv.METHODS]
    fig, axes = plt.subplots(3, len(labs), figsize=(7.0, 4.55), layout="constrained")
    for c, lab in enumerate(labs):
        cell = cells[lab]
        img = fs.grid_image(g, cell, cols)
        ax = axes[0, c]
        ax.imshow(img, interpolation="nearest")
        ax.add_patch(Rectangle((cx - half - 0.5, cy - half - 0.5), 2 * half, 2 * half, fill=False, ec="white",
                               lw=0.7))
        ax.set_title(f"{fv.SHORT[lab]}\n{ss.assignment_cost(Pn, g, cell) / n:.1f}", fontsize=7)
        axes[1, c].imshow(img[cy - half:cy + half, cx - half:cx + half], interpolation="nearest")
        nb = fv.neighbourhood_map(Pn, g, cell)
        nimg = np.full((g.height, g.width), np.nan)
        lat = g.lattice[cell]
        nimg[lat[:, 1], lat[:, 0]] = nb
        cm = matplotlib.colormaps["Oranges"].copy()
        cm.set_bad("white")
        im = axes[2, c].imshow(nimg, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
        axes[2, c].set_xlabel(f"mean {nb.mean():.2f}", fontsize=6.5)
    for ax in axes.ravel():
        _clean(ax)
    axes[0, 0].set_ylabel("whole grid", fontsize=7)
    axes[1, 0].set_ylabel("zoom (64×64 cells)", fontsize=7)
    axes[2, 0].set_ylabel("lattice precision", fontsize=7)
    cb = fig.colorbar(im, ax=axes[2, :], shrink=0.85, pad=0.01)
    cb.set_label("fraction", fontsize=6.5)
    cb.outline.set_visible(False)
    _save(fig, "fig_gallery_mnist")


# ------------------------------------------------------------------------------ failures
def fig_failures_row():
    """One row: loops between a ShuffleSnap layout and the optimum, their share of the
    excess, and the S-mask stripe of a coarse 6x6 halving schedule against the default preset."""
    import fig_failures as ff
    from shufflesnap import exact

    import methods

    fig = plt.figure(figsize=(7.0, 2.45), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.35, 0.8, 0.8])
    n = 10000
    Pn, g, _, _ = datasets.instance("uniform", n, 0)
    opt = reference.get("uniform", n, 0)["cell"]
    cell, _ = methods.m_shufflesnap(Pn, g, seed=0, **specs.DEFAULT)
    lens, exc, cyc, occ_opt = ff.excess_by_cycle(Pn, g, opt, cell)
    ax = fig.add_subplot(gs[0])
    Qo, Qs = g.centers[opt], g.centers[cell]
    diff = cell != opt
    big = np.zeros(n, bool)
    for c, L in zip(cyc, lens):
        if L > 30:
            big[occ_opt[c]] = True
    for mask_, col, wdt in ((diff & ~big, fs.MUTED, 0.003), (big, BLUE, 0.0034)):
        ax.quiver(Qo[mask_, 0], Qo[mask_, 1], (Qs - Qo)[mask_, 0], (Qs - Qo)[mask_, 1], angles="xy",
                  scale_units="xy", scale=1, width=wdt, color=col)
    ax.set_xlim(-1, g.width)
    ax.set_ylim(g.height, -1)
    ax.set_aspect("equal")
    _clean(ax)
    ax.set_title("(a) uniform, 10k points:\noptimal → ShuffleSnap cell", loc="left", fontsize=7.5)

    ax = fig.add_subplot(gs[1])
    rows = []
    for nn, lab in ((10000, "10k"), (100000, "100k")):
        P2, g2, _, _ = datasets.instance("uniform", nn, 0)
        o2 = reference.get("uniform", nn, 0)["cell"]
        c2, _ = methods.m_shufflesnap(P2, g2, seed=0, **specs.DEFAULT)
        L2, E2, _, _ = ff.excess_by_cycle(P2, g2, o2, c2)
        bins = [(2, 10), (11, 30), (31, 100), (101, 300), (301, 10 ** 9)]
        rows.append((lab, [E2[(L2 >= a) & (L2 <= b)].sum() / max(E2.sum(), 1e-12) for a, b in bins], len(L2)))
    x = np.arange(5)
    for k, (lab, share, ncyc) in enumerate(rows):
        ax.bar(x + (k - 0.5) * 0.36, share, width=0.34, color=["#86b6ef", "#1c5cab"][k],
               label=f"{lab} points ({ncyc} loops)")
    ax.set_xticks(x, ["2–10", "11–30", "31–100", "101–300", ">300"], fontsize=6.3)
    ax.set_xlabel("loop length (points)")
    ax.set_ylabel("share of the excess")
    ax.set_ylim(0, 1)
    fs.recess(ax)
    ax.legend(loc="upper left", fontsize=6.2)
    ax.set_title("(b) the excess sits in long loops", loc="left", fontsize=7.5)

    n3 = 4000
    P3, g3 = ff.s_inside(n3)
    o3 = exact.solve_certified(P3, g3, ss.assign(P3, g3, normalize="none").cell)["cell"]
    base = o3
    for k, (name, cfg) in enumerate((("6×6, halving", dict(window=6, ratio=2.0)), ("default", specs.DEFAULT))):
        c3, _ = methods.m_shufflesnap(P3, g3, seed=0, **cfg)
        d = np.sqrt(((P3 - g3.centers[c3]) ** 2).sum(1))
        ax = fig.add_subplot(gs[2 + k])
        img = np.full((g3.height, g3.width), np.nan)
        lat = g3.lattice[c3]
        img[lat[:, 1], lat[:, 0]] = d
        cmap = matplotlib.colormaps["Blues"].copy()
        cmap.set_bad("white")
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=25, interpolation="nearest")
        _clean(ax)
        msd, msd_opt = ss.assignment_cost(P3, g3, c3) / n3, ss.assignment_cost(P3, g3, base) / n3
        ax.set_title(f"({'cd'[k]}) S mask, {name}:\n{msd:.2f} (optimum {msd_opt:.2f})", loc="left", fontsize=7.5)
    cb = fig.colorbar(im, ax=fig.axes[-2:], shrink=0.8, pad=0.02)
    cb.set_label("displacement (cells)", fontsize=6.5)
    cb.outline.set_visible(False)
    _save(fig, "fig_failures_row")


# ------------------------------------------------------------------------------ masks
def fig_masks_row():
    """S mask and 10% surplus cells, five methods (cached assignments of fig_masks)."""
    import fig_masks as fm

    targets = [("srect", "S mask"), ("free10", "10% surplus cells")]
    fig, axes = plt.subplots(len(targets), len(fm.ROWS), figsize=(7.0, 3.55), layout="constrained",
                             gridspec_kw=dict(height_ratios=[1.3, 1.0]))
    for r, (grid, gname) in enumerate(targets):
        Pn, g, _, _ = datasets.instance("mnist", 20000, 1, grid)
        cols = fs.source_colors(Pn)
        opt = None
        for c, (lab, spec, name) in enumerate(fm.ROWS):
            cell = fm.get_cell("mnist", 20000, 1, grid, lab, spec, Pn, g)
            cost = ss.assignment_cost(Pn, g, cell)
            if lab == "Exact-certified":
                opt = cost
            ax = axes[r, c]
            ax.imshow(fs.grid_image(g, cell, cols), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            tag = "optimum" if lab == "Exact-certified" else f"+{100 * (cost / opt - 1):.2f}%"
            ax.set_xlabel(tag, fontsize=6.8, labelpad=1.5)
            if r == 0:
                ax.set_title(name.replace("ShuffleSnap", "ShuffleSnap\n").replace("(64 plans)", "\n(64 plans)")
                             .replace("(master)", "\n(master)").replace("exact optimum", "exact\noptimum"),
                             fontsize=7)
            if c == 0:
                ax.set_ylabel(gname, fontsize=7)
    _save(fig, "fig_masks_row")


def main(which="all"):
    fs.setup()
    # every edition presents the default preset (specs.DEFAULT)
    todo = dict(schematic=fig_schematic, progress=fig_progress, summary=fig_summary, gallery=fig_gallery,
                failures=fig_failures_row, masks=fig_masks_row)
    for k, f in todo.items():
        if which in (k, "all"):
            out = f()
            if out is not None:
                print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
