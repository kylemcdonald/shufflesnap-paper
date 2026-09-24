"""Figure: anatomy of ShuffleSnap's excess over the certified optimum.

(a) uniform points: the permutation between the ShuffleSnap and optimal layouts is a
    union of closed loops of one-cell moves; arrows show (optimal cell -> ShuffleSnap cell).
(b) excess share by loop length (100k uniform, baseline).
(c) 'S' mask with jittered-lattice input: baseline vs balanced vs optimum displacement.
Development instances (seed 0 / 3) are used here; they illustrate mechanisms, not claims.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shufflesnap as ss
from shufflesnap import exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets  # noqa: E402
import figstyle as fs  # noqa: E402
import methods  # noqa: E402
import reference  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")


def cycles_between(opt, cell, M):
    nxt = np.full(M, -1)
    nxt[opt] = cell
    seen = np.zeros(M, bool)
    out = []
    for c in range(M):
        if seen[c] or nxt[c] < 0 or nxt[c] == c:
            seen[c] = True
            continue
        cyc = []
        x = c
        while not seen[x]:
            seen[x] = True
            cyc.append(x)
            x = nxt[x]
        out.append(np.array(cyc))
    return out


def excess_by_cycle(Pn, g, opt, cell):
    M = g.n_cells
    occ_opt = np.full(M, -1)
    occ_opt[opt] = np.arange(len(opt))
    cyc = cycles_between(opt, cell, M)
    d_ss = ((Pn - g.centers[cell]) ** 2).sum(1)
    d_op = ((Pn - g.centers[opt]) ** 2).sum(1)
    lens, exc = [], []
    for c in cyc:
        pts = occ_opt[c]
        lens.append(len(c))
        exc.append((d_ss[pts] - d_op[pts]).sum())
    return np.array(lens), np.array(exc), cyc, occ_opt


def main():
    fs.setup()
    fig = plt.figure(figsize=(7.0, 6.2), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], width_ratios=[1, 1, 1])

    # (a) loops on uniform 10k
    n = 10000
    Pn, g, _, _ = datasets.instance("uniform", n, 0)
    opt = reference.get("uniform", n, 0)["cell"]
    cell, _ = methods.m_shufflesnap(Pn, g, window=6, ratio=2.0, seed=0)
    lens, exc, cyc, occ_opt = excess_by_cycle(Pn, g, opt, cell)
    ax = fig.add_subplot(gs[0, 0])
    Qo, Qs = g.centers[opt], g.centers[cell]
    diff = cell != opt
    big = np.zeros(n, bool)
    for c, L in zip(cyc, lens):
        if L > 30:
            big[occ_opt[c]] = True
    ax.quiver(Qo[diff & ~big, 0], Qo[diff & ~big, 1], (Qs - Qo)[diff & ~big, 0], (Qs - Qo)[diff & ~big, 1],
              angles="xy", scale_units="xy", scale=1, width=0.0025, color=fs.MUTED)
    ax.quiver(Qo[big, 0], Qo[big, 1], (Qs - Qo)[big, 0], (Qs - Qo)[big, 1], angles="xy", scale_units="xy", scale=1,
              width=0.0028, color="#2a78d6")
    ax.set_xlim(-1, g.width)
    ax.set_ylim(g.height, -1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(fs.AXIS)
    c_ss = ss.assignment_cost(Pn, g, cell)
    c_op = ss.assignment_cost(Pn, g, opt)
    ax.set_title(f"(a) uniform, N = 10k, baseline:\noptimal cell → ShuffleSnap cell", loc="left")
    print("(a)", diff.sum(), "points differ; excess %.3f%%" % (100 * (c_ss / c_op - 1)))

    # (b) excess share by loop length on 100k
    ax = fig.add_subplot(gs[0, 1:])
    rows = []
    for nn, lab in ((10000, "N = 10k"), (100000, "N = 100k")):
        P2, g2, _, _ = datasets.instance("uniform", nn, 0)
        o2 = reference.get("uniform", nn, 0)["cell"]
        for name, cfg in (("baseline", dict(window=6, ratio=2.0)), ("balanced", dict(window=8, ratio=2 ** 0.5))):
            c2, _ = methods.m_shufflesnap(P2, g2, seed=0, **cfg)
            L2, E2, _, _ = excess_by_cycle(P2, g2, o2, c2)
            bins = [(2, 10), (11, 30), (31, 100), (101, 300), (301, 10 ** 9)]
            share = [E2[(L2 >= a) & (L2 <= b)].sum() / max(E2.sum(), 1e-12) for a, b in bins]
            rows.append((f"{lab}, {name}", share, len(L2)))
    labels = ["2–10", "11–30", "31–100", "101–300", ">300"]
    x = np.arange(len(labels))
    width = 0.2
    cols = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    for k, (name, share, ncyc) in enumerate(rows):
        ax.bar(x + (k - 1.5) * width, share, width=width * 0.9, color=cols[k], label=f"{name} ({ncyc} loops)")
    ax.set_xticks(x, labels)
    ax.set_xlabel("loop length (points)")
    ax.set_ylabel("share of excess")
    ax.set_ylim(0, 1)
    fs.recess(ax)
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title("(b) share of the excess by loop length (uniform)", loc="left")

    # (c) S-mask with jittered lattice
    n = 4000
    P3, g3, _, _ = datasets.instance("jitter", n, 3, "logo")
    o3 = exact.solve_certified(P3, g3, ss.assign(P3, g3, normalize="none").cell)["cell"]
    vmax = 25
    for k, (name, cfg) in enumerate((("baseline", dict(window=6, ratio=2.0)),
                                     ("balanced", dict(window=8, ratio=2 ** 0.5)), ("optimum", None))):
        if cfg is None:
            c3 = o3
        else:
            c3, _ = methods.m_shufflesnap(P3, g3, seed=0, **cfg)
        d = np.sqrt(((P3 - g3.centers[c3]) ** 2).sum(1))
        ax = fig.add_subplot(gs[1, k])
        img = np.full((g3.height, g3.width), np.nan)
        lat = g3.lattice[c3]
        img[lat[:, 1], lat[:, 0]] = d
        cmap = matplotlib.colormaps["Blues"].copy()
        cmap.set_bad("white")
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(fs.AXIS)
        cst = ss.assignment_cost(P3, g3, c3) / n
        ax.set_title(f"({'cde'[k]}) {name}: {cst:.2f}", loc="left")
    cb = fig.colorbar(im, ax=fig.axes[-3:], shrink=0.75, pad=0.01)
    cb.set_label("displacement (cells)")
    cb.outline.set_visible(False)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_failures.pdf"))
    fig.savefig(os.path.join(OUT, "fig_failures.png"), dpi=150)
    print("wrote fig_failures")


if __name__ == "__main__":
    main()
