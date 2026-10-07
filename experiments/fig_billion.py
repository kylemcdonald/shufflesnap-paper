"""Figure and numbers for the billion-point meander (results/billion/, experiments/billion.py).

Left: the final assignment drawn 75% of the way from each point to its cell
(experiments/render_billion.py, downsampled).  Right: cost after every phase, as excess
over the final endpoint, next to the single-plan BSP-OT port.  The earlier implementation's
6 h run is not drawn; its numbers are computed for the text (its per-round trace rescaled
to this frame: it priced both points and cells in [0.03, 0.97], a uniform scale of the grid
frame, so costs differ by one constant factor, recovered by pricing its saved assignment here).

Writes paper/figures/fig_billion.{pdf,png} and results/billion/billion_numbers.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results", "billion")
OLD = os.path.join(ROOT, "..", "..", "..", "shufflesnap-paper", "benchmarks", "results")
OUT = os.path.join(ROOT, "paper", "figures")
OLD_ASSIGNMENT = "billion_meander_31623_seed0.npy"
BSPOT_ASSIGNMENT = "billion_bspot_31623_seed0.npy"


# the 6 h run is three segments, each resumed from the previous one's saved assignment:
# balanced for 1 h (stopped at stride 16), balanced resumed at stride 16 until its fixed point,
# then the quality preset (12x12) from stride 4 until convergence or 6 h in total.
SEGMENTS = [("", "SS-balanced"), ("_6h", "SS-balanced"), ("_q", "SS-quality")]


def load():
    segs = []
    for tag, method in SEGMENTS:
        path = os.path.join(RES, f"billion_31623_seed0{tag}.json")
        if not os.path.exists(path):
            continue
        s = json.load(open(path))
        tr = list(csv.DictReader(open(os.path.join(RES, f"billion_trace_31623_seed0{tag}.csv"))))
        t = np.array([float(r["elapsed_s"]) for r in tr])  # already offset by the earlier segments
        c = np.array([float(r["cost"]) for r in tr])
        sx = np.array([int(r["sx"]) for r in tr])
        segs.append(dict(tag=tag, method=method, s=s, t=t, c=c, sx=sx))
    first = segs[0]["s"]
    old_s = json.load(open(os.path.join(OLD, "billion_31623_seed0.json")))
    bsp_s = json.load(open(os.path.join(OLD, "billion_bspot_31623_seed0.json")))
    k = first["compare_costs"][OLD_ASSIGNMENT] / old_s["final_cost"]  # old frame -> grid frame
    otr = list(csv.DictReader(open(os.path.join(OLD, "billion_traces_31623_seed0.csv"))))
    # the earlier trace starts after its first round; its start cost is the same permutation's
    ot = np.array([0.0] + [float(r["elapsed_s"]) for r in otr])
    oc = np.array([first["start_cost"]] + [float(r["cost"]) * k for r in otr])
    return segs, old_s, ot, oc, bsp_s, k


def main():
    fs.setup()
    segs, old_s, ot, oc, bsp_s, k = load()
    last = segs[-1]
    final = last["s"]["final_cost"]
    cmp = segs[0]["s"]["compare_costs"]
    old_final, bsp = cmp[OLD_ASSIGNMENT], cmp[BSPOT_ASSIGNMENT]
    t = np.concatenate([g["t"] for g in segs])
    c = np.concatenate([g["c"] for g in segs])

    fig = plt.figure(figsize=(6.5, 3.1))
    ax_img = fig.add_axes([0.0, 0.0, 0.46, 1.0])
    render = os.path.join(RES, f"billion_final_interp75_10000.png")
    if not os.path.exists(render):
        render = os.path.join(RES, "billion_shufflesnap_interp75_10000.png")
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(render).resize((2400, 2400), Image.Resampling.BOX)
    img.save(os.path.join(OUT, "billion_interp75.png"))
    ax_img.imshow(np.asarray(img), interpolation="none")
    ax_img.set_axis_off()

    ax = fig.add_axes([0.575, 0.16, 0.41, 0.78])
    fs.recess(ax, "both")
    labels = {"SS-balanced": "ShuffleSnap default (8×8)", "SS-quality": "then quality (12×12)"}
    seen = set()
    prev = None
    for g in segs:
        tt, cc = g["t"], g["c"]
        if prev is not None:  # join the segments into one continuous curve
            tt, cc = np.concatenate([[prev[0]], tt]), np.concatenate([[prev[1]], cc])
        ex = 100 * (cc / final - 1)
        m = ex > 0
        lab = labels[g["method"]] if g["method"] not in seen else None
        seen.add(g["method"])
        ax.step(tt[m] / 60, ex[m], where="post", color=fs.COLORS[g["method"]], lw=1.4, label=lab)
        prev = (g["t"][-1], g["c"][-1])
    ax.plot([bsp_s["solve_time_s"] / 60], [100 * (bsp / final - 1)], ls="none", marker=fs.MARKERS["BSP-OT"],
            color=fs.COLORS["BSP-OT"], ms=5, label="BSP-OT, 1 plan, 1 thread")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 400)
    ax.set_ylim(3e-8, 1e3)
    ax.set_xticks([1, 10, 60, 360])
    ax.set_xticklabels(["1 min", "10 min", "1 h", "6 h"])
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1, 100])
    ax.set_yticklabels(["$10^{-6}$%", "$10^{-4}$%", "0.01%", "1%", "100%"])
    ax.set_xlabel("elapsed time (24 threads)")
    ax.set_ylabel(f"excess over the final cost ({t[-1] / 3600:.1f} h)")
    ax.legend(loc="lower left", fontsize=6.5, handlelength=1.6)
    ax.minorticks_off()
    fig.savefig(os.path.join(OUT, "fig_billion.pdf"), dpi=300)
    fig.savefig(os.path.join(OUT, "fig_billion.png"), dpi=150)

    # numbers for the text
    def excess(x):
        return 100 * (x / final - 1)

    def first_at(level):
        hit = np.nonzero(c <= level)[0]
        return t[hit[0]] / 60 if len(hit) else None

    old_at_1h = oc[ot <= 3600][-1]
    bal = [g for g in segs if g["method"] == "SS-balanced"]
    bal_end = bal[-1]
    nums = {
        "peak_gib": max(g["s"]["peak_rss_gib"] for g in segs),
        "cost_1h_excess_pct": excess(segs[0]["s"]["final_cost"]),
        "balanced_converged": bal_end["s"]["finished"],
        "balanced_end_min": bal_end["t"][-1] / 60,
        "balanced_end_excess_pct": excess(bal_end["c"][-1]),
        "stride1_reached_min": bal_end["s"]["time_to_stride1_s"] / 60 if bal_end["s"]["time_to_stride1_s"] else None,
        "balanced_stride1_rounds": bal_end["s"]["stride1_rounds"],
        "final_min": t[-1] / 60, "final_converged": last["s"]["finished"],
        "within_1pct_min": first_at(final * 1.01), "within_0p1pct_min": first_at(final * 1.001),
        "within_0p01pct_min": first_at(final * 1.0001), "within_0p001pct_min": first_at(final * 1.00001),
        "old_6h_excess_pct": excess(old_final), "old_1h_excess_pct": excess(old_at_1h),
        "passed_old_6h_min": first_at(old_final), "passed_old_1h_min": first_at(old_at_1h),
        "bspot_excess_pct": excess(bsp), "bspot_s": bsp_s["solve_time_s"],
        "mean_sq_cells2": final / last["s"]["n"], "rms_cells": (final / last["s"]["n"]) ** 0.5,
        "frame_scale": k,
    }
    with open(os.path.join(RES, "billion_numbers.json"), "w") as f:
        json.dump(nums, f, indent=1)
    for key, v in nums.items():
        print(key, v)


if __name__ == "__main__":
    main()
