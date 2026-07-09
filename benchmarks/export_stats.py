"""Export headline numbers from benchmark CSVs as LaTeX macros (paper/stats.tex).

Every number quoted in the manuscript comes from this file, so text and
figures cannot drift apart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
OUT = REPO / "paper" / "stats.tex"


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    macros: dict[str, str] = {}

    env = json.loads((RESULTS / "environment.json").read_text())
    macros["StatMachine"] = env.get("processor", "unknown CPU")
    macros["StatThreads"] = str(env.get("cpu_count", "?"))
    # report nominal installed memory (nearest common module size)
    measured_gib = float(env.get("memory_gib", 0))
    nominal = min([8, 16, 32, 64, 128, 256, 512], key=lambda x: abs(x - measured_gib))
    macros["StatMemory"] = str(nominal)

    conv = pd.read_csv(RESULTS / "convergence_traces.csv")
    finals = conv.sort_values("round").groupby(["method", "instance_seed"]).last()

    def method_final(method: str) -> float:
        return float(finals.loc[method]["gap_pct"].median())

    def method_time_to(method: str, threshold: float) -> float | None:
        times = []
        for seed, group in conv[conv["method"] == method].groupby("instance_seed"):
            hit = group[(group["round"] > 0) & (group["gap_pct"] <= threshold)]
            if hit.empty:
                return None
            times.append(float((hit["elapsed_s"] + hit["seed_time_s"]).min()))
        return float(np.median(times))

    macros["StatConvRandomFinal"] = fmt(method_final("random+multiscale"), 3)
    macros["StatConvLocalFinal"] = fmt(method_final("raster+local"), 2)
    macros["StatConvRasterFairyLocalFinal"] = fmt(method_final("rasterfairy+local"), 2)
    seed_rows = conv[conv["round"] == 0]
    rf_seed = seed_rows[seed_rows["method"] == "rasterfairy+local"]["gap_pct"].median()
    macros["StatConvRasterFairySeedGap"] = fmt(rf_seed, 0)
    t = method_time_to("random+multiscale", 0.1)
    macros["StatConvRandomTimeToTenthS"] = fmt(t, 2) if t else "--"

    # rounds to 0.1% for multiscale
    rounds_needed = []
    for seed, group in conv[conv["method"] == "random+multiscale"].groupby("instance_seed"):
        hit = group[(group["round"] > 0) & (group["gap_pct"] <= 0.1)]
        rounds_needed.append(int(hit["round"].min()) if not hit.empty else 999)
    macros["StatConvRoundsToTenth"] = str(int(np.median(rounds_needed)))

    dist = pd.read_csv(RESULTS / "distributions.csv")
    macros["StatDistWorstFinal"] = fmt(dist["final_gap_pct"].max(), 2)
    macros["StatDistMedianFinal"] = fmt(dist["final_gap_pct"].median(), 3)
    # uniform's near-zero optimum makes its relative seed gap degenerate
    nondegenerate = dist[dist["distribution"] != "uniform"]
    macros["StatDistMaxSeedGap"] = fmt(nondegenerate["seed_gap_pct"].max() / 100.0, 0)
    macros["StatDistMaxTime"] = fmt(dist["elapsed_s"].max(), 2)

    scal = pd.read_csv(RESULTS / "scaling.csv")
    row200 = scal[scal["side"] == 200].median(numeric_only=True)
    macros["StatFortyKConvergeS"] = fmt(row200["elapsed_s"], 1)
    macros["StatFortyKGap"] = fmt(row200["final_gap_pct"], 3)
    macros["StatFortyKOracleS"] = fmt(row200["oracle_time_s"], 0)
    macros["StatFortyKToTenthS"] = fmt(row200["time_to_0p1pct_s"], 2)
    row1024 = scal[scal["side"] == 1131].median(numeric_only=True)
    macros["StatMillionWithinTenthS"] = fmt(row1024["time_within_0p1pct_of_final_s"], 1)
    macros["StatMillionPerRoundS"] = fmt(row1024["per_round_s"], 2)
    largest = scal[scal["side"] == scal["side"].max()].median(numeric_only=True)
    macros["StatLargestSide"] = str(int(scal["side"].max()))
    macros["StatLargestN"] = f"{largest['n'] / 1e6:.1f} million"
    macros["StatLargestWithinTenthS"] = fmt(largest["time_within_0p1pct_of_final_s"], 1)
    macros["StatLargestConvergeMin"] = fmt(largest["elapsed_s"] / 60.0, 0)
    macros["StatLargestDenseGiB"] = f"{largest['dense_matrix_gib']/1024:.0f}"
    macros["StatLargestPeakRssGiB"] = fmt(scal["peak_rss_gib"].max(), 1)
    conv800 = scal[scal["side"] == 800].median(numeric_only=True)
    macros["StatSixHundredKConvergeS"] = fmt(conv800["elapsed_s"], 0)

    # gigascale stress run (bench_billion.py); thresholds are relative to the
    # run's own final cost at budget end, not a converged fixed point
    giga = json.loads((RESULTS / "billion_31623_seed0.json").read_text())
    macros["StatGigaSide"] = str(giga["side"])
    macros["StatGigaNBillion"] = f"{giga['n'] / 1e9:.2f}"
    macros["StatGigaRounds"] = str(giga["rounds"])
    macros["StatGigaBudgetH"] = fmt(giga["budget_s"] / 3600.0, 0)
    macros["StatGigaPerRoundMin"] = fmt(giga["per_round_s"] / 60.0, 1)
    macros["StatGigaSeedS"] = fmt(giga["seed_time_s"], 0)
    macros["StatGigaWithinOneMin"] = fmt(giga["time_within_1pct_of_final_s"] / 60.0, 0)
    macros["StatGigaWithinTenthMin"] = fmt(giga["time_within_0p1pct_of_final_s"] / 60.0, 0)
    macros["StatGigaWithinHundredthMin"] = fmt(giga["time_within_0p01pct_of_final_s"] / 60.0, 0)
    macros["StatGigaWithinThousandthH"] = fmt(giga["time_within_0p001pct_of_final_s"] / 3600.0, 1)
    macros["StatGigaPeakRssGiB"] = fmt(giga["peak_rss_gib"], 1)
    macros["StatGigaDenseMTiB"] = fmt(giga["dense_matrix_gib"] / (1024.0 * 1e6), 1)

    holes = pd.read_csv(RESULTS / "holes.csv")
    h85 = holes[holes["fill_ratio"] == 0.85].median(numeric_only=True)
    macros["StatHolesFreeGap"] = fmt(h85["free_gap_pct"], 2)
    macros["StatHolesPinnedGap"] = fmt(h85["pinned_gap_pct"], 0)

    window = pd.read_csv(RESULTS / "ablation_window.csv")
    w = window.groupby("window_size").median(numeric_only=True)
    macros["StatWindowBest"] = str(int(w["time_to_0p1pct_s"].idxmin()))

    threads = pd.read_csv(RESULTS / "ablation_threads.csv")
    t1 = threads[threads["threads"] == 1]["rounds_per_s"].iloc[0]
    tbest = threads["rounds_per_s"].max()
    macros["StatThreadSpeedup"] = fmt(tbest / t1, 1)

    oracle = pd.read_csv(RESULTS / "oracle_check.csv")
    macros["StatOracleMaxRelErr"] = f"{oracle[['jv_vs_scipy_rel','auction_vs_scipy_rel']].max().max():.1e}"

    mnist = json.loads((RESULTS / "mnist_summary.json").read_text())
    macros["StatMnistRounds"] = str(mnist["rounds"])
    macros["StatMnistElapsed"] = fmt(mnist["cleanup_elapsed_s"], 1)
    macros["StatMnistConverged"] = "converged" if mnist["converged"] else "did not converge"
    macros["StatMnistKnnSeed"] = fmt(100 * mnist["knn10_preservation_seed"], 1)
    macros["StatMnistKnnFinal"] = fmt(100 * mnist["knn10_preservation_final"], 1)
    macros["StatMnistKnnRasterFairy"] = fmt(100 * mnist["knn10_preservation_rasterfairy"], 1)
    macros["StatMnistLabelEmbed"] = fmt(100 * mnist["label10_agreement_embedding"], 1)
    macros["StatMnistLabelFinal"] = fmt(100 * mnist["label10_agreement_final"], 1)
    macros["StatMnistLabelRasterFairy"] = fmt(100 * mnist["label10_agreement_rasterfairy"], 1)
    macros["StatMnistCostDropPct"] = fmt(
        100 * (mnist["seed_cost"] - mnist["final_cost"]) / mnist["seed_cost"], 1)
    macros["StatMnistFinalCost"] = fmt(mnist["final_cost"], 1)
    macros["StatMnistRasterFairyCost"] = fmt(mnist["rasterfairy_cost"], 1)
    macros["StatMnistRasterFairyExcessPct"] = fmt(
        100 * (mnist["rasterfairy_cost"] - mnist["final_cost"]) / mnist["final_cost"], 0)

    bspot_q = pd.read_csv(RESULTS / "bspot_quality.csv")
    b16 = bspot_q[bspot_q["plans"] == 16]
    meander16 = b16[b16["distribution"] == "meander"]
    macros["StatBspotMeanderGap"] = fmt(meander16["gap_pct"].median(), 1)
    macros["StatBspotMeanderTime"] = fmt(meander16["time_s"].median(), 2)
    bspot_s = pd.read_csv(RESULTS / "bspot_scaling.csv")
    macros["StatBspotLargestTime"] = fmt(bspot_s["time_s"].max(), 1)
    bspot_seed = pd.read_csv(RESULTS / "bspot_seed.csv")
    macros["StatBspotSeedGap"] = fmt(bspot_seed["bspot_gap_pct"].median(), 1)
    macros["StatBspotSeedFinal"] = fmt(bspot_seed["final_gap_pct"].median(), 3)
    bspot_mnist = json.loads((RESULTS / "bspot_mnist.json").read_text())
    macros["StatBspotMnistCost"] = fmt(bspot_mnist["cost"], 1)
    macros["StatBspotMnistExcessPct"] = fmt(
        100 * (bspot_mnist["cost"] - mnist["final_cost"]) / mnist["final_cost"], 1)
    macros["StatBspotMnistTimeMs"] = fmt(1000 * bspot_mnist["time_s"], 0)
    macros["StatBspotMnistKnn"] = fmt(100 * bspot_mnist["knn10_preservation"], 1)
    macros["StatBspotMnistLabel"] = fmt(100 * bspot_mnist["label10_agreement"], 1)

    # per-family comparison table: BSP-OT (16 plans) vs converged descent at 40k
    dist_med = dist.groupby("distribution")[["final_gap_pct", "elapsed_s"]].median()
    b16_med = b16.groupby("distribution")[["gap_pct", "time_s"]].median()
    order = ["meander", "uniform", "two_blobs", "ring", "spiral", "mixture"]
    names = {"meander": "meander", "uniform": "uniform", "two_blobs": "two blobs",
             "ring": "ring", "spiral": "spiral", "mixture": "mixture"}
    table_rows = []
    for family in order:
        bg, bt = b16_med.loc[family, "gap_pct"], b16_med.loc[family, "time_s"]
        mg, mt = dist_med.loc[family, "final_gap_pct"], dist_med.loc[family, "elapsed_s"]
        table_rows.append(
            f"    {names[family]} & {bg:.2f} & {bt:.2f} & {mg:.3f} & {mt:.2f} \\\\"
        )
    body = "\n".join(table_rows)
    (REPO / "paper" / "bspot_table.tex").write_text(
        "% Auto-generated by benchmarks/export_stats.py -- do not edit.\n"
        "\\begin{tabular}{lrrrr}\n"
        "  \\toprule\n"
        "  & \\multicolumn{2}{c}{BSP-OT} & \\multicolumn{2}{c}{ours} \\\\\n"
        "  \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
        "  family & gap (\\%) & time (s) & gap (\\%) & time (s) \\\\\n"
        "  \\midrule\n"
        f"{body}\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n")

    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items())]
    OUT.write_text("% Auto-generated by benchmarks/export_stats.py -- do not edit.\n"
                   + "\n".join(lines) + "\n")
    print(f"wrote {OUT} with {len(macros)} macros and bspot_table.tex")


if __name__ == "__main__":
    main()
