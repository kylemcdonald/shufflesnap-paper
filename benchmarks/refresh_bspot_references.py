"""Synchronize ShuffleSnap reference costs in the giant-scale BSP-OT result.

The BSP-OT billion-point run is a fixed reference, but its reported excess
depends on the ShuffleSnap initializer.  After rerunning the random-default
ShuffleSnap benchmarks, run this script before ``export_stats.py``:

  python benchmarks/bench_billion.py --side 31623 --seed 0 --budget 21600
  python benchmarks/bench_billion.py --side 10000 --seed 0 --budget 500 \
      --no-save-assignment
  python benchmarks/refresh_bspot_references.py

It preserves the BSP-OT measurements and replaces only their ShuffleSnap
baselines, recording enough provenance to prevent another mixed-seed update.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
DEFAULT_BSPOT = RESULTS / "billion_bspot_31623_seed0.json"
DEFAULT_GIGA = RESULTS / "billion_31623_seed0.json"
DEFAULT_HUNDRED_M = RESULTS / "billion_10000_seed0.json"


def load_summary(path: Path, *, side: int, budget_s: float) -> dict:
    summary = json.loads(path.read_text())
    if summary.get("side") != side or summary.get("n") != side * side:
        raise ValueError(f"{path} is not the expected {side}x{side} run")
    if summary.get("instance_seed") != 0:
        raise ValueError(f"{path} does not use instance seed 0")
    if not math.isclose(float(summary.get("budget_s", -1)), budget_s):
        raise ValueError(f"{path} does not use a {budget_s:g}-second budget")
    if not math.isfinite(float(summary.get("final_cost", float("nan")))):
        raise ValueError(f"{path} has no finite final cost")
    return summary


def provenance(summary: dict, source: Path) -> dict:
    return {
        "source": source.name,
        "initializer": "deterministic random permutation",
        "budget_s": float(summary["budget_s"]),
        "elapsed_s": float(summary["elapsed_s"]),
        "rounds": int(summary["rounds"]),
        "converged": bool(summary["converged"]),
        "final_cost": float(summary["final_cost"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bspot", type=Path, default=DEFAULT_BSPOT)
    parser.add_argument("--giga", type=Path, default=DEFAULT_GIGA)
    parser.add_argument("--hundred-m", type=Path, default=DEFAULT_HUNDRED_M)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    giga = load_summary(args.giga, side=31623, budget_s=21600)
    hundred_m = load_summary(args.hundred_m, side=10000, budget_s=500)
    bspot = json.loads(args.bspot.read_text())
    if bspot.get("side") != giga["side"] or bspot.get("n") != giga["n"]:
        raise ValueError(f"{args.bspot} does not match the billion-point run")
    ref = bspot.get("ref_1e8")
    if not isinstance(ref, dict) or ref.get("side") != hundred_m["side"]:
        raise ValueError(f"{args.bspot} has no matching 100M reference")

    # Archived summaries predate the explicit coordinate field. Never mix their
    # inset costs with a fresh unit-square run.
    if any(summary.get("coordinate_system", "legacy-inset-0.03") !=
           comparator.get("coordinate_system", "legacy-inset-0.03")
           for summary, comparator in ((giga, bspot), (hundred_m, ref))):
        raise ValueError("Coordinate conventions differ; rerun the comparator before refreshing costs")

    giga_final = float(giga["final_cost"])
    bspot_cost = float(bspot["assignment_cost"])
    hundred_m_final = float(hundred_m["final_cost"])
    bspot["shufflesnap_final_cost_6h"] = giga_final
    bspot["excess_over_shufflesnap_pct"] = 100.0 * (bspot_cost / giga_final - 1.0)
    bspot["shufflesnap_6h_reference"] = provenance(giga, args.giga)
    ref["shufflesnap_final_cost_500s_run"] = hundred_m_final
    ref["shufflesnap_reference"] = provenance(hundred_m, args.hundred_m)

    if args.dry_run:
        print(json.dumps(bspot, indent=1))
        return
    args.bspot.write_text(json.dumps(bspot, indent=1) + "\n")
    print(f"updated {args.bspot}")


if __name__ == "__main__":
    main()
