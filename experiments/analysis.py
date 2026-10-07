"""Load raw JSONL results into a tidy table and join certified references.

Excess is always computed here (not at run time) against the cached certified optimum
for the same (dist, n, seed, grid, normalization).  Only certified references are used
for "excess over optimum"; an uncertified comparison is labelled as such by callers.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)


def _ref_table():
    rows = []
    for p in glob.glob(os.path.join(ROOT, "results", "reference", "*.npz")):
        z = np.load(p, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        rows.append(m)
    return pd.DataFrame(rows)


def load(pattern="*.jsonl", with_refs=True):
    raw = os.environ.get("SS_RESULTS_RAW", os.path.join(ROOT, "results", "raw"))
    files = sorted(glob.glob(os.path.join(raw, pattern)))
    recs = []
    for f in files:
        base = os.path.basename(f)
        if base.startswith(("explore", "dev_")):
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sp = r.get("spec", {})
                if not sp:
                    continue
                d = dict(file=base, id=r.get("id"), method=sp.get("method"), label=sp.get("label"),
                         dist=sp.get("dist"), n=sp.get("n"), seed=sp.get("seed"), grid=sp.get("grid", "square"),
                         threads=r.get("threads"), cpus=r.get("cpus"), time_s=r.get("time_s"),
                         warmup_s=r.get("warmup_s"), peak_rss_mb=r.get("peak_rss_mb"), cost=r.get("cost"),
                         mean_sq=r.get("mean_sq"), n_cells=r.get("n_cells"), valid=r.get("valid"),
                         error=r.get("error"), knn10=r.get("knn10"),
                         lattice_nbr=r.get("lattice_nbr_in_input_8nn"))
                for k, v in (r.get("disp") or {}).items():
                    d["disp_" + k] = v
                for k, v in (sp.get("params") or {}).items():
                    d["p_" + k] = v if not isinstance(v, (list, dict)) else json.dumps(v)
                info = r.get("info") or {}
                for k in ("finished", "windows_solved", "gap", "certified", "rounds", "lower_bound"):
                    if k in info:
                        d["info_" + k] = info[k]
                if "trace" in r:
                    d["trace"] = r["trace"]
                recs.append(d)
    df = pd.DataFrame(recs)
    if df.empty or not with_refs:
        return df
    ref = _ref_table()
    if not ref.empty:
        parts = ref["key"].str.extract(r"^(?P<dist>[a-z0-9]+)-n(?P<n>\d+)-s(?P<seed>\d+)-(?P<grid>[a-z0-9]+)-bbox$")
        ref = pd.concat([ref, parts], axis=1)
        ref["n"] = ref["n"].astype(int)
        ref["seed"] = ref["seed"].astype(int)
        ref = ref.rename(columns={"cost": "ref_cost", "certified": "ref_certified", "gap": "ref_gap",
                                  "lower_bound": "ref_lb"})
        df = df.merge(ref[["dist", "n", "seed", "grid", "ref_cost", "ref_certified", "ref_gap", "ref_lb"]],
                      on=["dist", "n", "seed", "grid"], how="left")
        ok = df["ref_certified"].fillna(False).astype(bool)
        df["excess_pct"] = np.where(ok, 100.0 * (df["cost"] / df["ref_cost"] - 1.0), np.nan)
        df["excess_per_pt"] = np.where(ok, (df["cost"] - df["ref_cost"]) / df["n"], np.nan)
    # Beyond the reference cache (>200k points), a timed exact run that certified itself
    # provides the optimum for the same instance.
    if "info_certified" in df and "ref_cost" in df:
        cert = df[df["method"].eq("exact") & df["info_certified"].fillna(False).astype(bool) & df["error"].isna()]
        if not cert.empty:
            best = cert.groupby(["dist", "n", "seed", "grid"])["cost"].min()
            key = list(zip(df["dist"], df["n"], df["seed"], df["grid"]))
            fill = pd.Series([best.get(k, np.nan) for k in key], index=df.index)
            miss = df["ref_cost"].isna() & fill.notna()
            df.loc[miss, "ref_cost"] = fill[miss]
            df.loc[miss, "ref_certified"] = True
            ok = df["ref_certified"].fillna(False).astype(bool)
            df["excess_pct"] = np.where(ok, 100.0 * (df["cost"] / df["ref_cost"] - 1.0), np.nan)
            df["excess_per_pt"] = np.where(ok, (df["cost"] - df["ref_cost"]) / df["n"], np.nan)
    # an "exact" run counts as exact only if it certified itself; otherwise it is flagged
    if "info_certified" in df:
        is_exact = df["method"].eq("exact")
        df["exact_status"] = np.where(~is_exact, "",
                                      np.where(df["error"].notna(), "timeout",
                                               np.where(df["info_certified"].fillna(False).astype(bool),
                                                        "certified", "not certified")))
    return df


def family(label):
    if isinstance(label, str) and label.startswith("BSP-OT"):
        return "BSP-OT"
    return label


def summarize(df, by=("dist", "n", "grid", "label"), vals=("excess_pct", "excess_per_pt", "time_s", "peak_rss_mb")):
    g = df.groupby(list(by))
    out = g[list(vals)].agg(["mean", "std", "median", "count"])
    return out
