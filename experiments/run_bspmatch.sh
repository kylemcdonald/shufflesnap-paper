#!/usr/bin/env bash
# Rerun every ShuffleSnap-dependent measurement after the preset change (fast/balanced/quality =
# 5/8/12 windows, ratio sqrt 2, time-matched to BSP-OT 16/64; 2026-09-26), against the frozen snapshot of
# that code, into a separate result set.
# results/raw_bspmatch/ already holds copies of the comparator records (BSP-OT, RasterFairy,
# dense, OR-Tools from other starts), which do not depend on the window solver; the resumable
# harness skips those and runs only the missing ShuffleSnap and ShuffleSnap-seeded specs.
# Earlier result sets (results/raw/, raw_geometric/, raw_presets/) are left untouched.
set -u
cd "$(dirname "$0")/.."
export SHUFFLESNAP_SRC=$(cd ../frozen/shufflesnap-bspmatch-20260926/src && pwd)
export SS_RESULTS_RAW=$(pwd)/results/raw_bspmatch
export SS_RESULTS_ASSIGN=$(pwd)/results/assignments_bspmatch
export PYTHONPATH="$SHUFFLESNAP_SRC:experiments:${PYTHONPATH:-}"
PY=${PYTHON:-python}
LOG=results/run_bspmatch.log
echo "=== start $(date -Is)" >> $LOG
$PY experiments/run.py main --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/run.py main --threads 24 --only-parallel >> $LOG 2>&1
$PY experiments/run.py real --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/run.py real --threads 24 --only-parallel >> $LOG 2>&1
$PY experiments/run.py anytime --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/run.py anytime --threads 24 --only-parallel >> $LOG 2>&1
$PY experiments/run.py scaling --threads 1 --cpus 8 --timeout 1800 >> $LOG 2>&1
$PY experiments/run.py scaling --threads 24 --only-parallel >> $LOG 2>&1
$PY experiments/run.py threads >> $LOG 2>&1
$PY experiments/run.py masks --threads 24 >> $LOG 2>&1
$PY experiments/run.py primes --threads 24 >> $LOG 2>&1
$PY experiments/run.py ablations --threads 24 >> $LOG 2>&1
$PY experiments/run.py memory --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/solver_balanced.py --threads 24 >> $LOG 2>&1
$PY experiments/solver_balanced.py --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/small_exactness.py >> $LOG 2>&1
$PY experiments/polish_check.py >> $LOG 2>&1
echo "=== done $(date -Is)" >> $LOG
