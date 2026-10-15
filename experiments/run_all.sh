#!/usr/bin/env bash
# Full benchmark behind the reported results (results/raw/), sequential (one run at a
# time, fresh process per run). Prerequisites: certified references computed
# (experiments/references_all.py) and the embeddings (experiments/make_embeddings.py).
# Resumable: finished runs are skipped. RasterFairy's swap optimizer is run separately
# (experiments/run_swap.sh).
set -u
cd "$(dirname "$0")/.."
# Run against a frozen snapshot of the package (release v0.4.0, see make_frozen_snapshot.sh), so
# that editing the package during a long benchmark cannot change later runs.
if [ -z "${SHUFFLESNAP_SRC:-}" ]; then
  bash experiments/make_frozen_snapshot.sh
  SHUFFLESNAP_SRC=$(cd ../frozen/shufflesnap-bspmatch-20260926/src && pwd)
fi
export SHUFFLESNAP_SRC
export SS_RESULTS_RAW=${SS_RESULTS_RAW:-$(pwd)/results/raw}
export SS_RESULTS_ASSIGN=${SS_RESULTS_ASSIGN:-$(pwd)/results/assignments}
export PYTHONPATH="$SHUFFLESNAP_SRC:experiments:${PYTHONPATH:-}"
PY=${PYTHON:-python}
LOG=results/run_all.log
echo "=== start $(date -Is)" >> $LOG
# 1. headline comparison, single thread pinned to one performance core, then all 24 threads
$PY experiments/run.py main --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/run.py main --threads 24 --only-parallel >> $LOG 2>&1
# 2. real embeddings (neighbourhood metrics, saved assignments)
$PY experiments/run.py real --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/run.py real --threads 24 --only-parallel >> $LOG 2>&1
# 3. matched-budget curves
$PY experiments/run.py anytime --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/run.py anytime --threads 24 --only-parallel >> $LOG 2>&1
# 4. scaling to 1M points (30 min cap per run; a timeout is recorded as a result)
$PY experiments/run.py scaling --threads 1 --cpus 8 --timeout 1800 >> $LOG 2>&1
$PY experiments/run.py scaling --threads 24 --only-parallel >> $LOG 2>&1
# 5. thread scaling
$PY experiments/run.py threads >> $LOG 2>&1
# 6. masks (including the rectilinear S), prime point counts, ablations, memory
$PY experiments/run.py masks --threads 24 >> $LOG 2>&1
$PY experiments/run.py primes --threads 24 >> $LOG 2>&1
$PY experiments/run.py ablations --threads 24 >> $LOG 2>&1
$PY experiments/run.py memory --threads 1 --cpus 8 >> $LOG 2>&1
# 7. window solvers, small-instance exactness, post-processing
$PY experiments/solver_balanced.py --threads 24 >> $LOG 2>&1
$PY experiments/solver_balanced.py --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/small_exactness.py >> $LOG 2>&1
$PY experiments/polish_check.py >> $LOG 2>&1
echo "=== done $(date -Is)" >> $LOG
