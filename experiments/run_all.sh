#!/usr/bin/env bash
# Full benchmark, sequential (one run at a time, fresh process per run).
# Prerequisites: certified references computed (experiments/references_all.py) and the
# embeddings (experiments/make_embeddings.py). Resumable: finished runs are skipped.
set -u
cd "$(dirname "$0")/.."
# Run against a frozen snapshot of the package commit under test, so that editing the
# working tree during a long benchmark cannot change later runs.  The compiled extension is
# copied from the in-place build (./dev_build.sh), which must match HEAD.
if [ -z "${SHUFFLESNAP_SRC:-}" ]; then
  COMMIT=$(git -C ../shufflesnap rev-parse HEAD)
  SNAP=../frozen/shufflesnap-$COMMIT
  if [ ! -d "$SNAP/src" ]; then
    mkdir -p "$SNAP"
    git -C ../shufflesnap archive HEAD src | tar -x -C "$SNAP"
    cp ../shufflesnap/src/shufflesnap/_core*.so "$SNAP/src/shufflesnap/"
    echo "$COMMIT" > "$SNAP/COMMIT"
  fi
  SHUFFLESNAP_SRC=$(cd "$SNAP/src" && pwd)
fi
export SHUFFLESNAP_SRC
export PYTHONPATH="$SHUFFLESNAP_SRC:experiments:${PYTHONPATH:-}"
PY=${PYTHON:-python}
LOG=results/run_all.log
echo "=== start $(date -Is)" >> $LOG
# 1. headline comparison, single thread pinned to one performance core
$PY experiments/run.py main --threads 1 --cpus 8 >> $LOG 2>&1
# 2. same instances, parallel methods with all 24 hardware threads
$PY experiments/run.py main --threads 24 --only-parallel >> $LOG 2>&1
# 3. real embeddings (neighbourhood metrics, saved assignments)
$PY experiments/run.py real --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/run.py real --threads 24 --only-parallel >> $LOG 2>&1
# 4. matched-budget curves
$PY experiments/run.py anytime --threads 1 --cpus 8 >> $LOG 2>&1
$PY experiments/run.py anytime --threads 24 --only-parallel >> $LOG 2>&1
# 5. scaling to 1M points (30 min cap per run; a timeout is recorded as a result)
$PY experiments/run.py scaling --threads 1 --cpus 8 --timeout 1800 >> $LOG 2>&1
$PY experiments/run.py scaling --threads 24 --only-parallel >> $LOG 2>&1
# 6. thread scaling
$PY experiments/run.py threads >> $LOG 2>&1
# 7. quality-focused experiments with all threads
$PY experiments/run.py masks --threads 24 >> $LOG 2>&1
$PY experiments/run.py primes --threads 24 >> $LOG 2>&1
$PY experiments/run.py ablations --threads 24 >> $LOG 2>&1
echo "=== done $(date -Is)" >> $LOG
