#!/usr/bin/env bash
# RasterFairy's swap optimizer at N=2,500 (2026-10-08), added to the final result set
# (results/raw/) against the same frozen package snapshot as run_all.sh. Certified
# references for the new instances first, then all methods pinned to one performance core
# (CPU 8, as run_all.sh) and, concurrently on another physical core (CPU 12), the 1e7-iteration
# optimizer runs on the first seed.
set -u
cd "$(dirname "$0")/.."
export SHUFFLESNAP_SRC=$(cd ../frozen/shufflesnap-bspmatch-20260926/src && pwd)
export SS_RESULTS_RAW=$(pwd)/results/raw
export PYTHONPATH="$SHUFFLESNAP_SRC:experiments:${PYTHONPATH:-}"
PY=${PYTHON:-python}
LOG=results/run_swap.log
echo "=== start $(date -Is)" >> $LOG
$PY experiments/references_all.py 6 >> $LOG 2>&1
$PY experiments/run.py swap_long --threads 1 --cpus 12 >> results/run_swap_long.log 2>&1 &
$PY experiments/run.py swap --threads 1 --cpus 8 >> $LOG 2>&1
wait
echo "=== done $(date -Is)" >> $LOG
# added after the runs above: 1e7 iterations on seeds 2-3 as well
$PY experiments/run_swap_long_seeds.py >> results/run_swap_long.log 2>&1
