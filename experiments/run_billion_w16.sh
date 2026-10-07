#!/usr/bin/env bash
# Probe (2026-09-28): does a 16x16 pass improve the converged billion-point result?  Continue from
# the quality run's final assignment (suffix _q) with 16x16 windows on the fine levels (strides
# 4, 3, 2, 1, then stride-1 rounds to a fixed point), capped at 8 h.  Outputs: suffix _w16.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$(cd ../shufflesnap/src && pwd):experiments
PY=${PYTHON:-../.venv/bin/python}
LOG=results/billion/run_w16.log
SPENT=$($PY -c "import json; print(sum(json.load(open(f'results/billion/billion_31623_seed0{t}.json'))['total_s'] for t in ('', '_6h', '_q')))")
BUDGET=28800
echo "=== start $(date -Is) spent $SPENT budget $BUDGET load $(cat /proc/loadavg)" >> $LOG
$PY experiments/billion.py --budget $BUDGET --threads 24 --preset w16 \
  --resume results/billion/billion_assignment_31623_seed0_q.npy --resume-stride 4 \
  --time-offset $SPENT --tag _w16 >> $LOG 2>&1
echo "=== done $(date -Is) load $(cat /proc/loadavg)" >> $LOG
