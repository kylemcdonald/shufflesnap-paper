#!/usr/bin/env bash
# Fill the rest of a 6 h total (2026-09-28): the balanced continuation (run_billion_6h.sh) converged
# after 1 h 43 min in total, so continue from its fixed point with the quality preset (12x12 windows)
# on its fine levels (strides 4, 3, 2, 1) until convergence or the 6 h total.  Outputs: suffix _q.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$(cd ../shufflesnap/src && pwd):experiments
PY=${PYTHON:-../.venv/bin/python}
OLD=../../../shufflesnap-paper/benchmarks/results/assignments
LOG=results/billion/run_quality.log
SPENT=$($PY -c "import json; print(sum(json.load(open(f'results/billion/billion_31623_seed0{t}.json'))['total_s'] for t in ('', '_6h')))")
BUDGET=$($PY -c "print(21600 - $SPENT)")
echo "=== start $(date -Is) spent $SPENT budget $BUDGET load $(cat /proc/loadavg)" >> $LOG
$PY experiments/billion.py --budget $BUDGET --threads 24 --preset quality \
  --resume results/billion/billion_assignment_31623_seed0_6h.npy --resume-stride 4 \
  --time-offset $SPENT --tag _q \
  --compare $OLD/billion_meander_31623_seed0.npy $OLD/billion_bspot_31623_seed0.npy >> $LOG 2>&1
echo "=== done $(date -Is) load $(cat /proc/loadavg)" >> $LOG
