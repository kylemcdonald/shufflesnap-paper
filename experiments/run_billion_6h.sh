#!/usr/bin/env bash
# Continue the 1 h billion-point run for another 5 h (6 h in total; 2026-09-28): start from its saved
# assignment and rerun the schedule from the interrupted stride-16 level on.  Same idle check as
# run_billion.sh; outputs carry the suffix _6h, the 1 h files are kept.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$(cd ../shufflesnap/src && pwd):experiments
PY=${PYTHON:-../.venv/bin/python}
# the earlier work's saved assignments of this instance (4 GB each, not distributed), priced for comparison
OLD=${OLD:-../../../shufflesnap-paper/benchmarks/results/assignments}
LOG=results/billion/run_6h.log
idle=0
while [ $idle -lt 3 ]; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  load=$(awk '{print int($1)}' /proc/loadavg)
  if [ "$avail" -ge 55 ] && [ "$load" -le 2 ]; then idle=$((idle + 1)); else idle=0; fi
  echo "$(date -Is) waiting: available ${avail} GiB, load ${load}, idle ${idle}" >> $LOG
  [ $idle -lt 3 ] && sleep 60
done
echo "=== start $(date -Is) load $(cat /proc/loadavg)" >> $LOG
$PY experiments/billion.py --budget 18000 --threads 24 \
  --resume results/billion/billion_assignment_31623_seed0.npy --resume-stride 16 \
  --time-offset $($PY -c "import json; print(json.load(open('results/billion/billion_31623_seed0.json'))['total_s'])") \
  --tag _6h --compare $OLD/billion_meander_31623_seed0.npy $OLD/billion_bspot_31623_seed0.npy >> $LOG 2>&1
echo "=== done $(date -Is) load $(cat /proc/loadavg)" >> $LOG
