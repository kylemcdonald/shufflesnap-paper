#!/usr/bin/env bash
# The billion-point meander with the current kernel, balanced preset, 1 h budget, 24 threads
# (2026-09-26).  Waits until the machine is idle enough (about 53 GiB; the
# cell centers stay in RAM because the disk had too little room for a file-backed memmap), then runs
# and prices the earlier work's saved assignments of the same instance on the same frame.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$(cd ../shufflesnap/src && pwd):experiments
PY=${PYTHON:-../.venv/bin/python}
OLD=../../../shufflesnap-paper/benchmarks/results/assignments
LOG=results/billion/run.log
mkdir -p results/billion
idle=0
while [ $idle -lt 3 ]; do  # three consecutive idle minutes, so a gap between other jobs does not count
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  load=$(awk '{print int($1)}' /proc/loadavg)
  if [ "$avail" -ge 55 ] && [ "$load" -le 2 ]; then idle=$((idle + 1)); else idle=0; fi
  echo "$(date -Is) waiting: available ${avail} GiB, load ${load}, idle ${idle}" >> $LOG
  [ $idle -lt 3 ] && sleep 60
done
echo "=== start $(date -Is) load $(cat /proc/loadavg)" >> $LOG
$PY experiments/billion.py --budget 3600 --threads 24 \
  --compare $OLD/billion_meander_31623_seed0.npy $OLD/billion_bspot_31623_seed0.npy >> $LOG 2>&1
rm -f results/billion/centers.tmp
echo "=== done $(date -Is) load $(cat /proc/loadavg)" >> $LOG
