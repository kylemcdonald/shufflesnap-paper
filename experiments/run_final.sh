#!/usr/bin/env bash
# Final result set (2026-09-26). results/raw_final/ holds every record of results/raw_bspmatch/
# except the jittered-lattice distribution and the font-based S mask, which were dropped from
# the benchmark. This script adds every method on the rectilinear S mask of the earlier
# ShuffleSnap work (grid "srect"), against the same frozen package snapshot; the resumable
# harness runs only the missing specs. Earlier result sets are left untouched.
set -u
cd "$(dirname "$0")/.."
export SHUFFLESNAP_SRC=$(cd ../frozen/shufflesnap-bspmatch-20260926/src && pwd)
export SS_RESULTS_RAW=$(pwd)/results/raw_final
export SS_RESULTS_ASSIGN=$(pwd)/results/assignments_final
export PYTHONPATH="$SHUFFLESNAP_SRC:experiments:${PYTHONPATH:-}"
PY=${PYTHON:-python}
LOG=results/run_final.log
echo "=== start $(date -Is)" >> $LOG
$PY experiments/run.py masks --threads 24 >> $LOG 2>&1
echo "=== done $(date -Is)" >> $LOG
