#!/usr/bin/env bash
# Run queued eval_suite jobs within a time budget (resumable; call repeatedly).
# Queue file: one job per line:  NAME|MODEL_SPEC|SETS|EXTRA FLAGS|REQUIRED FILES (optional, space-separated)
# A job whose required files (e.g. a finished training history) do not exist yet is skipped.
# Usage: scripts/eval_queue.sh QUEUE_FILE BUDGET_SECONDS [THREADS]
cd "$(dirname "$0")/.."
Q=$1; BUDGET=$2; TH=${3:-4}
start=$(date +%s)
pending=0
while IFS='|' read -r name model sets extra require; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  marker="results/suite/$name/DONE_${sets//,/_}"
  [ -f "$marker" ] && continue
  ready=1
  for f in $require; do [ -f "$f" ] || ready=0; done
  if [ $ready = 0 ]; then pending=$((pending + 1)); continue; fi
  left=$(( BUDGET - ($(date +%s) - start) ))
  [ $left -lt 25 ] && { echo "budget used; next: $name"; exit 0; }
  python3 -W ignore scripts/eval_suite.py "$model" --name "$name" --sets "$sets" --budget $left --threads $TH $extra 2>&1 \
    | grep -E "PAUSED|SUITE DONE|Error|refused" | tail -2
  [ -f "$marker" ] || exit 0
done < "$Q"
echo "QUEUE EMPTY (waiting for $pending unfinished trainings)"
