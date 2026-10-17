#!/bin/bash
# Double-click this file in Finder (or run it in Terminal) to train the UNetFormer together with the
# strand-order head on this Mac's GPU: seeds 0, 1, 2, then the held-out scoring and the summary.
# It first stops an earlier run of the same training if one is still going, then starts
# scripts/run_unetformer_pointer_on_mac.sh, which keeps the Mac awake, uses the Apple GPU (MPS uses
# all of its cores) and keeps it fed with CPU workers. Finished steps are skipped on a restart.
# Options go in front, e.g. in Terminal:  SEEDS=0 BS=8 bash scripts/run_on_mac_gpu.command
cd "$(dirname "$0")/.." || exit 1

if pgrep -f "scripts/train_unetformer_pointer.py|scripts/run_unetformer_pointer_on_mac.sh|scripts/train_strand_pointer.py" >/dev/null; then
  echo "Stopping the earlier run of this training..."
  pkill -INT -f "scripts/run_unetformer_pointer_on_mac.sh" 2>/dev/null
  pkill -INT -f "scripts/train_unetformer_pointer.py|scripts/train_strand_pointer.py" 2>/dev/null
  sleep 10
  pkill -KILL -f "scripts/train_unetformer_pointer.py|scripts/train_strand_pointer.py|scripts/run_unetformer_pointer_on_mac.sh" 2>/dev/null
  sleep 2
fi

echo "Starting: UNetFormer + strand-order head on the Apple GPU (log: results/training/logs/)"
bash scripts/run_unetformer_pointer_on_mac.sh
status=$?
echo
if [ $status -eq 0 ]; then echo "Finished. Summary: results/uf_pointer_summary.json"; else echo "Stopped with an error (code $status); see the lines above."; fi
read -r -p "Press Return to close this window. " _
