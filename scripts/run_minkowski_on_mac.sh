#!/usr/bin/env bash
# Minkowski / Lebesgue ablation on the Mac's GPU (PyTorch MPS). Run in macOS Terminal from the
# repository root (MPS is not reachable from a Linux VM):
#     bash scripts/run_minkowski_on_mac.sh
# Trains three variants x three seeds with the full recipe (scripts/run_minkowski_queue.sh):
#   lebesgue, mk_bias, mk_full  ->  checkpoints/recipe/mps_<variant>[_seed1|_seed2].pth
# Safe to stop and re-run: finished models are skipped. Logs: results/training/logs/mac_minkowski.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
export PYTORCH_ENABLE_MPS_FALLBACK=1
$PY - <<'P'
import sys, torch
ok = torch.backends.mps.is_available()
print("torch", torch.__version__, "| MPS available:", ok)
sys.exit(0 if ok else "MPS is not available: run this in macOS Terminal with an arm64 Python that has torch")
P
if [ ! -f data/knot_aware/train_v3/X.npy ]; then
  if [ ! -f data/knot_aware/train_v3.npz ]; then
    for n in train val train_v3 val_v3; do cat data/knot_aware/_transfer/${n}_part*.bin > data/knot_aware/$n.npz; done
  fi
  (cd data/knot_aware && shasum -a 256 -c SHA256SUMS)
  $PY scripts/npz_to_npy.py train val train_v3 val_v3
fi
mkdir -p results/training/logs
DEVICE=mps PREFIX=mps_ SEEDS="0 1 2" VARIANTS="mk_full mk_bias lebesgue" \
  bash scripts/run_minkowski_queue.sh 4 2>&1 | tee -a results/training/logs/mac_minkowski.log
ls -la checkpoints/recipe/ | grep mps_
