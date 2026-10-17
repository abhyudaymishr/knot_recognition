#!/usr/bin/env bash
# Run the GPU share of the experiments on an Apple-silicon Mac (PyTorch MPS backend).
#   1. CenterNet-style keypoint baseline (U-Net, 12 epochs, same data as every other model)
#   2. three seeds (0, 1, 2) of the full knot-aware recipe (stage 1 -> 2 -> 3, scripts/train_recipe.sh)
# Usage, from the repository root in Terminal (not inside a Linux VM: MPS needs macOS):
#   bash scripts/run_on_mac_gpu.sh            # uses .venv/bin/python if present
# Outputs: checkpoints/recipe/{baseline_unet_kp,mps_seed0,mps_seed1,mps_seed2}.pth,
#          results/training/*.json, logs in results/training/mac_*.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
export PYTORCH_ENABLE_MPS_FALLBACK=1
$PY - <<'P'
import sys, torch
ok = torch.backends.mps.is_available()
print("torch", torch.__version__, "| MPS available:", ok)
sys.exit(0 if ok else "MPS is not available: run this in macOS Terminal with an arm64 Python")
P
# training data copied from the cloud workspace: reassemble once, check hashes, memory-map
if [ ! -f data/knot_aware/train_v3.npz ]; then
  for n in train val train_v3 val_v3; do cat data/knot_aware/_transfer/${n}_part*.bin > data/knot_aware/$n.npz; done
fi
(cd data/knot_aware && shasum -a 256 -c SHA256SUMS)
$PY scripts/npz_to_npy.py train val train_v3 val_v3
mkdir -p checkpoints/recipe results/training
L=results/training
if [ ! -f checkpoints/recipe/baseline_unet_kp.pth ]; then
  $PY -W ignore scripts/train_knot_aware.py --arch unet_kp --data-suffix _v3 --epochs 12 --lr 1.5e-3 --device mps \
    --out checkpoints/recipe/baseline_unet_kp.pth --history $L/baseline_unet_kp.json 2>&1 | tee $L/mac_unet_kp.log
fi
for s in 0 1 2; do
  [ -f checkpoints/recipe/mps_seed$s.pth ] && continue
  DEVICE=mps bash scripts/train_recipe.sh mps_seed$s $s 4 2>&1 | tee $L/mac_seed$s.log
done
echo "ALL DONE"; ls -la checkpoints/recipe/
