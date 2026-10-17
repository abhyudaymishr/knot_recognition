#!/usr/bin/env bash
# Next-point pointer head (knot_recognition.strand_pointer) on the Apple GPU: for each trained
# UNetFormer seed (checkpoints/beat_unet/mps_uf_s<seed>.pth, frozen), train one head that learns
# the order in which each strand visits its points, including the jump across every crossing.
# All 3,000 training images; evaluated on the 300 validation images against the parser's rules
# on the same points. The UNetFormer itself is not changed.
#
# Usage (macOS Terminal):  bash scripts/run_strand_pointer_on_mac.sh
#        (any machine):    DEVICE=cpu SEEDS=0 STEPS=1500 BS=4 bash scripts/run_strand_pointer_on_mac.sh
# Outputs: checkpoints/strand_pointer/<device>_ptr_s<seed>.pth, results/strand_pointer/<device>_ptr_s<seed>.json
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE=${DEVICE:-mps}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-6000}
BS=${BS:-8}
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ "$DEVICE" = mps ]; then
  $PY -c "import torch,sys; ok=torch.backends.mps.is_available(); print('torch',torch.__version__,'MPS',ok); sys.exit(0 if ok else 'MPS not available: run in macOS Terminal')"
fi
# ordered strand paths of the training images (deterministic re-render, checked image by image; ~1 s per image)
for split in val_v3 train_v3; do
  n=$($PY -c "import pickle,sys; print(len(pickle.load(open('data/knot_aware/${split}_strands.pkl','rb'))))" 2>/dev/null || echo 0)
  want=$([ $split = train_v3 ] && echo 3000 || echo 300)
  if [ "$n" != "$want" ]; then $PY -u scripts/make_strand_sidecar.py --split $split; fi
done
mkdir -p checkpoints/strand_pointer results/strand_pointer results/training/logs
for seed in $SEEDS; do
  out=checkpoints/strand_pointer/${DEVICE}_ptr_s${seed}.pth
  if [ -f "$out" ]; then echo "[skip] $out exists"; continue; fi
  $PY -u scripts/train_strand_pointer.py --device "$DEVICE" --trunk checkpoints/beat_unet/mps_uf_s${seed}.pth \
      --seed "$seed" --steps "$STEPS" --bs "$BS" --n-val 300 --eval-every 2000 \
      --out "$out" --log results/strand_pointer/${DEVICE}_ptr_s${seed}.json 2>&1 | tee results/training/logs/ptr_${DEVICE}_s${seed}.log
done
echo "done: $(ls checkpoints/strand_pointer/${DEVICE}_ptr_s*.pth 2>/dev/null | wc -l) head(s)"
