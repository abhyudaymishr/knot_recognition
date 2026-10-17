#!/usr/bin/env bash
# Full training recipe of the knot-aware NanoTransformer (v1 -> v2 -> v3), used for seed and
# no-FFT ablations. Usage: [DEVICE=mps] scripts/train_recipe.sh TAG SEED THREADS [--no-fft]
#   stage 1: data/knot_aware/train.npz (2,000 tube renders), 14 epochs, lr 1.5e-3
#   stage 2: data/knot_aware/train_v3.npz (3,000, 60% styled), 128-grid ridge only, 10 epochs, lr 8e-4
#   stage 3: same data, 256-grid ridge branch on the frozen trunk, 4 epochs, lr 2e-3
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=$1; SEED=$2; TH=$3; EXTRA="${4:-} --device ${DEVICE:-cpu}"
PY=${PY:-python3}
C=checkpoints/recipe; H=results/training
$PY -W ignore scripts/train_knot_aware.py --seed "$SEED" --threads "$TH" $EXTRA \
  --epochs 14 --lr 1.5e-3 --out $C/${TAG}_stage1.pth --history $H/${TAG}_stage1.json
$PY -W ignore scripts/train_knot_aware.py --seed "$SEED" --threads "$TH" $EXTRA --data-suffix _v3 --no-hi \
  --init $C/${TAG}_stage1.pth --epochs 10 --lr 8e-4 --out $C/${TAG}_stage2.pth --history $H/${TAG}_stage2.json
$PY -W ignore scripts/train_knot_aware.py --seed "$SEED" --threads "$TH" $EXTRA --data-suffix _v3 \
  --init $C/${TAG}_stage2.pth --freeze-trunk --epochs 4 --lr 2e-3 --out $C/${TAG}.pth --history $H/${TAG}_stage3.json
echo "RECIPE DONE $TAG"
