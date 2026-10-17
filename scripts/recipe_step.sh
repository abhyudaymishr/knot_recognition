#!/usr/bin/env bash
# One time slice of the full recipe (scripts/train_recipe.sh) with resumable training
# (scripts/train_chunked.py). Call repeatedly until it prints "RECIPE DONE".
# Usage: scripts/recipe_step.sh TAG SEED THREADS BUDGET_SECONDS "<extra flags>"
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=$1; SEED=$2; TH=$3; BUDGET=$4; EXTRA=${5:-}
PY=${PY:-python3}
C=checkpoints/recipe; H=results/training; S=$C/_state
mkdir -p $C $H $S
run() { $PY -W ignore scripts/train_chunked.py --seed "$SEED" --threads "$TH" --budget "$BUDGET" $EXTRA "$@" 2>&1 | grep -v -i "warn" | tail -4; }
if [ ! -f $H/${TAG}_stage1.json ]; then
  echo "[stage 1]"; run --epochs 14 --lr 1.5e-3 --out $C/${TAG}_stage1.pth --history $H/${TAG}_stage1.json --state $S/${TAG}_stage1.pt
elif [ ! -f $H/${TAG}_stage2.json ]; then
  echo "[stage 2]"; run --data-suffix _v3 --no-hi --init $C/${TAG}_stage1.pth --epochs 10 --lr 8e-4 \
    --out $C/${TAG}_stage2.pth --history $H/${TAG}_stage2.json --state $S/${TAG}_stage2.pt
elif [ ! -f $H/${TAG}_stage3.json ]; then
  echo "[stage 3]"; run --data-suffix _v3 --init $C/${TAG}_stage2.pth --freeze-trunk --epochs 4 --lr 2e-3 \
    --out $C/${TAG}.pth --history $H/${TAG}_stage3.json --state $S/${TAG}_stage3.pt
fi
[ -f $H/${TAG}_stage3.json ] && echo "RECIPE DONE $TAG"
exit 0
