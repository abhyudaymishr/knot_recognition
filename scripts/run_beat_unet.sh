#!/usr/bin/env bash
# Matched comparison against the U-Net pair: same data (train_v3: 3,000 renders, 60% styled), one-stage
# recipe (12 epochs, lr 1.5e-3, batch 16, all heads jointly), same device, three seeds each.
#   unet_seg / unet_kp  the U-Net pair baseline (1.6M + 1.6M)
#   uf                  UNetFormer: U-Net encoder/decoder + transformer bottleneck, all heads (2.9M)
#   nano3m              NanoTransformer widened to 2.9M (embed 192, depth 8, stem 64), one stage, joint 256-grid ridge
#   uf_topo             UNetFormer + clDice centre-line topology loss (weight 1)
# Seeds run in order 0, 1, 2 so that partial results are already a complete seed-0 comparison.
# Usage (macOS Terminal, Apple GPU):  bash scripts/run_beat_unet.sh
#        (any machine):               DEVICE=cpu THREADS=2 ONLY="uf nano3m" SEEDS=0 bash scripts/run_beat_unet.sh
# Outputs: checkpoints/beat_unet/<prefix><model>_s<seed>.pth, results/training/beat_unet/*.json, log in results/training/logs/
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE=${DEVICE:-mps}
THREADS=${THREADS:-4}
SEEDS=${SEEDS:-"0 1 2"}
ONLY=${ONLY:-"unet_seg unet_kp uf nano3m uf_topo"}
PREFIX=${PREFIX:-$( [ "$DEVICE" = mps ] && echo mps_ || echo "${DEVICE}_" )}
export PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ "$DEVICE" = mps ]; then
  $PY -c "import torch,sys; ok=torch.backends.mps.is_available(); print('torch',torch.__version__,'MPS',ok); sys.exit(0 if ok else 'MPS not available: run in macOS Terminal')"
fi
if [ ! -f data/knot_aware/train_v3/X.npy ]; then $PY scripts/npz_to_npy.py train_v3 val_v3; fi
C=checkpoints/beat_unet; H=results/training/beat_unet
mkdir -p $C $H results/training/logs
flags() {
  case $1 in
    unet_seg) echo "--arch unet_seg" ;;
    unet_kp)  echo "--arch unet_kp" ;;
    uf)       echo "--arch unetformer" ;;
    uf_topo)  echo "--arch unetformer --topo-loss 1.0" ;;
    nano3m)   echo "--arch knot_aware --embed-dim 192 --depth 8 --heads 6 --stem-dim 64" ;;
    *) echo "unknown model $1" >&2; exit 2 ;;
  esac
}
for s in $SEEDS; do
  for mdl in $ONLY; do
    tag=${PREFIX}${mdl}_s$s
    [ -f $H/$tag.json ] && { echo "skip $tag (done)"; continue; }
    echo "=== $tag  $(flags $mdl)  $(date)"
    $PY -W ignore scripts/train_knot_aware.py $(flags $mdl) --data-suffix _v3 --epochs 12 --lr 1.5e-3 --bs 16 \
      --seed $s --threads $THREADS --device $DEVICE --out $C/$tag.pth --history $H/$tag.json
  done
done
echo "ALL DONE $(date)"; ls -la $C
