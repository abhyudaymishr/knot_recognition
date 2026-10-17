#!/usr/bin/env bash
# Train the UNetFormer together with the next-point pointer head on the Apple GPU (protocol 13.8).
#
# For each seed the UNetFormer and the head are trained from scratch with the recipe of the
# UNetFormer comparison (train_v3, 12 epochs, batch 16, lr 1.5e-3); the loss is the UNetFormer's
# loss plus the pointer loss, so the strand order (including every jump across a crossing) shapes
# the UNetFormer's features. For comparison, a head is also trained on the plain UNetFormer of the
# same seed (frozen, checkpoints/beat_unet/mps_uf_s<seed>.pth). Both are scored on held-out
# validation images 150-299, alone and with the jump re-checks (strand_pointer.recheck: curved
# ridge check with re-linking, then the straight check). The summary table is printed at the end.
#
# Usage (macOS Terminal, in the repo folder):  bash scripts/run_unetformer_pointer_on_mac.sh
# Options (environment variables):
#   SEEDS="0 1 2"   BS=16   EPOCHS=12
#   WORKERS=0       CPU processes that prepare batches while the GPU trains (0 = 4; each holds ~0.4 GB)
#   MAXPTS=640      most points per image for the head; lower it (or BS) if the GPU runs out of memory
#   FROZEN=1        also train the comparison head on the frozen plain UNetFormer (0 = skip)
#   CONTROL=0       1 = also train the same pipeline without the head (pointer weight 0)
#   EVAL_SUITE=0    1 = also run the parser on KnotBench, close strands and real pictures with each new
#                   UNetFormer (CPU; about an hour per model)
# Finished steps are skipped, so the script can be stopped (Ctrl-C) and started again.
# Outputs: checkpoints/uf_pointer/, checkpoints/strand_pointer/mps_ptr_s<seed>.pth, results/uf_pointer/,
#          results/uf_pointer_summary.json; logs in results/training/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE=${DEVICE:-mps}
SEEDS=${SEEDS:-"0 1 2"}
BS=${BS:-16}
EPOCHS=${EPOCHS:-12}
WORKERS=${WORKERS:-0}
FROZEN=${FROZEN:-1}
CONTROL=${CONTROL:-0}
EVAL_SUITE=${EVAL_SUITE:-0}
MAXPTS=${MAXPTS:-640}
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
export PYTORCH_ENABLE_MPS_FALLBACK=1

# keep the Mac awake for the whole run (display may dim; plug in the power for overnight runs)
if [ "$(uname)" = Darwin ] && [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export CAFFEINATED=1
  exec caffeinate -ims bash "$0" "$@"
fi

if [ "$DEVICE" = mps ]; then
  cores=$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2; exit}' || true)
  ram=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1073741824}' || true)
  echo "GPU: ${cores:-?} cores (MPS uses all of them; the data workers keep them busy); memory: ${ram:-?} GB shared by CPU and GPU"
  echo "Watch the s/step and GPU memory in the progress lines: a jump in s/step means memory is swapping (then use BS=8)"
  $PY - <<'EOF'
import sys, torch
ok = torch.backends.mps.is_available()
print("torch", torch.__version__, "| Apple GPU (MPS):", ok)
import scipy, skimage, cv2  # noqa: F401  (needed by the data workers)
sys.exit(0 if ok else "MPS not available: run this in the macOS Terminal, not inside a Linux VM")
EOF
fi

# data: memory-mapped arrays and the ordered strand paths
for s in train_v3 val_v3; do
  if [ ! -f data/knot_aware/$s/X.npy ] || [ ! -f data/knot_aware/$s/center_hi.npy ]; then $PY scripts/npz_to_npy.py $s; fi
  want=$([ $s = train_v3 ] && echo 3000 || echo 300)
  n=$($PY -c "import pickle; print(len(pickle.load(open('data/knot_aware/${s}_strands.pkl','rb'))))" 2>/dev/null || echo 0)
  if [ "$n" != "$want" ]; then $PY -u scripts/make_strand_sidecar.py --split $s; fi
  if [ ! -f data/knot_aware/.${s}_strands_checked ]; then
    $PY scripts/check_strand_sidecar.py --split $s && touch data/knot_aware/.${s}_strands_checked
  fi
done

C=checkpoints/uf_pointer; R=results/uf_pointer; L=results/training/logs
mkdir -p $C $R $L checkpoints/strand_pointer results/strand_pointer

score() {   # held-out scoring of one head (CPU): links, jumps, re-checks
  local head=$1 tag=$2
  [ -f "$head" ] || return 0
  [ -f $R/heldout_$tag.json ] || $PY scripts/eval_strand_pointer.py --head "$head" --start 150 --n 150 --out $R/heldout_$tag.json > $L/heldout_$tag.log 2>&1
  [ -f $R/recheck_$tag.json ] || $PY scripts/eval_recheck.py 150 150 $R/recheck_$tag.json --head "$head" > $L/recheck_$tag.log 2>&1
  echo "  scored $tag on held-out images 150-299"
}

for seed in $SEEDS; do
  echo "=== seed $seed: UNetFormer + pointer head, trained together ==="
  J=$C/${DEVICE}_ufptr_s${seed}.pth
  if [ ! -f $R/${DEVICE}_ufptr_s${seed}_history.json ]; then
    $PY -u scripts/train_unetformer_pointer.py --device "$DEVICE" --seed "$seed" --bs "$BS" --epochs "$EPOCHS" \
        --workers "$WORKERS" --max-points "$MAXPTS" --out "$J" 2>&1 | tee $L/ufptr_${DEVICE}_s${seed}.log
  else echo "  [skip] trained"; fi

  if [ "$FROZEN" = 1 ]; then
    echo "=== seed $seed: pointer head on the frozen plain UNetFormer (comparison) ==="
    F=checkpoints/strand_pointer/${DEVICE}_ptr_s${seed}.pth
    if [ ! -f "$F" ]; then
      # 4500 steps of 8 images: about as many images as the joint head sees in 12 epochs
      $PY -u scripts/train_strand_pointer.py --device "$DEVICE" --trunk checkpoints/beat_unet/mps_uf_s${seed}.pth \
          --seed "$seed" --steps 4500 --bs 8 --n-val 150 --max-points "$MAXPTS" --out "$F" --log results/strand_pointer/${DEVICE}_ptr_s${seed}.json \
          2>&1 | tee $L/ptr_${DEVICE}_s${seed}.log
    else echo "  [skip] trained"; fi
  fi

  if [ "$CONTROL" = 1 ]; then
    echo "=== seed $seed: same pipeline without the head (control) ==="
    K=$C/${DEVICE}_ufctl_s${seed}.pth
    if [ ! -f $R/${DEVICE}_ufctl_s${seed}_history.json ]; then
      $PY -u scripts/train_unetformer_pointer.py --device "$DEVICE" --seed "$seed" --bs "$BS" --epochs "$EPOCHS" \
          --workers "$WORKERS" --pointer-weight 0 --out "$K" 2>&1 | tee $L/ufctl_${DEVICE}_s${seed}.log
    else echo "  [skip] trained"; fi
  fi

  score $C/${DEVICE}_ufptr_s${seed}_head.pth ${DEVICE}_ufptr_s${seed}
  [ "$FROZEN" = 1 ] && score checkpoints/strand_pointer/${DEVICE}_ptr_s${seed}.pth ${DEVICE}_ptr_s${seed}

  if [ "$EVAL_SUITE" = 1 ]; then
    for m in ufptr ufctl; do
      ck=$C/${DEVICE}_${m}_s${seed}.pth
      [ -f "$ck" ] || continue
      if [ ! -f results/suite/${m}_s${seed}/DONE_kb_close_real ]; then
        echo "=== seed $seed: parser on KnotBench, close strands, real pictures with $m ==="
        $PY -W ignore scripts/eval_suite.py "$ck" --name ${m}_s${seed} --sets kb,close,real --clutter --role ablation \
            2>&1 | tee $L/suite_${m}_s${seed}.log
      fi
    done
  fi
done

$PY scripts/summarize_uf_pointer.py --device "$DEVICE" --seeds "$(echo $SEEDS | tr ' ' ',')"
