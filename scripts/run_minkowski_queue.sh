#!/usr/bin/env bash
# Minkowski / Lebesgue variants of the knot-aware NanoTransformer, full recipe each
# (scripts/train_recipe.sh: stage 1 -> 2 -> 3), added one at a time:
#   lebesgue   layer-cake (Lebesgue-measure) loss only
#   mk_bias    + Minkowski position bias (learnable p and slope per head)
#   mk_full    + Minkowski content attention (L1 heads and L2 heads)
# The plain model's three seeds (knot_aware_nanotransformer.pth, recipe/seed1, recipe/seed2) are the reference.
# Usage: [DEVICE=cpu|mps] [SEEDS="0 1 2"] [VARIANTS="mk_full mk_bias lebesgue"] [PREFIX=] scripts/run_minkowski_queue.sh THREADS
set -euo pipefail
cd "$(dirname "$0")/.."
TH=${1:-2}
SEEDS=${SEEDS:-0}
VARIANTS=${VARIANTS:-"mk_full mk_bias lebesgue"}
PREFIX=${PREFIX:-}
flags() {   # (a case statement: macOS ships bash 3.2, which has no associative arrays)
  case $1 in
    lebesgue) echo "--layer-cake 1.0" ;;
    mk_bias)  echo "--attn-bias minkowski --layer-cake 1.0" ;;
    mk_full)  echo "--attn-bias minkowski --attn-content minkowski --layer-cake 1.0" ;;
    *) echo "unknown variant $1" >&2; exit 2 ;;
  esac
}
mkdir -p checkpoints/recipe results/training
for s in $SEEDS; do
  for v in $VARIANTS; do
    tag=${PREFIX}${v}$( [ "$s" = 0 ] && echo "" || echo "_seed$s" )
    [ -f checkpoints/recipe/$tag.pth ] && { echo "skip $tag (done)"; continue; }
    echo "=== $tag  seed $s  $(flags $v)  $(date)"
    bash scripts/train_recipe.sh "$tag" "$s" "$TH" "$(flags $v)"
  done
done
echo "QUEUE DONE $(date)"
