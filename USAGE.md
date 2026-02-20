# Usage

## Install
```bash
pip install knot-recognition
```

## Train a model
```bash
python -m knot_recognition.train --data-dir /path/to/data --outdir ./checkpoints --epochs 20 --batch 32 --lr 1e-3
```

Training on a specific device:
```bash
python -m knot_recognition.train --data-dir /path/to/data --device cpu
```

## Run inference (CLI)
```bash
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```

Include symmetry-invariant features:
```bash
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv --features
```

Force a device:
```bash
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --device cpu
```

## Detect Reidemeister moves (heuristic)
```bash
knot-moves --image /path/to/image.png --overlay results/figures/moves_overlay.png
```

Output is a JSON list of candidate moves with bounding boxes and scores.

## Reduce + classify (solver)
```bash
knot-solve --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```

Save the reduced skeleton:
```bash
knot-solve --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv \
  --reduced-skeleton results/figures/reduced_skeleton.png
```

## Run inference (module)
```bash
python -m knot_recognition.infer --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```

Force a device:
```bash
python -m knot_recognition.infer --image /path/to/image.png --checkpoint ./checkpoints/best.pth --device cuda
```

## Protein Knot Stages
Stage 1 (Cα backbones):
```bash
PYTHONPATH=./src python scripts/extract_knotprot_stage1.py \
  --pdb-dir data/knotprot/pdb \
  --out data/knotprot/backbones.npz \
  --manifest data/knotprot/backbones.csv
```

Stage 2–3 (projection + Gauss):
- Use `sample_viewpoints`, `project_polyline`, `detect_crossings`, `gauss_code_from_crossings`

Stage 4 (hybrid ML):
```bash
PYTHONPATH=./src python scripts/build_knotprot_hybrid_dataset.py \
  --viewpoints 32 --limit 20 --offset 0 --stride 3 --max-points 300 \
  --out data/knotprot/hybrid_dataset_part1.npz \
  --manifest data/knotprot/hybrid_manifest_part1.csv

PYTHONPATH=./src python scripts/merge_hybrid_parts.py \
  --out data/knotprot/hybrid_dataset.npz

python scripts/train_hybrid_classifier.py \
  --data data/knotprot/hybrid_dataset.npz \
  --out checkpoints/hybrid_classifier.pth \
  --epochs 2 --batch 64 --lr 1e-3
```

## Python API
```python
from knot_recognition.infer import infer_image

result = infer_image(
    img_path="path/to/image.png",
    checkpoint="./checkpoints/best.pth",
    mapping_csv="mapping_example.csv"
)

print(result)
```

## Output fields
- `predicted_label`, `pred_prob`
- `mapping_pd`, `mapping_gauss`
- `auto_gauss`, `auto_pd` (heuristic)
- `features` (symmetry-invariant descriptor if `--features` is set)
- `chirality`, `chirality_confidence`
