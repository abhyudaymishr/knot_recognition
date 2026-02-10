# Usage

## Install
```bash
pip install knot-recognition
```

## Train a model
```bash
python -m knot_recognition.train --data-dir /path/to/data --outdir ./checkpoints --epochs 20 --batch 32 --lr 1e-3
```

## Run inference (CLI)
```bash
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```

## Detect Reidemeister moves (heuristic)
```bash
knot-moves --image /path/to/image.png --overlay results/figures/moves_overlay.png
```

Output is a JSON list of candidate moves with bounding boxes and scores.

## Run inference (module)
```bash
python -m knot_recognition.infer --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
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
- `chirality`, `chirality_confidence`
