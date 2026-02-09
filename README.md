# Knot Recognition

This project classifies knot images with a ResNet-based CNN and optionally maps predicted labels to PD/Gauss codes. It also includes a heuristic Gauss/PD extractor that operates directly on skeletonized images.

## Project Layout
- `model.py`: Dataset class plus `get_resnet(...)` model builder.
- `dataset.py`: `KnotImageDataset` (folder-structured image dataset).
- `train.py`: Training loop and checkpoint saving.
- `infer.py`: Inference CLI, mapping lookup, heuristic Gauss/PD extraction, chirality heuristic.
- `preprocess.py`: Image preprocessing and skeletonization.
- `gauss_pd.py`: Heuristic Gauss/PD extraction pipeline.
- `features.py`: Handcrafted symmetry-invariant features (not wired into training).
- `generate_gauss_codes.py`: Fetches Gauss codes from KAtlas (network required).
- `mapping_example.csv`: Example label to PD/Gauss mapping format.
- `requirements.txt`: Python dependencies.
- `tests/test_gauss_pd.py`: Synthetic tests for Gauss/PD pipeline.
- `pytest.ini`: Pytest configuration.
- `raw_knot/`, `outputs/`: Example data and generated artifacts.

## Install
Local development install (recommended):
```bash
pip install -e .
```

If you only want dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Format
Directory structure:
```text
data_root/
  3_1/
  4_1/
```
Each subfolder is a class label and contains images.

## Training
```bash
python -m knot_recognition.train --data-dir /path/to/data --outdir ./checkpoints --epochs 20 --batch 32 --lr 1e-3
```
Outputs: `./checkpoints/best.pth`

## Model Builder
`get_resnet(num_classes=1000, pretrained=True, model_name="resnet18", freeze_backbone=False)`

Example:
```python
from knot_recognition.model import get_resnet
model = get_resnet(num_classes=10, pretrained=True, model_name="resnet18")
```

## Inference
```bash
python -m knot_recognition.infer --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```

Or, after installation:
```bash
knot-recognition --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```
Output JSON includes:
- `predicted_label`, `pred_prob`
- `mapping_pd`, `mapping_gauss` (from CSV)
- `auto_gauss`, `auto_pd` (heuristic extractor)
- `chirality`, `chirality_confidence`

## Mapping CSV Schema
`mapping_example.csv`:
```text
label,pd_code,gauss_code
3_1,"PD[ [1,2],[3,4] ]","1 -2 3"
```

## Heuristic Gauss/PD Extraction
- Entry point: `extract_gauss_code(skel, img_gray=None, cfg=None, return_debug=False)`
- Returns:
  - `gauss`: list of crossing IDs in traversal order
  - `pd`: dict with `crossings` and `edge_labels`
- Pipeline stages:
  - Skeleton graph -> spur pruning -> junction clustering -> graph simplification
  - Edge pairing at crossings -> curve traversal -> PD construction
- Designed for structural correctness. Over/under inference is not robust.

## Handcrafted Features
- `features.py` includes Hu moments, radial Fourier descriptors, and reflection symmetry scores.
- Not currently used in training or inference.

## Tests
```bash
pytest -q
```

## Known Limitations
- Chirality detection is heuristic and depends on how flips affect CNN confidence.
- Gauss/PD extractor assumes clean, high-contrast drawings.
- Over/under (sign) inference is not reliable from skeletons alone.
