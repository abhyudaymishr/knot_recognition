# Knot Recognition

## Abstract
This project provides a scientific pipeline for knot recognition from images. It combines a ResNet-based CNN classifier with a structured, heuristic Gauss/PD extractor operating on skeletonized drawings. The repository is organized to support reproducible experiments, clear documentation, and future extensions.

## Installation
```bash
pip install knot-recognition
```

## Quickstart
```bash
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```

```bash
knot-moves --image /path/to/image.png --overlay results/figures/moves_overlay.png
```

Training:
```bash
python -m knot_recognition.train --data-dir /path/to/data --outdir ./checkpoints --epochs 20 --batch 32 --lr 1e-3
```

## Project Structure
- `src/knot_recognition/`: Core Python package (models, dataset, preprocessing, inference, Gauss/PD extraction).
- `docs/`: Methods and reproducibility notes.
- `notebooks/`: Exploratory analysis and ablations.
- `scripts/`: Experiment drivers and automation helpers.
- `data/`: Reserved for datasets and processed artifacts.
- `results/`: Reserved for experiment outputs and figures.
- `raw_knot/`: Legacy dataset location (kept for compatibility).
- `outputs/`: Legacy outputs location (kept for compatibility).
- `tests/`: Synthetic tests for Gauss/PD extraction.

## Data Format
Folder-structured dataset:
```text
data_root/
  3_1/
  4_1/
```
Each subfolder is a class label and contains images.

## Methods (Summary)
`get_resnet(num_classes=1000, pretrained=True, model_name="resnet18", freeze_backbone=False)`

- Skeleton graph -> spur pruning -> junction clustering -> graph simplification
- Edge pairing at crossings -> curve traversal -> PD construction
- Entry point: `extract_gauss_code(skel, img_gray=None, cfg=None, return_debug=False)`

## Mapping CSV Schema
`mapping_example.csv`:
```text
label,pd_code,gauss_code
3_1,"PD[ [1,2],[3,4] ]","1 -2 3"
```

## Reproducibility
- Documented environment and workflow notes are in `docs/reproducibility.md`.
- Use a clean virtual environment and pinned versions for formal experiments.

## Tests
```bash
pytest -q
```

## Citation
See `CITATION.cff` for citation metadata.

## Usage
See `USAGE.md` for end-to-end examples.

## Known Limitations
- Chirality detection is heuristic and depends on how flips affect CNN confidence.
- Gauss/PD extractor assumes clean, high-contrast drawings.
- Over/under (sign) inference is not reliable from skeletons alone.
