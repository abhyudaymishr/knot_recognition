# Technical Specification

## 1. Scope and Goals
This document specifies the architecture, data contracts, algorithms, and interfaces for the knot
recognition repository. The system supports:
- Image-based knot classification with a CNN.
- Heuristic Gauss/PD extraction from skeletonized drawings.
- Reidemeister move candidate detection from skeleton graphs.
- Experimental protein-knot classification using 3D backbone projections.

## 2. Module Inventory
Core package: `src/knot_recognition/`
- `preprocess.py`: image normalization, thresholding, skeletonization.
- `gauss_pd.py`: skeleton graph, crossing pairing, Gauss/PD extraction.
- `model.py`: ResNet backbone creation (`get_resnet`).
- `dataset.py`: folder-structured dataset loader.
- `train.py`: CNN training loop with deterministic seeding and split control.
- `infer.py`: inference pipeline and CLI entry.
- `features.py`: optional symmetry-invariant feature extractor.
- `reidemeister.py`: R1/R2/R3 candidate detector from skeleton graphs.
- `solver.py`: diagram reducer + classifier (heuristic R1/R2 reductions + CNN).
- `protein.py`: Cα backbone parsing, projection, crossing detection, Gauss code for proteins.
- `utils.py`: IO helpers and shared utilities.

Scripts: `scripts/`
- `extract_knotprot_stage1.py`: build Cα backbone dataset from PDBs.
- `build_knotprot_labels.py`: parse KnotProt browse exports into labels.
- `build_knotprot_hybrid_dataset.py`: render projections + Gauss histograms.
- `merge_hybrid_parts.py`: merge dataset parts + manifest.
- `train_hybrid_classifier.py`: hybrid classifier training (projection CNN + Gauss MLP).

Tests: `tests/`
- Synthetic Reidemeister fixtures and protein stage tests.

## 3. Data Flow
### 3.1 Image-Based Pipeline
1. Load image → `preprocess.py` (grayscale → blur → threshold → skeleton).
2. Optional Gauss/PD extraction → `gauss_pd.py`.
3. CNN inference → `model.py` via `infer.py`.
4. Optional features → `features.py`.

### 3.1a Diagram Reducer + Classifier
1. Skeletonize image.
2. Iterative R1/R2 reductions on the skeleton graph (`solver.py`).
3. Extract Gauss/PD after reduction.
4. Classify using CNN (optional checkpoint).

### 3.2 Reidemeister Detector
1. Preprocess → skeleton.
2. Build pixel graph → `gauss_pd.py`.
3. Detect candidates → `reidemeister.py`.
4. Apply scoring + NMS → output JSON list.

### 3.3 Protein Pipeline (KnotProt)
1. **Cα extraction**: PDB → `ProteinBackbone`.
2. **Projection**: sample viewpoints, project to 2D.
3. **Crossing detection**: segment intersections with depth ordering.
4. **Hybrid dataset**: projection image + Gauss histogram.
5. **Hybrid classifier**: CNN + MLP on Gauss histogram.

## 4. Algorithms
### 4.1 Preprocessing
- Histogram equalization, Gaussian blur, adaptive thresholding with Otsu fallback.
- Morphological opening for noise removal.
- Skeletonization for topology extraction.

### 4.2 Gauss/PD Extraction
- Build skeleton graph (4/8 connectivity).
- Prune short spurs.
- Cluster junction nodes.
- Simplify graph into terminal-to-terminal paths.
- Pair edges at crossings via local orientation.
- Trace curve and assemble Gauss/PD codes.

### 4.3 CNN Classifier
- ResNet backbone (`get_resnet`).
- Cross-entropy loss; standard augmentations in `train.py`.
- Configurable split strategy: random or stratified.

### 4.4 Reidemeister Candidates
- R1: small cycles under junction/length constraints.
- R2: short parallel segments in simplified graph.
- R3: triangle among junctions under perimeter/ratio constraints.
- Optional geometry-based scoring + NMS.

### 4.5 Protein Projection + Gauss
- Viewpoints sampled on the sphere (Fibonacci lattice).
- 2D projections plus depth values.
- Crossing ordering inferred by depth at intersection.
- Gauss histogram used as MLP input.

## 5. Interfaces
### 5.1 CLI
- `knot --image ... --checkpoint ... --mapping ...`
- `knot-moves --image ... --overlay ...`
- `python -m knot_recognition.train ...`
- `python -m knot_recognition.infer ...`
- `python scripts/train_hybrid_classifier.py ...`

### 5.2 Python API
```python
from knot_recognition.infer import infer_image
result = infer_image(img_path, checkpoint, mapping_csv)
```

## 6. Data Contracts
### 6.1 Image Dataset
Folder layout:
```text
data_root/
  3_1/
  4_1/
```
Each subfolder name is a class label.

### 6.2 Mapping CSV
`mapping_example.csv`:
```text
label,pd_code,gauss_code
```

### 6.3 Protein Hybrid Dataset
`data/knotprot/hybrid_dataset.npz`:
- `images`: `N x H x W` uint8
- `gauss_feats`: `N x F` float32
- `labels`: `N` string labels
- `keys` (optional): protein group keys for leakage-free splits

`data/knotprot/hybrid_manifest.csv`:
```text
key,view_id
```

### 6.4 Checkpoints
`train.py` saves:
- `model_state`, `optimizer_state`, `epoch`, `train_config`, `class_to_idx`

`train_hybrid_classifier.py` saves:
- `model_state`, `class_to_idx`

## 7. Reproducibility
- Seed control in `train.py` for Python, NumPy, Torch.
- Deterministic behavior with `torch.backends.cudnn.deterministic=True`.
- Protein splits must be **grouped by protein key** to avoid leakage.

## 8. Outputs
- `checkpoints/`: trained models.
- `results/`: logs and plots.
- `docs/figures/`: publication-ready templates.

## 9. Known Limitations
- Gauss/PD extraction is heuristic and sensitive to binarization.
- Reidemeister detection yields candidates, not proofs.
- Protein projections are approximate; depth ties can misorder crossings.
