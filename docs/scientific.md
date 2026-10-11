# Scientific Documentation

## 1. Objective
This repository implements a computational pipeline for knot analysis in two domains:
1. 2D knot drawings (classification + structural code extraction).
2. 3D protein backbones (projection-based surrogate knot typing).

The implementation is hybrid by design: a learned classifier is combined with explicit geometric/topological heuristics.

## 2. System Architecture
### 2.1 Core Package
`src/knot_recognition/` is organized into orthogonal components:

| Module | Scientific role | Primary outputs |
|---|---|---|
| `preprocess.py` | Image conditioning and skeleton generation | Binary skeleton, grayscale image |
| `gauss_pd.py` | Heuristic topological code reconstruction | `gauss`, `pd` |
| `model.py` | CNN backbone construction | ResNet variants |
| `train.py` | Supervised image classifier training | `best.pth` |
| `infer.py` | Prediction, mapping lookup, chirality heuristic | JSON-like prediction dictionary |
| `features.py` | Symmetry-invariant descriptor extraction | Numeric feature vector |
| `reidemeister.py` | Candidate detection for R1/R2/R3 moves | Move list + boxes + scores |
| `solver.py` | Diagram reduction + optional classification | Reduced skeleton + codes + prediction |
| `protein.py` | C-alpha parsing and projection-level crossing logic | `ProteinBackbone`, crossings, over/under event sequence |

### 2.2 Experiment Scripts
`scripts/` contains the reproducible experiment drivers:
- `download_knotprot.py`: Downloads PDB structures from RCSB based on a chains list CSV.
- `extract_knotprot_stage1.py`: Parses C-alpha coordinates from PDB files and saves `backbones.npz`.
- `build_knotprot_hybrid_dataset.py`: Projects 3D backbones to 2D view angles and builds raster + Gauss histogram dataset.
- `merge_hybrid_parts.py`: Merges partitioned dataset `.npz` files into a unified dataset.
- `train_hybrid_classifier.py`: Trains and evaluates the hybrid CNN + Gauss feature classifier with group-stratified splits.

## 3. Methods
### 3.1 Image Preprocessing
The preprocessing stack in `preprocess.py` is:
1. Resize while preserving aspect ratio.
2. RGB to grayscale and histogram equalization.
3. Gaussian blur.
4. Adaptive thresholding with Otsu fallback for low-signal cases.
5. Morphological opening.
6. Skeletonization (`skimage.morphology.skeletonize`).

This stage is the dominant upstream determinant of graph quality for all structural methods.

### 3.2 Gauss/PD Extraction
`gauss_pd.py` transforms skeleton pixels into a graph and computes symbolic codes:
1. Pixel graph construction (`4` or `8` connectivity).
2. Spur pruning to suppress short dead-end branches.
3. Junction clustering by degree threshold.
4. Path simplification into a multigraph between terminal entities.
5. Crossing edge pairing via local angular opposition.
6. Traversal to recover crossing order and PD-like local arc order.

This is a heuristic extractor, not a complete knot-theoretic invariant computation.

### 3.3 Reidemeister Move Candidate Detector
`reidemeister.py` estimates move candidates in drawing space:
- R1: cycle basis filtering by size and junction density.
- R2: multi-edge near-parallel structures in simplified graph.
- R3: triangular junction motifs with perimeter and edge-ratio constraints.
- Optional geometric re-scoring and NMS for candidate consolidation.

`ReidemeisterConfig.synthetic_defaults()` is used for synthetic fixture evaluation.

### 3.4 Diagram Reducer + Classifier (Solver)
`solver.py` applies iterative structural simplification before optional classification:
1. Skeleton to graph (`connectivity=4` by default for stability).
2. Iterative R1-like and R2-like removals (`max_iters` bounded).
3. Spur pruning after each reduction step.
4. Re-extraction of Gauss/PD on the reduced skeleton.
5. Optional CNN prediction on the original image, returning reduction stats.

The reducer returns `crossings_before/after`, `r1_removed`, and `r2_removed` for transparent auditability.

### 3.5 Protein Projection Pipeline
`protein.py` and dataset scripts implement a projection surrogate:
1. Parse PDB ATOM records and extract C-alpha polyline.
2. Sample view directions on the sphere (Fibonacci lattice via `sample_viewpoints`).
3. Project 3D polyline to 2D with depth retained (`project_polyline`).
4. Detect segment intersections in projection (`detect_crossings`).
5. Assign over/under using interpolated depth at intersection.
6. Build crossing-event sequence with over/under passage signs (`gauss_code_from_crossings`): returns signed crossing encounter tokens ($\pm k$, where $+k$ marks over-strand passage and $-k$ marks under-strand passage). This represents an over/under encounter sequence rather than classical topological crossing chirality (handedness).

The hybrid dataset encodes each view as:
- rasterized projection image
- Gauss-code histogram features
- class label
- protein key (group id for split hygiene)

## 4. Training and Evaluation Protocol
### 4.1 Image Classifier
`train.py` supports:
- deterministic seeding (`random`, `numpy`, `torch`)
- random or stratified train/val split
- evaluation protocol: training uses random augmentations (flips, rotations); validation uses deterministic unaugmented transforms (resize, ToTensor)
- checkpoint metadata (`epoch`, `optimizer_state`, `train_config`, metrics)

### 4.2 Protein Hybrid Classifier
`train_hybrid_classifier.py` supports:
- group-stratified split by protein key (preventing view leakage across train/validation)
- Adam optimizer with cross-entropy loss
- training and validation curves, confusion matrix, and per-class plots
- `--eval-only` flag to evaluate saved checkpoints without retraining

## 5. Data Contracts
### 5.1 2D Drawing Dataset
Expected directory layout:
```text
data_root/
  3_1/
  4_1/
  ...
```
Folder name is interpreted as class label.

### 5.2 Mapping File
`mapping_example.csv` schema:
```text
label,pd_code,gauss_code
```

### 5.3 Protein Hybrid Dataset
`data/knotprot/hybrid_dataset.npz` fields:
- `images`: `N x H x W` (`uint8`)
- `gauss_feats`: `N x F` (`float32`)
- `labels`: `N` class labels
- `keys`: `N` protein identifiers (if available)

`data/knotprot/hybrid_manifest.csv`:
```text
key,view_id
```

## 6. Current Measured Results (Workspace Artifacts)
This section summarizes values audited directly against committed artifacts:

### 6.1 Hybrid Baseline (20 epochs)
> **Provenance & Git Commit Record**: The metrics recorded in `results/hybrid_training.csv` originate from commit `b3484d9` ("Leak-free protein split training outputs"). The committed CSV records 20 epochs with `epoch,train_loss,train_acc,val_loss,val_acc`; it does not record batch size, learning rate, or random seed in its file metadata.

Audited directly from historical artifact `results/hybrid_training.csv` (commit `b3484d9`):
- Terminal epoch 20 values:
  - `train_acc = 0.919615`
  - `val_acc = 0.901677`
  - `train_loss = 0.338764`
  - `val_loss = 0.434658`
- Best validation accuracy: `0.902439` (reached at epoch 1 and sustained through epoch 19).
- Behavior: early accuracy saturation on the validation split with plateaued cross-entropy loss.


## 7. Reproducibility Checklist & Recommendations
1. Use a clean virtual environment and pinned direct dependencies (`pyproject.toml` / `requirements.txt`).
2. Use fixed seeds for all stochastic components.
3. For protein experiments, split by protein key rather than sample index.
4. Current workspace artifacts: The trainer writes model checkpoints (`checkpoints/*.pth`), training metric logs (`results/*.csv`), and confusion matrix plots (`results/*.png`).
5. *Recommended for future runs*: Implement and archive a structured split-metrics JSON (recording exact train/val key partition splits, per-class confusion matrices, and macro-F1 diagnostics); current scripts produce CSV logs and visualization plots rather than split JSON files.
6. Report per-class metrics in addition to top-1 accuracy for imbalanced regimes.

## 8. Known Limitations
1. Gauss/PD extraction is heuristic and sensitive to binarization artifacts.
2. Reidemeister detector returns candidates, not formal move proofs.
3. Chirality inference in `infer.py` is confidence-difference based, not invariant-based.
4. Protein projection approach approximates topology; near-depth ties can flip over/under assignment (+k vs -k encounter token).
5. High aggregate accuracy can conceal minority-class failure under severe class imbalance.

## 9. Recommended Scientific Extensions
1. Add invariant-level baselines (e.g., Alexander polynomial features) for explicit comparison.
2. Add macro-F1 and balanced accuracy to all hybrid training logs.
3. Run grouped cross-validation over protein keys, not a single split.
4. Compare solver reductions against manually annotated Reidemeister edits for calibration.
5. Add uncertainty calibration (temperature scaling) for classifier outputs.
