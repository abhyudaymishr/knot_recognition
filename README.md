# Knot Recognition & Topology Analysis Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 24 passed](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)](tests/)

## Overview
**Knot Recognition** is a scientific Python framework combining deep learning, computer vision, and computational knot theory. It provides unified workflows for:
1. **Topological Image Traversal (`KnotTraverser`)**: Skeleton graph tracing, crossing junction detection, strand pairing, intensity-based over/under strand resolution, and PD/Gauss code extraction with visual crossing annotations.
2. **2D Diagram Classification & Solving**: ResNet CNN classifiers, 24D rotation/reflection feature extraction, Reidemeister move candidate detection, and graph-based diagram simplification.
3. **3D Biological Knotting (Protein Pipeline)**: PDB $C_\alpha$ backbone extraction, Fibonacci sphere projections, ray-segment crossing resolution, and hybrid (CNN image + Gauss histogram) classification.
4. **Procedural Generative Knot Engine (`Knotgen`)**: Repository-local knot construction and diagram tools for braid closures, torus knots, rational/2-bridge knots, pretzel and Montesinos knots, diagram operations, polynomial invariants, rendering, and optional SnapPy integration. The bundled braid table has 250 entries: the unknot plus the 249 prime knots from $3_1$ through $10_{165}$.

---

## Installation

Install in editable development mode from the repository root:
```bash
pip install -e .
```

### Optional Dependencies
- **Testing**:
  ```bash
  pip install -e ".[test]"
  ```
- **Hyperbolic Geometry (`SnapPy`)**: Required for 3-manifold volume and census isometry verification:
  ```bash
  pip install -e ".[snappy]"
  ```
  *(Note: SnapPy is optional; basic knot generation, census table loading, and the core test suite run without it.)*

---

## Quickstart Workflows

### 1. Raster Image Traversal & Topological Code Extraction (`KnotTraverser`)
The `KnotTraverser` analyzes raster knot drawings (grayscale or RGB), identifies strand intersections, determines strand depth via orthogonal intensity profiles, and outputs topological planar diagram (PD) and Gauss codes along with marked visual annotations:

```python
import cv2
from knot_recognition import KnotTraverser

# 1. Initialize traverser
traverser = KnotTraverser(cluster_radius=12, sample_radius=18)

# 2. Process an input image (grayscale or RGB knot drawing)
image = cv2.imread("path/to/knot_drawing.png")
result = traverser.process_image(image)

# 3. Access detected topological codes
print(f"Detected Crossings: {len(result.intersections)}")
print(f"Gauss Code:         {result.gauss_code}")
print(f"PD Code:            {result.pd_string}")

# 4. Save or display the annotated image with over/under strands marked
cv2.imwrite("annotated_knot.png", result.annotated_image)
```

### 2. 2D ResNet Image Classification Workflow
> **Note on Model Checkpoints**:
> - `checkpoints/best.pth` referenced below is the output of the **2D ResNet training workflow** (`python -m knot_recognition.train`). It is not pre-packaged in the repository.
> - The committed checkpoint `checkpoints/hybrid_classifier.pth` is for the **protein projection model** and has an incompatible architecture (`cnn.0.weight` vs ResNet `conv1.weight`), so it cannot be loaded by `knot` or `knot-solve`.

#### Train a 2D ResNet:
```bash
python -m knot_recognition.train \
  --data-dir /path/to/dataset_root \
  --outdir ./checkpoints \
  --epochs 20 --batch 32 --lr 1e-3
```

#### Run Inference via CLI:
```bash
# Standard inference
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv

# With 24D symmetry features
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv --features

# Force CPU execution
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --device cpu
```

#### Detect Reidemeister Move Candidates:
```bash
knot-moves --image /path/to/image.png --overlay results/figures/moves_overlay.png
```

#### Diagram Reducer + Classifier (Solver):
```bash
knot-solve --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```

---

### 3. Procedural Knot Generation & Diagram Rendering (`Knotgen/`)
`Knotgen` provides braid-based knot constructors, PD/Gauss representations, Alexander and Jones polynomials, determinants, Burau calculations, common knot operations, and rendering helpers. `Knot.render_image()` returns an RGB `uint8` array and can also save it to a file. `render_torus_smooth()` renders a projected 3D torus-knot embedding; it is a visualization, not a canonical diagram encoding.

**Import scope:** `Knotgen` is currently a repository-local package and is not included by the `src/`-only package discovery in `pyproject.toml`. Run these examples from the repository root (or add the checkout root to `PYTHONPATH`). `pip install -e .` installs `knot_recognition`, but by itself does not make `import Knotgen` available outside the checkout.

```python
from Knotgen import braid, torus, rational, pretzel, load_census

# 1. Torus knot T(2, 3) (Trefoil) & Image Rendering
trefoil = torus(2, 3)
print("Alexander Polynomial:", trefoil.alexander())
print("Determinant:", trefoil.determinant())

# Render to PNG image with over/under crossings
img = trefoil.render_image("trefoil_diagram.png", size=(512, 512), line_width=4)
print("Rendered diagram array shape:", img.shape)

# 2. Rational 2-bridge knot 4_1 (continued fraction [2, 2])
figure_eight = rational([2, 2])
print("4_1 Determinant:", figure_eight.determinant())
figure_eight.render_image("figure_eight.png")

# 3. Artin Braid closure & Burau matrix
b = braid([1, -2, 1, -2], nstrands=3)
print("Braid PD Code:", b.pd_string())

# 4. Load the bundled table: "unknot" plus 249 prime knots (3_1 through 10_165)
census = load_census()
print(f"Loaded {len(census)} census knot representations.")
print(census["10_165"]["verified"], census["10_165"]["check"])
```

The `verified` and `check` fields above are records stored in `Knotgen/census_braids.json`; `load_census()` only reads the JSON and does **not** rerun the checks. The builder uses SnapPy: it compares hyperbolic braid-closure exteriors by isometry and checks the listed torus-knot entries against Alexander polynomials. Rebuilding this table requires the optional SnapPy dependencies:

```bash
pip install -e ".[snappy]"
python Knotgen/build_census.py
```

The separate `Knotgen/knots_through_9.json` file contains 84 generated prime-knot records for crossing numbers 3–9. The from-scratch enumerator writes a new JSON file and uses polynomial/module fingerprints; its own source documents the risk that distinct knots sharing those finite fingerprints could be merged. Treat its output as a research enumeration, not as an independent proof of census completeness:

```bash
python Knotgen/enumerate.py --max 8 --workers 4 --out /tmp/knots_through_8.json
```

The `Knotgen/knotgen.py --selftest` command also exercises SnapPy identification and volume checks, so it requires SnapPy even though basic constructors and polynomial calculations do not.

---

### 4. Protein Knot Pipeline (Stages 1–4)
The biological pipeline models knotting in 3D protein macromolecules (such as KnotProt chains).

#### Prerequisites:
- External chain and label metadata:
  - `data/knotprot/knotprot_chains.csv` (`pdb_id,chain_id`)
  - `data/knotprot/knotprot_labels.csv` (`pdb_id,chain_id,knot_type`)
- PDB coordinate files in `data/knotprot/pdb/` (downloaded via `python scripts/download_knotprot.py`).

#### Execution:
**Stage 1: Extract Cα Backbones**:
```bash
python scripts/extract_knotprot_stage1.py \
  --pdb-dir data/knotprot/pdb \
  --chains-csv data/knotprot/knotprot_chains.csv \
  --out data/knotprot/backbones.npz \
  --manifest data/knotprot/backbones.csv
```

**Stages 2 & 3: Viewpoint Projections & Crossing Detection**:
- Spherical viewpoints via `sample_viewpoints` (Fibonacci lattice)
- Planar projection via `project_polyline`
- Ray intersection and depth-based over/under classification via `detect_crossings`
- Over/under passage event sequence via `gauss_code_from_crossings`

**Stage 4: Hybrid ML Classifier**:

##### Option A: Quick local experiment (subset training)
```bash
# 1. Build a 20-chain subset:
python scripts/build_knotprot_hybrid_dataset.py \
  --backbones data/knotprot/backbones.npz \
  --labels data/knotprot/knotprot_labels.csv \
  --viewpoints 32 --limit 20 --stride 3 --max-points 300 \
  --out data/knotprot/my_subset_dataset.npz \
  --manifest data/knotprot/my_subset_manifest.csv

# 2. Train a custom model on this subset:
python scripts/train_hybrid_classifier.py \
  --data data/knotprot/my_subset_dataset.npz \
  --out checkpoints/my_subset_classifier.pth \
  --epochs 5 --batch 64 --lr 1e-3
```

##### Option B: Regenerating & evaluating the committed 4-class checkpoint
> **Note on Provenance & Evaluation Scope**:
> - `data/knotprot/hybrid_manifest.csv` records chain and projection viewpoint identifiers (`key,view_id`); class labels reside in `hybrid_dataset.npz`, while class-to-index mappings reside in the dataset and checkpoint dictionary (`class_to_idx`).
> - The 200-chain training provenance (6,400 samples across classes `+3.1`, `-3.1`, `-5.2`, `4.1`) is reported provenance derived from `data/knotprot/knotprot_chains.csv` and `data/knotprot/hybrid_manifest.csv`, as `checkpoints/hybrid_classifier.pth` stores only `model_state` and `class_to_idx` without embedded split indices or dataset hashes.
> - The `--eval-only` command evaluates on the regenerated dataset's current holdout split (recomputed dynamically via group-stratified split using `--seed`, defaulting to 7). It verifies class-to-index schema compatibility, but does not verify identical chain partitions to the historical run unless explicit split metadata is preserved.

```bash
# 1. Generate the complete matching 4-class dataset:
python scripts/build_knotprot_hybrid_dataset.py \
  --backbones data/knotprot/backbones.npz \
  --labels data/knotprot/knotprot_labels.csv \
  --viewpoints 32 --stride 3 --max-points 300 \
  --out data/knotprot/hybrid_dataset.npz \
  --manifest data/knotprot/hybrid_manifest.csv

# 2. Evaluate the committed checkpoint without overwriting weights or logs:
python scripts/train_hybrid_classifier.py \
  --data data/knotprot/hybrid_dataset.npz \
  --out checkpoints/hybrid_classifier.pth \
  --eval-only
```

---

## Project Structure

```text
knot_recognition/
├── src/knot_recognition/          # Core Python package
│   ├── traverser.py               # KnotTraverser: curve tracing, over/under resolution, annotations
│   ├── gauss_pd.py                # Graph skeletonization, junction clustering, PD extraction
│   ├── features.py                # 24D feature vector (Hu moments, radial Fourier, reflection score)
│   ├── infer.py                   # KnotRecognizer & CLI inference engine
│   ├── model.py                   # PyTorch ResNet backbone wrappers
│   ├── dataset.py                 # KnotImageDataset loader & transform integration
│   ├── preprocess.py              # Grayscale binarization & skimage morphological skeletonization
│   ├── protein.py                 # 3D backbone projections & crossing elevation detectors
│   ├── reidemeister.py            # Synthetic R1, R2, R3 move candidate detection & visualization
│   ├── solver.py                  # Diagram loop simplification & reduced inference
│   └── train.py                   # 2D ResNet supervised training loop
├── Knotgen/                       # Repository-local knot generation tools
│   ├── knotgen.py                 # Knot model, constructors, invariants, operations, SnapPy bridge
│   ├── draw.py                    # Planar diagram and smooth torus-knot rendering
│   ├── enumerate.py               # Experimental shadow enumeration and finite-invariant filtering
│   ├── build_census.py            # SnapPy braid-table builder and stored-check verifier
│   ├── census_braids.json         # 250 rows: unknot plus prime knots 3_1 through 10_165
│   └── knots_through_9.json       # 84 generated prime-knot records for crossing numbers 3–9
├── docs/                          # Scientific & technical documentation
│   ├── scientific.md              # Research report, methodology, and audited baseline metrics
│   ├── reproducibility.md         # Environment setup, test benchmarks, and data guidelines
│   ├── figures/                   # Visualization galleries
│   └── methods.md                 # Mathematical and algorithmic reference
├── scripts/                       # Experiment drivers & dataset preparation
│   ├── build_knotprot_hybrid_dataset.py
│   ├── train_hybrid_classifier.py
│   ├── extract_knotprot_stage1.py
│   ├── download_knotprot.py
│   └── merge_hybrid_parts.py
├── data/                          # Dataset manifests and metadata
│   └── knotprot/hybrid_manifest.csv
├── results/                       # Training logs and figure outputs
│   └── hybrid_training.csv
└── tests/                         # 24 unit tests across 7 test modules
```

---

## Output Schemas & Feature Conventions

### 1. Inference Result Schema (`infer.py`)
`KnotRecognizer.infer_image()` returns a structured dictionary containing independent subsystem outputs:
- `predicted_label` (`str`): Predicted knot class from the CNN.
- `pred_prob` (`float`): Softmax confidence score for the winning class.
- `mapping_pd` (`str` | `None`): Reference PD code from CSV lookup if provided.
- `mapping_gauss` (`str` | `None`): Reference Gauss code from CSV lookup if provided.
- `auto_gauss` (`str` | `None`): Heuristic Gauss-like code extracted from the skeleton graph.
- `auto_pd` (`str` | `None`): Heuristic PD-like code extracted from the skeleton graph.
- `features` (`list[float]` | `None`): 24D symmetry and radial feature vector (if `--features` is enabled).
- `chirality` (`str`): Reflection-based heuristic chirality prediction (`left-handed`, `right-handed`, `achiral`).
- `chirality_confidence` (`float`): Margin difference between original and horizontally flipped inferences.

### 2. 24D Image Feature Representation (`features.py`)
- **7 Hu Moments** (`hu_moments`): Log-transformed central moments capturing 2D planar shape.
- **16 Radial Fourier Magnitudes** (`radial_fourier_descriptor`): $\ell_2$-normalized FFT magnitudes of 1D radial intensity averages from the image center.
- **1 Horizontal Reflection Symmetry Score** (`reflection_symmetry_score`): Normalized cross-correlation with `np.fliplr(img)`.

### 3. Topological Code Conventions
- **Classical Signed Gauss Codes**: Encounter sequence following a single knot loop, where $+1/-1$ indicates topological crossing handedness/chirality.
- **Protein Over/Under Encounter Sequences**: Signed tokens $+k$ and $-k$ signify whether the backbone passes **OVER** ($+k$) or **UNDER** ($-k$) at crossing $k$. This encodes spatial elevation along the chain rather than crossing chirality.

---

## Verification & Test Suite

The test suite contains **24 unit tests across 7 test modules**:
```bash
pytest tests/ -v
```
*(Dated benchmark: ~2.5–3.5s on macOS Apple Silicon, October 2026; duration is environment-dependent.)*

| Module | Scope Tested |
|---|---|
| `test_traverser.py` | Raster crossing detection, strand pairing, PD/Gauss code formatting, OpenCV and PIL compatibility, crossing continuity & gap classification |
| `test_knotgen.py` | Braids, Torus, Rational, Pretzel, Burau matrices, SnapPy integration, census table schema, procedural image rendering |
| `test_protein_stage2.py` | 3D backbone projections, parametric segment intersection, and depth sorting |
| `test_reidemeister_synthetic.py` | Synthetic R1 curl, R2 bigon, and R3 triangle move candidate detectors |
| `test_solver_reduction.py` | Graph reduction, Reidemeister loop removals, node cleanup |
| `test_gauss_pd.py` | Skeleton to graph, junction clustering, edge simplification, heuristic PD codes |
| `test_features.py` | 24D feature vector length, finiteness, and symmetry score range |

---

## Known Limitations
1. **Chirality Detection**: Reflection-based chirality inference is heuristic and evaluates CNN confidence deltas under image flips rather than formal chiral topological invariants.
2. **Gauss/PD Extraction**: The heuristic skeleton extractor in `gauss_pd.py` assumes clean, high-contrast drawings; image binarization noise or spur artifacts can impact crossing degree. For direct image analysis, `KnotTraverser` is recommended.
3. **Protein Depth Disambiguation**: Near-depth ties in 2D protein projections can flip over/under assignment ($+k$ vs $-k$) near boundary angles.

---

## Citation & Documentation
- For detailed scientific and algorithmic methodology, see [docs/scientific.md](docs/scientific.md) and [KNOT_RECOGNITION_ARCHITECTURE.md](KNOT_RECOGNITION_ARCHITECTURE.md).
- For reproducibility guidelines, see [docs/reproducibility.md](docs/reproducibility.md).
- To cite this work, please refer to [CITATION.cff](CITATION.cff).
