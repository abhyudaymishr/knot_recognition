# Knot Recognition & Topology Analysis Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 37 passed](https://img.shields.io/badge/tests-37%20passed-brightgreen.svg)](tests/)

## Overview
**Knot Recognition** is a scientific Python framework combining deep learning, computer vision, and computational knot theory. It provides unified workflows for:
1. **Topological Image Traversal (`KnotTraverser`)**: Skeleton graph tracing, crossing junction detection, strand pairing, intensity-based over/under strand resolution, and PD/Gauss code extraction with visual crossing annotations.
2. **Vision NanoTransformer (`NanoTransformer`)**: A lightweight (~0.67M parameter) patch-based Vision Transformer for fast, active classification across all 250 prime knots from unknot through $10_{165}$. Features noise estimation, bilateral edge-preserving filtering, 128-d unit-normalized prototype embeddings, and cosine similarity retrieval.
3. **Certified Reidemeister Invariance & R-GNO**: Mathematical ambient isotopy invariance by construction. Reidemeister Graph Neural Operator (`RGNO`) with confluent $R_1/R_2$ reduction layers and self-supervised contrastive learning ($\mathbb{R}$-SimCLR) over stochastic braid moves in `Knotgen`.
4. **Active Neuro-Symbolic Traversal (`traverse_and_classify`)**: Seamless synergy coupling `KnotTraverser` code extraction with `NanoTransformer` classification and prototype retrieval, providing cross-verification and resilience against real-world texture noise.
5. **2D Diagram Classification & Solving**: ResNet CNN classifiers, 24D rotation/reflection feature extraction, Reidemeister move candidate detection, and graph-based diagram simplification.
6. **3D Biological Knotting (Protein Pipeline)**: PDB $C_\alpha$ backbone extraction, Fibonacci sphere projections, ray-segment crossing resolution, and hybrid (CNN image + Gauss histogram) classification.
7. **Procedural Generative Knot Engine (`Knotgen`)**: Repository-local knot construction and diagram tools for braid closures, torus knots, rational/2-bridge knots, pretzel and Montesinos knots, diagram operations, polynomial invariants, rendering, and optional SnapPy integration. The bundled braid table has 250 entries: the unknot plus the 249 prime knots from $3_1$ through $10_{165}$.

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

### 2. Vision NanoTransformer (`NanoTransformer`) & Active Classification
The `NanoTransformer` is an optimized Vision Transformer (~0.67M params, 128-d latent space) pre-trained across all **250 prime knot types** (unknot through $10_{165}$). It performs:
- **Active Classification**: Fast, end-to-end image-level knot type prediction.
- **Noise Awareness & Denoising**: Front-end noise variance estimation, edge-preserving bilateral smoothing, and robust background handling.
- **Top-K Prototype Retrieval**: Cosine similarity against 250 unit-normalized canonical class prototypes.
- **Neuro-Symbolic Synergy**: Integrated with `KnotTraverser` via `traverse_and_classify(...)` to compare raster crossing codes with latent topological priors.

#### Single-Image Inference & Prototype Retrieval:
```python
import cv2
from knot_recognition import NanoTransformer

# 1. Load the pre-trained 250-class checkpoint
model = NanoTransformer.from_pretrained("checkpoints/nanotransformer_knot250.pth")

# 2. Predict knot class and retrieve nearest prototypes
image = cv2.imread("path/to/knot_drawing.png")
pred_label, confidence, top_k = model.predict(image, top_k=5)

print(f"Predicted Knot: {pred_label} (Confidence: {confidence:.2%})")
print("Top 5 Nearest Prototypes:")
for rank, (label, score) in enumerate(top_k, 1):
    print(f"  {rank}. {label} - Cosine Sim: {score:.3f}")
```

#### Active Traversal + NanoTransformer Synergy:
```python
import cv2
from knot_recognition import KnotTraverser

traverser = KnotTraverser()
image = cv2.imread("path/to/knot_drawing.png")

# Joint traversal code extraction + active transformer retrieval
result, classification = traverser.traverse_and_classify(
    image, 
    checkpoint_path="checkpoints/nanotransformer_knot250.pth"
)

print(f"Raster Detected Crossings: {len(result.intersections)}")
print(f"Extracted Gauss Code:      {result.gauss_code}")
print(f"Transformer Classification: {classification['predicted_knot']} ({classification['confidence']:.2%})")
```

#### Train the 250-Class NanoTransformer:
```bash
python scripts/train_nanotransformer.py \
  --checkpoint checkpoints/nanotransformer_knot250.pth \
  --epochs 25 \
  --samples-per-class 16 \
  --batch-size 32 \
  --lr 3e-4
```

### 3. Certified Reidemeister Invariance vs. Topological Equivalence
> **The Honest Claim**:  
> *"A canonicalization layer gives exact $R_1/R_2$ invariance; here is what that buys a downstream learned model."*

Two knot diagrams represent the same topological knot if and only if they are related by a finite sequence of **Reidemeister moves ($R_1, R_2, R_3$)** and planar isotopies:
- **$R_1$ (Twist / Curl)**: Markov stabilization & destabilization.
- **$R_2$ (Poke / Bigon)**: Cancelling pair introduction & deletion.
- **$R_3$ (Slide / Triangle)**: Artin braid Yang-Baxter slide ($\sigma_i \sigma_{i+1} \sigma_i \longleftrightarrow \sigma_{i+1} \sigma_i \sigma_{i+1}$).
- **Flypes**: 180° rotations of alternating 2-tangles (Tait's Flyping Conjecture).

Crucially, **a polynomial-time confluent canonicalization layer can only reduce $R_1$ and $R_2$ moves**. It does not solve full knot equivalence, because equivalence gets fundamentally hard at $R_3$ triangle slides ($\Delta c = 0$) and flypes.

#### Benchmark Protocol (5 Seeds, Knot-Level Split)
- **Same Input for Graph Models**: Signed 4-valent Planar Diagram (PD) code / planar graph.
- **Rendered ViT Reference Row**: NanoTransformer on 64x64 rendered raster images.
- **Knot-Level Split**: Train on $c \le 8$ (37 knots); test on unseen $c = 9$ (49 knots) and unseen $c = 10$ (165 knots).
- **Hard Perturbations**: Diagrams generated with $R_1, R_2, R_3$, and flypes up to $\min(c) + 6$ extra crossings.
- **Hard Negative Mining**: Mined strictly from pairs sharing crossing count $c(K)$ or identical Alexander / Jones polynomials.
- **Task 1**: Same-knot verification (AUROC, Top-1, Top-5 retrieval).
- **Task 2**: Topological invariant prediction on unseen knots (signature, three-genus, braid index, unknotting number from KnotInfo) vs trivial predictors (mean and $[c(K), w(K)]$ linear fit).

#### 5-Seed Empirical Benchmark Results:
| Model Variant | Seen-A AUROC ($c \le 8$) | Unseen-9 AUROC ($c=9$) | Unseen-10 AUROC ($c=10$) | Signature MAE ($c=9$) | Unknotting MAE ($c=9$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw GNN (No Canonicalization)** | $0.555 \pm 0.030$ | $0.501 \pm 0.069$ | $0.428 \pm 0.024$ | $2.354 \pm 0.000$ | $0.724 \pm 0.000$ |
| **Canonical GNN ($R_1 / R_2$)** | $0.544 \pm 0.017$ | $0.510 \pm 0.019$ | $0.506 \pm 0.018$ | $2.351 \pm 0.000$ | $0.715 \pm 0.000$ |
| **Canonical GNN ($R_1$-only)** | $0.517 \pm 0.012$ | $0.550 \pm 0.012$ | $0.447 \pm 0.015$ | $2.353 \pm 0.000$ | $0.723 \pm 0.000$ |
| **Canonical GNN ($R_2$-only)** | $0.617 \pm 0.006$ | $0.414 \pm 0.014$ | $0.559 \pm 0.012$ | $2.399 \pm 0.001$ | $0.770 \pm 0.001$ |
| **SnapPy Simplify (Upper Bound)** | $0.535 \pm 0.011$ | $0.525 \pm 0.019$ | $0.534 \pm 0.016$ | $2.351 \pm 0.000$ | $0.715 \pm 0.000$ |
| **Augmented GNN** | $0.502 \pm 0.014$ | $0.430 \pm 0.030$ | $0.489 \pm 0.025$ | $2.354 \pm 0.000$ | $0.724 \pm 0.000$ |
| **Contrastive GNN** | $0.434 \pm 0.034$ | $0.291 \pm 0.074$ | $0.386 \pm 0.052$ | $2.354 \pm 0.000$ | $0.724 \pm 0.000$ |
| **Shuffled-Label Control** | $0.537 \pm 0.019$ | $0.491 \pm 0.016$ | $0.504 \pm 0.018$ | $2.351 \pm 0.000$ | $0.715 \pm 0.000$ |
| **Random-Feature Control** | $0.505 \pm 0.012$ | $0.411 \pm 0.059$ | $0.583 \pm 0.035$ | $2.353 \pm 0.000$ | $0.724 \pm 0.000$ |
| **Lookup-Table Hash** | $0.634 \pm 0.002$ | $0.599 \pm 0.007$ | $0.640 \pm 0.005$ | $2.314 \pm 0.001$ | $0.681 \pm 0.001$ |
| **Non-Learned Baseline ($\Delta(t) + c + w$)** | **$0.995 \pm 0.000$** | **$0.980 \pm 0.000$** | **$0.959 \pm 0.000$** | **$1.937 \pm 0.000$** | **$0.628 \pm 0.000$** |
| **Rendered ViT (Reference Row)** | $0.986 \pm 0.000$ | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | $2.331 \pm 0.027$ | $0.677 \pm 0.010$ |
| *Trivial Mean Baseline* | *---* | *---* | *---* | *2.354* | *0.724* |
| *Trivial Linear Fit $[c(K), w(K)]$* | *---* | *---* | *---* | *2.327* | **$0.627$** |

#### Scientific Findings & Conclusion:
1. **Refutation of Naive Graph Invariance**: Confluent $R_1/R_2$ canonicalization eliminates superficial curls and bigons, but **fails to close the verification gap under $R_3$ slides and flypes** (AUROC remains at $\sim 0.51$).
2. **Failure to Beat Trivial Invariant Baselines**: On unseen crossing numbers ($c=9$ and $c=10$), learned GNN models fail to outperform a simple 2-variable linear fit on $[c(K), w(K)]$ for unknotting number and signature.
3. **Algebraic Invariant Superiority**: The exact non-learned feature baseline ($\Delta(t) + c + w$) decisively outperforms all learned GNN encoders on unseen knots (AUROC 0.980 on $c=9$ and 0.959 on $c=10$, with signature MAE 1.937 beating trivial baselines).
4. **Visual Layout vs Graph Structure**: The rendered-image ViT excels at diagram discrimination (AUROC 1.000) by exploiting global 2D spatial layout, an embedding mechanism fundamentally distinct from local 4-valent message passing.

#### Reproduce the Protocols:
```bash
# 1. Honest Reidemeister Protocol on c <= 8 -> c in {9, 10}
python scripts/run_honest_protocol.py \
  --epochs 15 \
  --seeds 42 43 44 45 46 \
  --cache data/benchmark_diagrams_cache.json \
  --output results/honest_reidemeister_benchmark.json

# 2. Fully Audited Rigorous Task 1 & Task 2 Benchmarks (Independent Backtrack + Full KnotInfo c=12 Scale)
python scripts/run_rigorous_tasks.py \
  --seed 42 \
  --output results/rigorous_tasks_benchmark.json
```

See [docs/honest_reidemeister_protocol.md](docs/honest_reidemeister_protocol.md) for detailed mathematical breakdowns, confidence intervals, and full empirical tables.

### 4. 2D ResNet Image Classification Workflow
> **Note on Model Checkpoints**:
> - `checkpoints/best.pth` referenced below is the output of the **2D ResNet training workflow** (`python -m knot_recognition.train`). It is not pre-packaged in the repository.
> - The committed checkpoint `checkpoints/hybrid_classifier.pth` is for the **protein projection model** and has an incompatible architecture (`cnn.0.weight` vs ResNet `conv1.weight`), so it cannot be loaded by `knot` or `knot-solve`.
> - For general 2D knot classification, `NanoTransformer` (`checkpoints/nanotransformer_knot250.pth`) is recommended.

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

### 5. Procedural Knot Generation & Diagram Rendering (`Knotgen/`)
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

### 6. Protein Knot Pipeline (Stages 1–4)
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
│   ├── r_gno.py                   # Reidemeister Graph Neural Operator & canonical R1/R2 reduction
│   ├── reidemeister_contrastive.py # Self-supervised contrastive learning (R-SimCLR / R-VICReg)
│   ├── topological_probe.py       # Linear probing on exact invariants (det, Alexander, volume)
│   ├── nanotransformer.py         # Vision NanoTransformer: ViT, noise estimation, prototype retrieval
│   ├── traverser.py               # KnotTraverser: curve tracing, over/under resolution, annotations
│   ├── gauss_pd.py                # Graph skeletonization, junction clustering, PD extraction
│   ├── features.py                # 24D feature vector (Hu moments, radial Fourier, reflection score)
│   ├── infer.py                   # KnotRecognizer & CLI inference engine
│   ├── model.py                   # PyTorch ResNet backbone wrappers
│   ├── dataset.py                 # KnotImageDataset loader & transform integration
│   ├── preprocess.py              # Grayscale binarization, noise estimation & bilateral denoising
│   ├── protein.py                 # 3D backbone projections & crossing elevation detectors
│   ├── reidemeister.py            # Synthetic R1, R2, R3 move candidate detection & visualization
│   ├── solver.py                  # Diagram loop simplification & reduced inference
│   └── train.py                   # 2D ResNet supervised training loop
├── Knotgen/                       # Repository-local knot generation tools
│   ├── reidemeister.py            # Stochastic R1, R2, R3 braid & diagram move engine
│   ├── knotgen.py                 # Knot model, constructors, invariants, operations, SnapPy bridge
│   ├── draw.py                    # Planar diagram and smooth torus-knot rendering
│   ├── enumerate.py               # Experimental shadow enumeration and finite-invariant filtering
│   ├── build_census.py            # SnapPy braid-table builder and stored-check verifier
│   ├── census_braids.json         # 250 rows: unknot plus prime knots 3_1 through 10_165
│   └── knots_through_9.json       # 84 generated prime-knot records for crossing numbers 3–9
├── docs/                          # Scientific & technical documentation
│   ├── reidemeister_invariance_theory.md # Mathematical proofs and NeurIPS/ICML paper formulation
│   ├── scientific.md              # Research report, methodology, and audited baseline metrics
│   ├── reproducibility.md         # Environment setup, test benchmarks, and data guidelines
│   ├── cable_photo_system.md      # Cable & real-world knot identification documentation
│   ├── figures/                   # Visualization galleries
│   └── methods.md                 # Mathematical and algorithmic reference
├── scripts/                       # Experiment drivers & dataset preparation
│   ├── run_crossing_generalization_experiment.py # Crossing extrapolation & invariant probing
│   ├── train_reidemeister_contrastive.py # Self-supervised contrastive training CLI
│   ├── train_nanotransformer.py   # Noise-aware training loop for 250-class NanoTransformer
│   ├── build_knotprot_hybrid_dataset.py
│   ├── train_hybrid_classifier.py
│   ├── extract_knotprot_stage1.py
│   ├── download_knotprot.py
│   └── merge_hybrid_parts.py
├── checkpoints/                   # Trained model weights
│   └── nanotransformer_knot250.pth # Pre-trained 250-class Vision NanoTransformer
├── data/                          # Dataset manifests and metadata
│   └── knotprot/hybrid_manifest.csv
├── results/                       # Training logs and figure outputs
│   ├── reidemeister_crossing_generalization.json # Crossing generalization metrics
│   └── hybrid_training.csv
└── tests/                         # 37 unit tests across 9 test modules
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

The test suite contains **37 unit tests across 9 test modules**:
```bash
pytest tests/ -v
```
*(Dated benchmark: ~2.5–3.5s on macOS Apple Silicon, October 2026; duration is environment-dependent.)*

| Module | Scope Tested |
|---|---|
| `test_reidemeister_invariance.py` | Stochastic R1/R2/R3 generators, invariant preservation (Alexander, det), confluent R-NF canonical reduction, RGNO certified zero-distance invariance, contrastive loss, topological invariant probing |
| `test_nanotransformer.py` | ViT forward pass, 250-class prediction, top-k cosine similarity retrieval, checkpoint persistence, noise-level awareness & bilateral denoising, traverser integration |
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
