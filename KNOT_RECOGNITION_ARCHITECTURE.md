# Knot Recognition & Knotgen: Comprehensive Architectural Reference

A complete technical specification of the `knot_recognition` repository, detailing the dual-engine framework combining computer vision, deep learning, graph topology, 3D structural biology, and combinatorial knot generation.

---

## 1. System Overview & Architecture

The `knot_recognition` ecosystem unifies classical computational knot theory with modern computer vision and deep learning. It addresses three core challenges:

1. **Topological Extraction from Images**: Processing 2D raster line drawings of knots and extracting rigorous topological invariants (Planar Diagram (PD) codes and signed Gauss codes), resolving crossing overlaps and handedness.
2. **Topological Simplification & Invariance**: Detecting and applying Reidemeister moves ($R_1, R_2, R_3$) on skeletal graphs to simplify knot diagrams and evaluate topological equivalence.
3. **Generative Knot Engine (`Knotgen`)**: Layered procedural synthesis of all prime knots up to 10 crossings ($0_1 \to 10_{165}$), computing polynomial invariants (Alexander, Jones, Burau) and linking directly with hyperbolic 3-manifold geometry (SnapPy).
4. **Biophysical Knots (Protein Knots)**: Parsing 3D $C_\alpha$ backbones from PDB/KnotProt files, projecting onto arbitrary viewing spheres, solving ray intersections in $t, u \in [0, 1]$, and categorizing protein entanglement.

```
                           +-------------------------------------+
                           |            Input Domain             |
                           +-------------------------------------+
                           | 2D Line Drawings | 3D PDB Backbones |
                           +------------------+------------------+
                                    |                  |
                                    v                  v
                       +-----------------------+ +--------------------+
                       | preprocess.py         | | protein.py         |
                       | Otsu / Skeletonize    | | 3D Projection &    |
                       +-----------------------+ | Ray Intersection   |
                                    |            +--------------------+
                                    v                      |
                 +--------------------------------------+  |
                 | traverser.py / gauss_pd.py           |  |
                 | Junction Clustering (4-valent)       |  |
                 | Collinear Tangent Branch Pairing     |  |
                 | Over/Under Continuity Disambiguation |  |
                 | Cycle Traversal & Arc Ordering       |<-+
                 +--------------------------------------+
                                    |
          +-------------------------+-------------------------+
          v                                                   v
+-----------------------------+                 +-----------------------------+
| Topological Invariants      |                 | reidemeister.py / solver.py |
| - PD Code: PD[X[...], ...]  |                 | - R1 Curl Removal           |
| - Gauss Code: [1, -2, 3...] |                 | - R2 Bigon Elimination      |
| - Invariant Verification    |                 | - R3 Triangle Sliding       |
+-----------------------------+                 +-----------------------------+
          ^                                                   |
          |                                                   v
+-----------------------------+                 +-----------------------------+
| Knotgen/ Generator Engine   |                 | model.py / infer.py         |
| Braids, Torus, Rational,    |                 | ResNet Classification       |
| Pretzel, SnapPy, Census     |                 | Hybrid ML + Topological Feat|
+-----------------------------+                 +-----------------------------+
```

---

## 2. Directory Hierarchy

```
knot_recognition/
├── KNOT_RECOGNITION_ARCHITECTURE.md # (This document)
├── pyproject.toml                   # Project metadata, dependencies & build configuration
├── pytest.ini                       # Test suite runner configuration
├── requirements.txt                 # Pinned dependencies
├── generate_gauss_codes.py          # Batch Gauss code generator script
├── infer.py                         # Top-level CLI entry point for knot image inference
├── train.py                         # Top-level CLI entry point for neural network training
│
├── src/knot_recognition/            # Core Python package
│   ├── __init__.py                  # Public API exports
│   ├── __main__.py                  # python -m knot_recognition entry point
│   ├── traverser.py                 # Knot image traverser (crossing detection, PD & Gauss codes)
│   ├── gauss_pd.py                  # Skeletal graph extraction, junction clustering & cycles
│   ├── reidemeister.py              # Reidemeister move detection (R1, R2, R3) & multi-scale fusion
│   ├── solver.py                    # Iterative graph simplification solver via Reidemeister reductions
│   ├── protein.py                   # 3D protein backbone projections & ray-intersection depth solver
│   ├── features.py                  # Geometric, skeletal, Fourier & rotational symmetry features
│   ├── preprocess.py                # Grayscale, adaptive binarization & morphological thinning
│   ├── dataset.py                   # PyTorch KnotImageDataset implementation
│   ├── model.py                     # ResNet backbone architectures (ResNet-18 to ResNet-152)
│   ├── train.py                     # Training loop, checkpoints & validation metrics
│   ├── infer.py                     # Joint neural + topological knot recognizer engine
│   └── utils.py                     # File I/O helpers & safe image readers
│
├── Knotgen/                         # Generative knot framework
│   ├── __init__.py                  # Knotgen module exports
│   ├── knotgen.py                   # 6-layer knot generator & polynomial invariant computer
│   ├── build_census.py              # Automated builder & validator of 250 census braids
│   ├── enumerate.py                 # Planar 4-valent diagram generator via DT codes
│   ├── census_braids.json           # Validated braid representations for 250 Rolfsen/HTW knots
│   └── knots_through_9.json         # Dowker-Thistlethwaite table through 9 crossings
│
├── scripts/                         # Dataset preparation & experimental training scripts
│   ├── download_knotprot.py         # Automated KnotProt database downloader
│   ├── build_knotprot_hybrid_dataset.py # Hybrid synthetic + biophysical dataset compiler
│   ├── merge_hybrid_parts.py        # Dataset partition combiner
│   └── train_hybrid_classifier.py   # Hybrid model trainer (ResNet + topological features)
│
├── docs/                            # Research documentation & paper drafts
│   ├── scientific.md                # Scientific context & mathematical foundations
│   ├── methods.md                   # Formal methodology specification
│   └── reproducibility.md           # Instructions for reproducing experimental benchmarks
│
└── tests/                           # Complete test suite (21 unit tests, all passing)
    ├── test_traverser.py            # Image crossing traversal, pairing & PD code validation
    ├── test_knotgen.py              # Knotgen layers, invariants, census & SnapPy integration
    ├── test_protein_stage2.py       # 3D backbone projection & over/under crossing tests
    ├── test_reidemeister_synthetic.py# Synthetic R1, R2, R3 move detection benchmarks
    ├── test_solver_reduction.py     # Graph solver R1/R2 reduction validations
    ├── test_gauss_pd.py             # Skeleton-to-graph & junction clustering tests
    └── test_features.py             # Symmetry, Fourier & color feature vector tests
```

---

## 3. Detailed Subsystem Analysis (`src/knot_recognition/`)

### 3.1 `traverser.py` &mdash; Knot Image Traverser & PD/Gauss Code Synthesizer
* **Primary Classes**: `KnotTraverser`, `StrandIntersection`, `TraversalResult`
* **Purpose**: Performs continuous raster traversal over knot line drawings to extract crossings, distinguish over/under strands, number topological arcs, and output PD and Gauss codes.
* **Key Algorithmic Stages**:
  1. **Binarization & Thinning**: Converts input image to inverted grayscale and skeletonizes to a 1-pixel-wide line network.
  2. **Junction Detection**: Identifies pixels with transition counts $T(p) \ge 6$ or degree $\ge 4$, clustering them via spatial windowing into centroids $\mathbf{X}_k = (x_k, y_k)$.
  3. **Tangent Branch Sampling**: Samples a ring of radius $R_{\text{sample}}$ around each crossing to discover the 4 exiting branch vectors $\mathbf{v}_1, \dots, \mathbf{v}_4$.
  4. **Collinear Strand Pairing**: Finds opposing branch pairs $(\mathbf{v}_a, \mathbf{v}_b)$ that minimize directional deviation $|\mathbf{v}_a \cdot \mathbf{v}_b + 1|$, reconstructing continuous physical strands.
  5. **Over/Under Disambiguation**: Evaluates ink continuity / grayscale intensity across the crossing junction. The continuous strand is classified as **OVER**; the strand with central gaps or breaks is classified as **UNDER**.
  6. **Handedness / Sign**: Computes 2D oriented cross-product between the under-strand and over-strand vectors to assign crossing sign $\pm 1$.
  7. **Cycle Traversal & PD Codes**: Traces the knot loop, assigning sequential arc indices $1, \dots, 2n$, recording each 4-valent crossing in counter-clockwise order:
     $$\text{PD} = [X[i, j, k, l], \dots]$$
  8. **Image Annotation**: Generates visual overlays with circle markers, over-strand vectors, under-strand dashed vectors, and arc numbers.

### 3.2 `gauss_pd.py` &mdash; Skeletal Graph Extraction & Planar Diagrams
* **Primary Classes & Functions**: `GaussPDExtractor`, `GaussPDConfig`, `skeleton_to_graph`, `cluster_junctions`, `prune_spurs`, `simplify_graph`, `extract_gauss_code`
* **Purpose**: Converts 2D boolean skeletal images into `networkx.Graph` structures and resolves topological cycles into standard Gauss notation.
* **Key Mechanics**:
  - `skeleton_to_graph`: Transforms pixel coordinates into graph nodes with 4-way or 8-way adjacency edges.
  - `prune_spurs`: Removes dead-end terminal branches below length $L_{\text{spur}}$ caused by morphological thinning artifacts.
  - `cluster_junctions`: Contraction of multi-node junction cliques into single 4-valent or 3-valent vertices.
  - `simplify_graph`: Condenses long 2-degree chains of pixels into single weighted edges between high-degree junctions.

### 3.3 `reidemeister.py` &mdash; Reidemeister Move Detection
* **Primary Classes**: `ReidemeisterDetector`, `ReidemeisterConfig`, `MoveCandidate`
* **Purpose**: Detects local diagram configurations where Reidemeister moves can be performed:
  - **Type I (R1 &mdash; Twist/Curl)**: A single strand loops over or under itself. Identified by simple cycles of length $L \in [R1_{\min}, R1_{\max}]$ containing at most 1&ndash;2 junctions.
  - **Type II (R2 &mdash; Poke/Bigon)**: Two distinct strands cross each other twice, forming a 2-edge bigon. Detected by parallel paths between shared junction pairs with comparable edge lengths.
  - **Type III (R3 &mdash; Slide/Triangle)**: A strand slides past an existing crossing, forming a 3-crossing triangular region. Detected by 3-cycles with restricted perimeters and balanced edge ratios.
* **Multi-Scale Fusion & NMS**: Reconstructs skeletons at multiple image resolutions (e.g., $0.85\times, 1.0\times, 1.15\times$) and applies Non-Maximum Suppression (NMS) to eliminate duplicate candidate bounding boxes.

### 3.4 `solver.py` &mdash; Iterative Topological Graph Solver
* **Primary Classes & Functions**: `SolverConfig`, `reduce_skeleton`, `solve_image`
* **Purpose**: Iteratively applies verified Reidemeister moves to reduce complex diagrams to minimal crossing representations.
* **Pipeline**:
  1. Preprocesses image and extracts skeleton.
  2. Runs loop searching for R1 curls and R2 bigons.
  3. Surgically modifies the graph topology, removing unnecessary crossing nodes and reconnecting boundary edges.
  4. Returns the reduced skeleton alongside before/after Gauss codes and reduction statistics.

### 3.5 `protein.py` &mdash; 3D Biophysical Knots & Spatial Projections
* **Primary Classes & Functions**: `ProteinBackbone`, `Crossing`, `extract_ca_polyline`, `sample_viewpoints`, `project_polyline`, `detect_crossings`, `gauss_code_from_crossings`
* **Purpose**: Analyzes knotting in biological macromolecules (proteins from KnotProt / PDB).
* **Algorithm**:
  1. Extracts 3D coordinates of alpha-carbon ($C_\alpha$) backbones.
  2. Samples viewing directions uniformly on the unit sphere $\mathbb{S}^2$ (via Fibonacci lattice).
  3. Projects 3D segments onto orthogonal 2D viewing planes.
  4. Solves line segment intersections parametrically:
     $$P_1(t) = A + t(B-A), \quad P_2(u) = C + u(D-C), \quad t, u \in (0, 1)$$
  5. Computes depths $Z_1(t)$ and $Z_2(u)$: if $Z_1(t) > Z_2(u)$, strand 1 is **OVER**; otherwise it is **UNDER**.
  6. Traverses backbone polyline from N-terminus to C-terminus, outputting signed Gauss codes.

### 3.6 `features.py` &mdash; Geometric & Symmetry Invariant Features
* **Functions**: `extract_symmetry_invariant_features`, `compute_rotational_symmetry`, `fourier_descriptors`
* **Purpose**: Extracts compact numerical feature vectors for hybrid knot classifiers:
  - Skeleton area, perimeter, and bounding-box aspect ratio.
  - Normalized Fourier contour descriptors invariant to translation, scale, and starting point.
  - Rotational symmetry scoring under discrete rotation angles ($90^\circ, 180^\circ, 270^\circ$).
  - Mean, standard deviation, and skewness of color channels.

### 3.7 `preprocess.py` &mdash; Morphological Binarization & Skeletonization
* **Primary Classes**: `Preprocessor`, `PreprocessConfig`, `preprocess_for_skeleton`
* **Purpose**: Cleans raw input images into clean single-pixel skeletons.
* **Steps**: Gaussian smoothing $\to$ Adaptive Gaussian thresholding (with Otsu fallback) $\to$ Morphological opening $\to$ Medial axis skeletonization.

### 3.8 `model.py` &mdash; Convolutional Neural Network Architectures
* **Primary Classes & Functions**: `ResNetConfig`, `build_resnet`, `get_resnet`
* **Purpose**: Wraps standard `torchvision` ResNet backbones (ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152) with configurable head layers, transfer learning weights, and selective backbone freezing.

### 3.9 `dataset.py` &mdash; Data Ingestion & Batch Loading
* **Primary Classes**: `KnotImageDataset`, `DatasetConfig`
* **Purpose**: PyTorch `Dataset` parsing directory structures organized by knot classes (`unknot`, `3_1`, `4_1`, etc.), supporting PIL image loading, data augmentations, and index-to-class mappings.

### 3.10 `train.py` & `infer.py` &mdash; Training & Evaluation Pipelines
* **Training (`Trainer`, `TrainConfig`)**: Supports multi-epoch SGD/Adam optimization, Cross-Entropy loss, learning rate scheduling, model checkpointing, and validation metrics (accuracy, precision, recall, F1).
* **Inference (`KnotRecognizer`, `InferenceConfig`)**: Combines ResNet class probabilities with Gauss/PD extraction into a single end-to-end prediction payload.

---

## 4. Generative Framework (`Knotgen/`)

The `Knotgen` sub-package provides a comprehensive generative knot engine covering the table through 10 crossings ($0_1 \to 10_{165}$).

### The 6 Generative Layers
1. **Layer 1: Artin Braid Group $B_n$ (`braid`)**:
   - Represents any knot as the closure of an $n$-strand braid word $\beta = \sigma_{i_1}^{e_1} \dots \sigma_{i_k}^{e_k}$ (Alexander's Theorem).
   - Generates planar diagrams by connecting strands via Artin generator crossings.
2. **Layer 2: Torus Knots $T(p, q)$ (`torus`)**:
   - Closed braids on $p$ strands with word $(\sigma_1 \sigma_2 \dots \sigma_{p-1})^q$.
   - Verified condition: $\gcd(p, q) = 1$ ensures single-component knots.
3. **Layer 3: Rational / 2-Bridge Knots (`rational`)**:
   - Knots classified by continued fractions $[c_1, c_2, \dots, c_k] = p/q$ (Schubert classification).
   - Enforces $p$ is odd for knots; generates 4-plat diagram closures.
4. **Layer 4: Pretzel Knots (`pretzel`)**:
   - Knots formed by joining $k$ vertical 2-strand tangles $P(p_1, \dots, p_k)$ with $p_i$ half-twists.
5. **Layer 5: Hyperbolic 3-Manifold Integration (`from_snappy`)**:
   - Bridges with SnapPy to load Link diagrams, compute hyperbolic invariants (hyperbolic volume, Chern-Simons invariant), and identify manifold isometry classes.
6. **Layer 6: Complete Census Table (`build_census.py` / `census_braids.json`)**:
   - Verified table of 250 prime knots up to 10 crossings.
   - Every knot is verifiable by determinant, Alexander polynomial, and SnapPy identification.

### Polynomial & Determinant Computation
- **Alexander Polynomial $\Delta(t)$**: Constructed from the $(n-1) \times (n-1)$ Wirtinger presentation matrix and normalized such that $\Delta(1) = 1, \Delta(0) \neq 0$.
- **Knot Determinant $\det(K) = |\Delta(-1)|$**: Evaluated via integer matrix determinant at $t = -1$ for optimal performance.
- **Jones Polynomial $V(t)$**: Computed via state-sum Kauffman bracket expansion over $2^n$ smoothings.
- **Burau Representation**: Computes unreduced Burau matrix product over braid generators with verification of $\det(I - M)$.

---

## 5. Topological Notation Specifications

### Planar Diagram (PD) Code
A Planar Diagram code describes a knot diagram as a list of 4-valent crossing vertices. Each crossing is represented as:
$$X[a, b, c, d]$$
where $a, b, c, d$ are the arc labels encountered in counter-clockwise order around the crossing, beginning with the incoming under-strand:
- $a$: Incoming under-strand
- $b$: Outgoing/incoming over-strand
- $c$: Outgoing under-strand
- $d$: Incoming/outgoing over-strand

```
             d (Over)
             ^
             |
   a ------->+-------> c   (Under passes beneath)
 (Under-In)  |       (Under-Out)
             |
             b (Over)
   Order: X[a, b, c, d] (Counter-Clockwise from Under-In)
```

### Signed Gauss Code
A Gauss code follows the single knot strand along its oriented path, recording each crossing sequentially:
- First encounter at crossing $k$ (typically as an under-strand) is recorded as $-k$.
- Second encounter at crossing $k$ (as an over-strand) is recorded as $+k$.
- Handedness (positive right-handed vs negative left-handed) is determined by strand orientation cross product.

---

## 6. Verification & Test Suite

The test suite contains 21 unit tests executed via `pytest`:

```bash
$ pytest tests/ -v
============================== 21 passed in 2.24s ==============================
```

### Test Coverage Matrix
| Test File | Test Cases | Scope Tested |
|---|---|---|
| `test_traverser.py` | `test_knot_traverser_crossing_detection` | Raster image crossing detection, pairing, PD/Gauss code formatting |
| `test_knotgen.py` | `test_knotgen_braid_and_torus`<br>`test_knotgen_rational`<br>`test_knotgen_pretzel`<br>`test_knotgen_burau`<br>`test_knotgen_snappy_integration`<br>`test_census_braids_table`<br>`test_knots_through_9_json` | Braid calculus, Torus knots, 2-bridge knots, Pretzel knots, Burau matrix, SnapPy volume/isometry, 250-knot census |
| `test_protein_stage2.py` | `test_crossing_over_under` | 3D backbone projections, parametric ray intersection & depth sorting |
| `test_reidemeister_synthetic.py` | `test_reidemeister_synthetic_r1`<br>`test_reidemeister_synthetic_r2`<br>`test_reidemeister_synthetic_r3` | Synthetic detection of R1 curls, R2 bigons, R3 triangle moves |
| `test_solver_reduction.py` | `test_reduce_r1_loop_removes_nodes`<br>`test_reduce_no_change_on_line` | Graph reduction, node removal & invariant preservation |
| `test_gauss_pd.py` | `test_cluster_junctions_plus_shape`<br>`test_simplify_graph_plus_shape_edges`<br>`test_extract_gauss_no_crossings_rectangle`<br>`test_extract_gauss_crossing_pd_plus_shape`<br>`test_extract_gauss_lissajous_structure` | Skeleton to graph, junction clustering, edge simplification & PD codes |
| `test_features.py` | `test_feature_vector_length_and_finite_grayscale`<br>`test_feature_vector_rgb_and_symmetry_score_range` | Feature vector length, finite checks, RGB statistics & symmetry bounds |
