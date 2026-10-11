# Knot Recognition & Knotgen: Comprehensive Architectural Reference

A complete technical specification of the `knot_recognition` repository, detailing the dual-engine framework combining computer vision, deep learning, graph topology, 3D structural biology, and combinatorial knot generation.

---

## 1. System Overview & Architecture

The `knot_recognition` ecosystem unifies classical computational knot theory with modern computer vision and deep learning. It addresses three core challenges:

1. **Topological Extraction from Images**: Processing 2D raster line drawings of knots and extracting candidate topological representations (heuristic Planar Diagram (PD) and Gauss-like codes) via skeletal graph analysis (`gauss_pd.py`) or raster path traversal with grayscale continuity (`traverser.py`).
2. **Topological Simplification & Invariance**: Detecting candidate Reidemeister moves ($R_1, R_2, R_3$) and applying heuristic local $R_1/R_2$ graph reductions to simplify skeletal diagrams (without formal equivalence proofs or minimal crossing guarantees).
3. **Generative Knot Engine (`Knotgen`)**: Layered procedural synthesis of all prime knots up to 10 crossings ($0_1 \to 10_{165}$), computing polynomial invariants (Alexander, Jones, Burau) and linking directly with hyperbolic 3-manifold geometry (SnapPy).
4. **Biophysical Knots (Protein Knots)**: Parsing 3D $C_\alpha$ backbones from PDB/KnotProt files, projecting onto arbitrary viewing spheres, solving projected 2D segment intersections in $t, u \in (0, 1)$ with depth interpolation, and categorizing protein entanglement.

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
                       +-----------------------+ | Segment Intersect  |
                                    |            +--------------------+
                                    v                      |
                 +--------------------------------------+  |
                 | traverser.py (raster over/under)     |  |
                 | gauss_pd.py (skeleton graph cycles)  |  |
                 | Junction Clustering (4-valent)       |  |
                 | Collinear Branch Pairing (traverser) |  |
                 | Cycle Traversal & Arc Ordering       |<-+
                 +--------------------------------------+
                                    |
          +-------------------------+-------------------------+
          v                                                   v
+-----------------------------+                 +-----------------------------+
| Heuristic Topological Codes |                 | reidemeister.py / solver.py |
| - PD Code: PD[X[...], ...]  |                 | - Candidate R1/R2/R3 moves  |
| - Gauss Code: [1, -2, 3...] |                 | - Heuristic R1/R2 reductions|
| - Heuristic graph traversal |                 |   (no formal minimal proof) |
+-----------------------------+                 +-----------------------------+
          ^                                                   |
          |                                                   v
+-----------------------------+                 +-----------------------------+
| Knotgen/ Generator Engine   |                 | model.py / infer.py         |
| Braids, Torus, Rational,    |                 | ResNet Classification       |
| Pretzel, SnapPy, Census     |                 | Bundled Inference Outputs   |
+-----------------------------+                 +-----------------------------+
```

---

## 2. Directory Hierarchy (Tracked Workspace)

```
knot_recognition/
├── KNOT_RECOGNITION_ARCHITECTURE.md # Technical architecture specification
├── pyproject.toml                   # Project metadata, dependencies & optional extras
├── pytest.ini                       # Test suite runner configuration
├── requirements.txt                 # Pinned dependencies for reproducible environment
├── generate_gauss_codes.py          # Batch Gauss code generator script
├── infer.py                         # Top-level CLI entry point for knot image inference
├── train.py                         # Top-level CLI entry point for neural network training
├── mapping_example.csv              # Reference mapping file schema
├── mapping_generated.csv            # Generated dataset mapping table
│
├── src/knot_recognition/            # Core Python package
│   ├── __init__.py                  # Public API exports (with PEP 562 lazy CLI imports)
│   ├── __main__.py                  # python -m knot_recognition entry point
│   ├── traverser.py                 # Knot image traverser (crossing detection, PD & Gauss codes)
│   ├── gauss_pd.py                  # Skeletal graph extraction, junction clustering & cycles
│   ├── reidemeister.py              # Reidemeister move candidate detector (R1, R2, R3)
│   ├── solver.py                    # Iterative graph simplification solver via R1/R2 reductions
│   ├── protein.py                   # 3D protein backbone projections & segment intersection depth solver
│   ├── features.py                  # 24D image features: Hu moments, radial Fourier profile & reflection symmetry
│   ├── preprocess.py                # Grayscale, adaptive binarization & morphological thinning
│   ├── dataset.py                   # PyTorch KnotImageDataset implementation
│   ├── model.py                     # ResNet backbone architectures (ResNet-18 to ResNet-152)
│   ├── train.py                     # Training loop, checkpoints & validation metrics
│   ├── infer.py                     # Bundled neural classification + heuristic topological extraction engine
│   └── utils.py                     # File I/O helpers & safe image readers
│
├── Knotgen/                         # Generative knot framework
│   ├── __init__.py                  # Knotgen module exports
│   ├── knotgen.py                   # 6-layer knot generator & polynomial invariant computer
│   ├── build_census.py              # Automated builder & validator of 250 census braids (lazy SnapPy)
│   ├── enumerate.py                 # Planar 4-valent diagram generator via DT codes
│   ├── census_braids.json           # Validated braid representations for 250 Rolfsen/HTW knots
│   └── knots_through_9.json         # Dowker-Thistlethwaite table through 9 crossings
│
├── scripts/                         # Experiment drivers & dataset preparation
│   ├── download_knotprot.py         # Automated KnotProt PDB downloader
│   ├── extract_knotprot_stage1.py   # Stage 1 PDB C-alpha backbone extractor
│   ├── build_knotprot_hybrid_dataset.py # Hybrid projection + Gauss histogram dataset builder
│   ├── merge_hybrid_parts.py        # Dataset partition combiner
│   └── train_hybrid_classifier.py   # Hybrid model trainer and evaluator
│
├── data/knotprot/                   # Protein pipeline data and metadata
│   ├── README.md                    # Data provenance and dataset regeneration instructions
│   └── hybrid_manifest.csv          # Metadata-only view manifest (6,400 samples)
│
├── docs/                            # Research documentation
│   ├── scientific.md                # Scientific context, methodology & audited results
│   ├── methods.md                   # Formal methodology specification
│   └── reproducibility.md           # Environment setup, test commands & reproducibility guide
│
├── results/                         # Experiment outputs
│   ├── hybrid_training.csv          # Audited training run metrics
│   └── figures/                     # Experiment visualization plots
│
├── checkpoints/                     # Model weights
│   └── hybrid_classifier.pth        # Trained hybrid protein projection model
│
└── tests/                           # Complete test suite (7 test modules)
    ├── test_traverser.py            # Image crossing traversal, pairing & PD code validation
    ├── test_knotgen.py              # Knotgen layers, invariants, census & SnapPy integration
    ├── test_protein_stage2.py       # 3D backbone projection & over/under crossing tests
    ├── test_reidemeister_synthetic.py# Synthetic R1, R2, R3 move detection benchmarks
    ├── test_solver_reduction.py     # Graph solver R1/R2 reduction validations
    ├── test_gauss_pd.py             # Skeleton-to-graph & junction clustering tests
    └── test_features.py             # Symmetry, Fourier & color feature vector tests
```

---

## 2.1 System Capabilities & Maturity Status

To maintain scientific transparency, system components are categorized by maturity:

| Component | Status | Description & Operational Scope |
|---|---|---|
| **Knotgen Procedural Engine** | **Implemented & Verified** | Procedural generation across braids ($B_n$), torus knots $T(p,q)$, rational knots, pretzel knots, and 250-knot census. Verified Alexander polynomials, determinants, and Jones state sums. |
| **SnapPy Manifold Bridge** | **Optional / Integration** | Provides hyperbolic volume and exterior isometry verification when SnapPy is installed; gracefully optional otherwise. |
| **3D Protein Segment Intersections** | **Implemented** | Uniform viewpoint sampling on $\mathbb{S}^2$, closed-form projected 2D segment intersection $t, u \in (0,1)$ with $Z$-depth interpolation, and over/under assignment. |
| **Image Skeletonization & Traversal** | **Heuristic Pipeline** | Thinning to 1-pixel skeletons, 4-valent junction clustering, collinear branch pairing, and ink-continuity over/under scoring (`traverser.py`). Note that `gauss_pd.py` operates strictly on binary skeletons without grayscale depth scoring. Heuristic; sensitive to stroke thickness and raster gaps. |
| **Reidemeister Detection & Solver** | **Heuristic Detection & Local Reducer** | Heuristic candidate detection for R1, R2, R3 motifs using cycle bases and multi-edges; iterative heuristic R1 curl and R2 bigon reductions on simplified graphs. Returns candidates and reduced graphs, without formal equivalence proofs or guaranteed minimal diagrams. |
| **ResNet Image Classifier** | **Implemented** | Standard torchvision transfer learning on 2D images with deterministic validation transform protocol. |
| **Hybrid Protein Classifier** | **Implemented** | CNN + Gauss histogram model with group-stratified splits across protein chains. |


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
  5. **Over/Under Disambiguation (Traverser Specific)**: Evaluates ink continuity / grayscale intensity across the crossing junction in `img_gray`. The strand with uninterrupted dark pixels is classified as **OVER**; the strand exhibiting central intensity breaks is classified as **UNDER**.
  6. **Handedness / Sign**: Computes 2D oriented cross-product between the under-strand and over-strand vectors to assign crossing sign $\pm 1$.
  7. **Cycle Traversal & PD Codes**: Traces the knot loop, assigning sequential arc indices $1, \dots, 2n$, recording each 4-valent crossing in counter-clockwise order:
     $$\text{PD} = [X[i, j, k, l], \dots]$$
  8. **Image Annotation**: Generates visual overlays with circle markers, over-strand vectors, under-strand dashed vectors, and arc numbers.

### 3.2 `gauss_pd.py` &mdash; Skeletal Graph Extraction & Planar Diagrams
* **Primary Classes & Functions**: `GaussPDExtractor`, `GaussPDConfig`, `skeleton_to_graph`, `cluster_junctions`, `prune_spurs`, `simplify_graph`, `extract_gauss_code`
* **Purpose**: Converts 2D boolean skeletal images into `networkx.Graph` structures and resolves topological cycles into heuristic candidate Gauss and PD representations.
* **Operational Scope & Subsystem Boundaries**:
  - `GaussPDExtractor.extract(skeleton)` operates strictly on the 2D boolean binary skeletal array; it does **not** consume grayscale image data (`img_gray`) and does **not** evaluate central gap intensity or resolve over/under strand continuity.
  - Crossing handedness and 3D depth assignments are not recovered; outputs represent planar graph cycles and candidate junction incidence lists rather than validated topological invariants.
  - Image inference via `knot_recognition.infer` currently bundles `GaussPDExtractor` for skeleton extraction rather than `KnotTraverser`.
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
* **Purpose**: Iteratively applies heuristic local graph reductions (R1 curl removal and R2 bigon elimination) to simplify skeletal diagrams. It does not perform formal Reidemeister equivalence proofs and does not guarantee a minimal crossing diagram.
* **Pipeline**:
  1. Preprocesses image and extracts skeleton.
  2. Runs loop searching for candidate R1 curls and R2 bigons.
  3. Surgically modifies the graph topology, removing detected curl/bigon crossing nodes and reconnecting boundary edges.
  4. Returns the reduced skeleton alongside heuristic traversal codes and reduction statistics.

### 3.5 `protein.py` &mdash; 3D Biophysical Knots & Spatial Projections
* **Primary Classes & Functions**: `ProteinBackbone`, `Crossing`, `extract_ca_polyline`, `sample_viewpoints`, `project_polyline`, `detect_crossings`, `gauss_code_from_crossings`
* **Purpose**: Analyzes knotting in biological macromolecules (proteins from KnotProt / PDB).
* **Algorithm**:
  1. Extracts 3D coordinates of alpha-carbon ($C_\alpha$) backbones.
  2. Samples viewing directions uniformly on the unit sphere $\mathbb{S}^2$ (via Fibonacci lattice).
  3. Projects 3D segments onto orthogonal 2D viewing planes.
  4. Solves 2D segment intersections between projected backbone segments parametrically in the projection plane:
     $$P_1(t) = A + t(B-A), \quad P_2(u) = C + u(D-C), \quad t, u \in (0, 1)$$
  5. Computes interpolated depths $Z_1(t)$ and $Z_2(u)$ along the original 3D segments: if $Z_1(t) > Z_2(u)$, strand 1 is **OVER**; otherwise it is **UNDER**.
  6. Traverses backbone polyline from N-terminus to C-terminus, outputting an over/under encounter sequence (+k for over, -k for under; tracking traversal elevation rather than crossing handedness/chirality).

### 3.6 `features.py` &mdash; 24D Image Feature Representation
* **Primary Functions**: `extract_symmetry_invariant_features`, `hu_moments`, `radial_fourier_descriptor`, `reflection_symmetry_score`
* **Feature Inventory (24-element vector)**:
  - **7 Log-Transformed Hu Moments** (`hu_moments`): Computed from image central moments on blurred grayscale.
  - **16 Radial Fourier Magnitudes** (`radial_fourier_descriptor`): 1D profile of radial bin intensity averages from the image center, followed by FFT magnitude extraction and $\ell_2$ normalization.
  - **1 Reflection Symmetry Score** (`reflection_symmetry_score`): Normalized cross-correlation between the zero-mean image and its horizontal reflection (`np.fliplr`).
* **Implementation Scope**: Does not compute contour Fourier descriptors, 90°/180°/270° rotational symmetry, color channel statistics, or skeletal morphology metrics; does not claim full scale/rotation/start-point invariance.

### 3.7 `preprocess.py` &mdash; Morphological Binarization & Skeletonization
* **Primary Classes**: `Preprocessor`, `PreprocessConfig`, `preprocess_for_skeleton`
* **Purpose**: Cleans raw input images into clean single-pixel skeletons.
* **Steps**: Grayscale conversion & histogram equalization $\to$ Gaussian smoothing $\to$ Adaptive Gaussian thresholding (with Otsu fallback) $\to$ Morphological opening $\to$ Morphological skeletonization (`skimage.morphology.skeletonize`).

### 3.8 `model.py` &mdash; Convolutional Neural Network Architectures
* **Primary Classes & Functions**: `ResNetConfig`, `build_resnet`, `get_resnet`
* **Purpose**: Wraps standard `torchvision` ResNet backbones (ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152) with configurable head layers, transfer learning weights, and selective backbone freezing.

### 3.9 `dataset.py` &mdash; Data Ingestion & Batch Loading
* **Primary Classes**: `KnotImageDataset`, `DatasetConfig`
* **Purpose**: PyTorch `Dataset` parsing directory structures organized by knot classes (`unknot`, `3_1`, `4_1`, etc.), supporting PIL image loading, user-supplied torchvision transforms (with data augmentation configured upstream in `Trainer`), and index-to-class mappings.

### 3.10 `train.py` & `infer.py` &mdash; Training & Evaluation Pipelines
* **Training (`Trainer`, `TrainConfig`)**: Implements supervised ResNet training using the Adam optimizer with fixed learning rate (`lr`), Cross-Entropy loss, configurable batch size and epochs, and either random or stratified train/val index splitting. Metrics logged per epoch are training loss, training accuracy, validation loss, and validation accuracy (learning rate scheduling, SGD, and precision/recall/F1 metrics are not included in the standard trainer).
* **Inference (`KnotRecognizer`, `InferenceConfig`)**: Executes model evaluation by producing a result bundle of independent outputs: predicted class label (`predicted_label`) and its probability score (`pred_prob`), optional CSV lookup codes (`mapping_pd`, `mapping_gauss`), heuristic skeleton extraction results (`auto_gauss`, `auto_pd`), optional 24D symmetry features (`features`), and reflection-based chirality inference (`chirality`, `chirality_confidence`). Subsystem outputs are bundled together in the return dictionary without joint fusion; it does not return top-$k$ candidates or full probability arrays.

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
   - Bridges with SnapPy to load Link diagrams, compute hyperbolic volume, DT codes, and identify manifold isometry classes in the SnapPy census.
6. **Layer 6: Complete Census Table (`build_census.py` / `census_braids.json`)**:
   - Stored braid-word representations for the unknot plus 249 prime knots ($3_1 \to 10_{165}$).
   - Verification protocol performed by `build_census.py`:
     - **Hyperbolic knots**: Verifies that the knot complement exterior of the braid closure is isometric to the reference SnapPy census manifold (`M.is_isometric_to(M_ref)`).
     - **Non-hyperbolic torus knots** (`3_1`, `5_1`, `7_1`, `9_1`, `8_19`, `10_124`): Verifies that the Alexander polynomial of the braid closure matches the analytic torus knot invariant ($\Delta(\text{braid}) = \Delta(T(p, q))$).
     - **Unknot**: Validated via trivial empty braid check.
   - **Verification Metadata vs. Test Suite**: Stored entries in `census_braids.json` record the resulting boolean `verified` and `check` string tags. The automated test suite (`tests/test_knotgen.py:test_census_braids_table`) validates the structural schema of the JSON table (250 entries, integer braid generators within strand bounds, presence of verification flags); it does not re-compute all 250 SnapPy manifold isometries or polynomial invariants during test execution.

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

### Gauss Codes vs. Over/Under Encounter Sequences
- **Classical Signed Gauss Code**: In classical knot theory, a Gauss code follows the single oriented knot loop, recording each crossing encounter sequentially. The sign of each entry represents local topological crossing chirality (handedness: $+1$ right-handed, $-1$ left-handed).
- **Protein Pipeline Over/Under Encounter Sequence**: In `protein.py` (`gauss_code_from_crossings`), the signed tokens $+k$ and $-k$ signify whether the protein backbone passes **OVER** ($+k$) or **UNDER** ($-k$) at crossing $k$. This encodes relative spatial elevation and backbone traversal ordering rather than classical topological chirality. Depth near-ties affect over/under classification, not handedness.

---

## 6. Verification & Test Suite

The test suite contains 24 unit tests executed via `pytest` (benchmark verified in Python `3.13.13` on macOS `Darwin 24.3.0`, October 11, 2026):

```bash
$ pytest tests/ -v
============================== 24 passed in 1.90s ==============================
```


### Test Coverage Matrix
| Test File | Test Cases | Scope Tested |
|---|---|---|
| `test_traverser.py` | `test_knot_traverser_crossing_detection`<br>`test_knot_traverser_user_api_and_opencv_compatibility`<br>`test_crossing_classifier_resolves_clear_gap_without_forcing_ambiguous` | Raster image crossing detection, pairing, PD/Gauss code formatting, OpenCV `cv2.imwrite` and PIL `.save()` compatibility, gap vs ambiguous crossing classification |
| `test_knotgen.py` | `test_knotgen_braid_and_torus`<br>`test_knotgen_rational`<br>`test_knotgen_pretzel`<br>`test_knotgen_burau`<br>`test_knotgen_snappy_integration`<br>`test_census_braids_table`<br>`test_knots_through_9_json`<br>`test_knotgen_image_rendering` | Braid calculus, Torus knots, 2-bridge knots, Pretzel knots, Burau matrix, SnapPy volume/isometry, census table schema validation, procedural diagram rendering |
| `test_protein_stage2.py` | `test_crossing_over_under` | 3D backbone projections, projected segment intersection & depth sorting |
| `test_reidemeister_synthetic.py` | `test_reidemeister_synthetic_r1`<br>`test_reidemeister_synthetic_r2`<br>`test_reidemeister_synthetic_r3` | Synthetic detection of R1 curls, R2 bigons, R3 triangle moves |
| `test_solver_reduction.py` | `test_reduce_r1_loop_removes_nodes`<br>`test_reduce_no_change_on_line` | Graph reduction, node removal & invariant preservation |
| `test_gauss_pd.py` | `test_cluster_junctions_plus_shape`<br>`test_simplify_graph_plus_shape_edges`<br>`test_extract_gauss_no_crossings_rectangle`<br>`test_extract_gauss_crossing_pd_plus_shape`<br>`test_extract_gauss_lissajous_structure` | Skeleton to graph, junction clustering, edge simplification & PD codes |
| `test_features.py` | `test_feature_vector_length_and_finite_grayscale`<br>`test_feature_vector_rgb_and_symmetry_score_range` | Feature vector length, finite checks, RGB statistics & symmetry bounds |
