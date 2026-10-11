# Reproducibility Guide

## 1. Supported Environments & Python Runtimes
- **Python Version & Platform**: Package metadata specifies `python >= 3.9`. Active development, execution, and unit-test verification have been specifically benchmarked on **Python `3.13.13` on macOS `Darwin 24.3.0` (arm64)**. Full dependency resolution across earlier Python versions (e.g., 3.9–3.11) may require platform-specific wheel builds.
- **Dependencies**:
  - `pyproject.toml` declares permissive semantic version bounds for package distributions.
  - `requirements.txt` provides a **pinned direct-dependency list** matching the tested environment (PyTorch 2.14.1, torchvision 0.29.1, numpy 2.5.3, OpenCV 5.0.0, scikit-image 0.26.0, SymPy 1.14.0, pytest 9.1.1). Note that `requirements.txt` is not an exhaustive, hashed transitive lockfile; transitive dependencies and platform-specific wheels are resolved at installation time.

## 2. Installation Protocol

### Standard Core + Test Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[test]"
```

### Pinned Direct-Dependency Environment
To install the verified direct-dependency set:
```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

### Optional Extras
- **Hyperbolic Geometry (`SnapPy`)**: To enable manifold isometry verification and volume calculations:
  ```bash
  pip install -e ".[snappy]"
  ```
  *(Note: SnapPy is optional; basic knot synthesis, census loading, and tests run without it.)*

## 3. Running the Test Suite
Execute unit tests from the workspace root:
```bash
pytest tests/ -v
```
The test suite comprises 24 unit tests across 7 test modules (dated benchmark: ~2.5–3.5s on macOS Apple Silicon, October 2026; execution duration is environment-dependent):
- `test_traverser.py`: Crossing extraction, strand pairing, PD/Gauss code formatting, and crossing gap/continuity resolution.
- `test_knotgen.py`: Braids, Torus, Rational, Pretzel, Census table, SnapPy integration, and procedural knot image rendering.
- `test_protein_stage2.py`: 3D backbone projection and depth-based over/under crossing classification.
- `test_reidemeister_synthetic.py`: Synthetic R1, R2, R3 move candidate detectors.
- `test_solver_reduction.py`: Graph reducer and Reidemeister loop removals.
- `test_gauss_pd.py`: Skeleton-to-graph extraction, spur pruning, junction clustering.
- `test_features.py`: 24D feature vector (Hu moments, radial Fourier descriptors, horizontal reflection symmetry).

## 4. Determinism & Seeding
- **Image Training (`knot_recognition.train`)**:
  - Set `--seed <INT>` to seed Python `random`, `numpy`, and PyTorch RNG.
  - Training employs randomized data augmentations (flips, rotations).
  - Validation employs a **deterministic, unaugmented evaluation protocol** (aspect-ratio resize and normalization).
- **Protein Hybrid Training (`scripts/train_hybrid_classifier.py`)**:
  - Set `--seed <INT>` to ensure reproducible group-stratified splits across protein chains.

## 5. Pipeline Data Requirements & External Prerequisites

| Pipeline | Required Inputs | Sourced From | Produced Artifacts |
|---|---|---|---|
| **2D Drawing Classifier** | Image folder: `data_root/<knot_class>/*.png` | Synthetic / user dataset | Checkpoint: `checkpoints/best.pth` |
| **Protein Knot Pipeline** | 1. `data/knotprot/knotprot_chains.csv`<br>2. `data/knotprot/knotprot_labels.csv`<br>3. PDB files in `data/knotprot/pdb/` | External: [KnotProt](https://knotprot.cent.uw.edu.pl) & RCSB PDB | `backbones.npz`, `hybrid_dataset.npz`, `checkpoints/hybrid_classifier.pth` |
| **Knotgen Census Engine** | Self-contained (`Knotgen/census_braids.json`) | Tracked in Git | Procedural knots, polynomial invariants, DT codes |

