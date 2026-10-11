# KnotProt Dataset & Manifest Metadata

## Overview

This directory houses metadata and processed artifacts for the **KnotProt protein knot projection pipeline**.

### Committed Files
- **`hybrid_manifest.csv`**: Metadata-only manifest containing **6,400 samples** (200 unique protein chains $\times$ 32 viewpoint angles).
  - Schema: `key,view_id`
  - **Important**: This CSV contains **metadata only**. The corresponding binary tensor archive (`hybrid_dataset.npz`, containing raster images and Gauss feature vectors) is **not committed to Git** due to repository size limits.

## Provenance & Generation Parameters

The sample index in `hybrid_manifest.csv` was generated using the following experimental protocol:
- **Source Database**: KnotProt (https://knotprot.cent.uw.edu.pl)
- **Viewpoint Sampling**: 32 quasi-uniform directions sampled via spherical Fibonacci lattice (`sample_viewpoints(32)`)
- **Polyline Sampling**: Coordinate stride `stride=3`, maximum length capped at `max_points=300` residues
- **Command Used**:
  ```bash
  python scripts/build_knotprot_hybrid_dataset.py \
    --backbones data/knotprot/backbones.npz \
    --labels data/knotprot/knotprot_labels.csv \
    --viewpoints 32 \
    --stride 3 \
    --max-points 300 \
    --out data/knotprot/hybrid_dataset.npz \
    --manifest data/knotprot/hybrid_manifest.csv
  ```

## How to Obtain Inputs and Reproduce the Dataset

To regenerate `hybrid_dataset.npz` from scratch:

1. **Obtain Chain & Label Records**:
   Download knot annotations and chain definitions from [KnotProt](https://knotprot.cent.uw.edu.pl) and save them as:
   - `data/knotprot/knotprot_chains.csv` (`pdb_id,chain_id`)
   - `data/knotprot/knotprot_labels.csv` (`pdb_id,chain_id,knot_type`)

2. **Download PDB Files**:
   ```bash
   python scripts/download_knotprot.py \
     --csv data/knotprot/knotprot_chains.csv \
     --out-dir data/knotprot/pdb
   ```

3. **Extract C-alpha Backbones (Stage 1)**:
   ```bash
   python scripts/extract_knotprot_stage1.py \
     --pdb-dir data/knotprot/pdb \
     --chains-csv data/knotprot/knotprot_chains.csv \
     --out data/knotprot/backbones.npz \
     --manifest data/knotprot/backbones.csv
   ```

4. **Synthesize Hybrid Projection Dataset (Stages 2–4)**:
   ```bash
   python scripts/build_knotprot_hybrid_dataset.py \
     --backbones data/knotprot/backbones.npz \
     --labels data/knotprot/knotprot_labels.csv \
     --out data/knotprot/hybrid_dataset.npz \
     --manifest data/knotprot/hybrid_manifest.csv
   ```

## Model Checkpoint Compatibility, Metadata & Provenance

The committed model checkpoint `checkpoints/hybrid_classifier.pth` was trained on the 200-chain dataset spanning **4 knot classes**:
- `+3.1` (class index 0)
- `-3.1` (class index 1)
- `-5.2` (class index 2)
- `4.1` (class index 3)

### Metadata & Provenance Scope:
- **Manifest vs. Dataset**: `data/knotprot/hybrid_manifest.csv` records chain and viewpoint metadata (`key,view_id`). Knot class labels reside in `hybrid_dataset.npz`, while class-to-index mappings reside in the dataset and checkpoint dictionary (`class_to_idx`).
- **Checkpoint Contents**: `checkpoints/hybrid_classifier.pth` stores `model_state` and `class_to_idx`. It does not embed raw training key partitions, dataset hashes, or split index arrays. Its 200-chain training provenance is reported provenance documented from `knotprot_chains.csv` and `hybrid_manifest.csv`.

### Holdout Evaluation & Class Mapping Verification:
When evaluating `checkpoints/hybrid_classifier.pth` via `python scripts/train_hybrid_classifier.py --data ... --out checkpoints/hybrid_classifier.pth --eval-only`:
- The script evaluates the model against the holdout split dynamically recomputed from the current dataset using `--seed` (default 7).
- It validates that the dataset's `class_to_idx` mapping matches the checkpoint mapping exactly.
- If an exploratory subset (e.g. `--limit 20`) is supplied with fewer than all 4 classes, evaluation halts with a class-mapping mismatch error to prevent invalid evaluations.
- For subset experiments, train a new model using a dedicated output path (e.g. `--out checkpoints/my_subset_classifier.pth`).

