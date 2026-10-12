# Usage

## Install
```bash
pip install knot-recognition
```

## Train a model
```bash
python -m knot_recognition.train --data-dir /path/to/data --outdir ./checkpoints --epochs 20 --batch 32 --lr 1e-3
```

Training on a specific device:
```bash
python -m knot_recognition.train --data-dir /path/to/data --device cpu
```

> **Note on Checkpoints**:
> - `checkpoints/best.pth` is the output produced by the 2D ResNet training workflow (`python -m knot_recognition.train`). It is not pre-packaged in the repository.
> - The committed checkpoint `checkpoints/hybrid_classifier.pth` is for the **protein projection model** (`HybridModel`) and cannot be loaded by `knot` / `knot-solve`.

## Run inference (CLI)
```bash
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```

Include symmetry-invariant features:
```bash
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv --features
```

Force a device:
```bash
knot --image /path/to/image.png --checkpoint ./checkpoints/best.pth --device cpu
```

## Detect Reidemeister moves (heuristic)
```bash
knot-moves --image /path/to/image.png --overlay results/figures/moves_overlay.png
```

Output is a JSON list of candidate moves with bounding boxes and scores.

## Reduce + classify (solver)
```bash
knot-solve --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```

Save the reduced skeleton:
```bash
knot-solve --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv \
  --reduced-skeleton results/figures/reduced_skeleton.png
```

## Run inference (module)
```bash
python -m knot_recognition.infer --image /path/to/image.png --checkpoint ./checkpoints/best.pth --mapping mapping_example.csv
```

Force a device:
```bash
python -m knot_recognition.infer --image /path/to/image.png --checkpoint ./checkpoints/best.pth --device cuda
```

## Vision NanoTransformer & Active Traverser (250 Prime Knots: 0_unknot to 10_165)

Train the lightweight Vision NanoTransformer locally across all 250 prime knot classes:
```bash
python scripts/train_nanotransformer.py --epochs 25 --batch-size 32
```

Perform active diagram traversal, top-k classification, and topological similarity retrieval:
```python
from knot_recognition import KnotTraverser, load_nanotransformer

# Load locally trained NanoTransformer
model = load_nanotransformer("checkpoints/nanotransformer_knot250.pth")

# Attach to traverser
traverser = KnotTraverser(nanotransformer=model)

# Traverse knot diagram, classify topology, and retrieve nearest topological neighbors
result = traverser.traverse_and_classify("path/to/knot_diagram.png", top_k=5)

print("Predicted Knot:", result.classification["top_label"])
print("Confidence:", result.classification["top_probability"])
print("Consensus Check:", result.consensus)
print("Planar Diagram (PD):", result.pd_string)
print("Similar Knots from Census:")
for knot in result.similar_knots:
    print(f"  Rank {knot['rank']}: {knot['label']} (Cosine Sim: {knot['similarity']:.3f}, Crossings: {knot['crossings']})")
```

## Protein Knot Pipeline

### External Prerequisites
1. Download chain definitions and labels from [KnotProt](https://knotprot.cent.uw.edu.pl):
   - `data/knotprot/knotprot_chains.csv` (`pdb_id,chain_id`)
   - `data/knotprot/knotprot_labels.csv` (`pdb_id,chain_id,knot_type`)
2. Download PDB structures:
   ```bash
   python scripts/download_knotprot.py --csv data/knotprot/knotprot_chains.csv --out-dir data/knotprot/pdb
   ```

### Execution Steps
Stage 1 (Cα backbones):
```bash
python scripts/extract_knotprot_stage1.py \
  --pdb-dir data/knotprot/pdb \
  --chains-csv data/knotprot/knotprot_chains.csv \
  --out data/knotprot/backbones.npz \
  --manifest data/knotprot/backbones.csv
```

Stage 2–3 (projection + crossing sequence):
- Use `sample_viewpoints`, `project_polyline`, `detect_crossings`, `gauss_code_from_crossings`

Stage 4 (hybrid ML):

#### Option A: Quick local experiment (subset training)
```bash
# 1. Build a small 20-chain subset:
python scripts/build_knotprot_hybrid_dataset.py \
  --backbones data/knotprot/backbones.npz \
  --labels data/knotprot/knotprot_labels.csv \
  --viewpoints 32 --limit 20 --stride 3 --max-points 300 \
  --out data/knotprot/my_subset_dataset.npz \
  --manifest data/knotprot/my_subset_manifest.csv

# 2. Train on the subset into a new checkpoint:
python scripts/train_hybrid_classifier.py \
  --data data/knotprot/my_subset_dataset.npz \
  --out checkpoints/my_subset_classifier.pth \
  --epochs 5 --batch 64 --lr 1e-3
```

#### Option B: Regenerating & evaluating the committed 4-class checkpoint
> **Note on Provenance & Holdout Evaluation**:
> - `data/knotprot/hybrid_manifest.csv` records chain and projection viewpoint identifiers (`key,view_id`); class labels reside in `hybrid_dataset.npz` and class mappings reside in `class_to_idx`.
> - The 200-chain training provenance (`+3.1`, `-3.1`, `-5.2`, `4.1`) is reported provenance derived from `knotprot_chains.csv` and the generation manifest, as `checkpoints/hybrid_classifier.pth` stores `model_state` and `class_to_idx` without embedded split indices.
> - Running `--eval-only` evaluates the model against the holdout split dynamically recomputed from the current dataset using `--seed` (default 7).

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



## Python API
```python
from knot_recognition.infer import infer_image

result = infer_image(
    img_path="path/to/image.png",
    checkpoint="./checkpoints/best.pth",
    mapping_csv="mapping_example.csv"
)

print(result)
```

## Output fields
- `predicted_label`, `pred_prob`
- `mapping_pd`, `mapping_gauss`
- `auto_gauss`, `auto_pd` (heuristic)
- `features` (symmetry-invariant descriptor if `--features` is set)
- `chirality`, `chirality_confidence`
