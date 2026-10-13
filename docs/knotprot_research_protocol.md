# KnotProt Research Evaluation Protocol

This protocol is intended for experiments where the independent unit is a
protein chain. Never split projection rows independently: all views of one
chain belong to one partition.

## Required, versioned inputs

Keep `knotprot_chains.csv`, `knotprot_labels.csv`, the exact PDB coordinate
files, and their checksums. The downloader writes
`data/knotprot/pdb/download_manifest.json`; it records each retrieved file's
SHA-256 because a PDB accession can be revised. Review label provenance and
labeling conventions before using a release as ground truth. The repository
does not include those source records or the PDB files.

```bash
python scripts/download_knotprot.py \
  --csv data/knotprot/knotprot_chains.csv \
  --out-dir data/knotprot/pdb

python scripts/extract_knotprot_stage1.py \
  --pdb-dir data/knotprot/pdb \
  --chains-csv data/knotprot/knotprot_chains.csv \
  --out data/knotprot/backbones.npz \
  --manifest data/knotprot/backbones.csv

python scripts/build_knotprot_hybrid_dataset.py \
  --backbones data/knotprot/backbones.npz \
  --labels data/knotprot/knotprot_labels.csv \
  --viewpoints 32 --stride 1 \
  --out data/knotprot/hybrid_dataset.npz \
  --manifest data/knotprot/hybrid_manifest.csv
```

Stage 1 selects one C-alpha alternate location per residue by occupancy,
prefers the blank alternate-location identifier on ties, preserves insertion
codes, and records gaps in residue numbering. The projection builder excludes
segments spanning recorded gaps. Its provenance sidecar hashes the coordinate
archive, label CSV, output dataset, and view manifest and records per-chain
input/output point counts and sampling parameters. Preserve these outputs;
do not overwrite a dataset used for a reported experiment.

## Evaluation

```bash
python scripts/evaluate_knotprot.py \
  --data data/knotprot/research_2026_10/development/hybrid_dataset.npz \
  --manifest data/knotprot/research_2026_10/development/hybrid_manifest.csv \
  --backbones data/knotprot/research_2026_10/development/backbones.npz \
  --external data/knotprot/research_2026_10/external/hybrid_dataset.npz \
  --external-backbones data/knotprot/research_2026_10/external/backbones.npz \
  --out data/knotprot/research_2026_10/evaluation.json \
  --seed 23 --folds 5 --test-fraction 0.20 \
  --views 1,2,4,8,16,all --view-modes uniform,random
```

The runner reserves a stratified chain-level holdout, performs grouped
cross-validation on the remaining chains, and records exact IDs and hashes for
each fold and the final holdout. For each split it compares image-only,
projection-feature-only, and concatenated hybrid inputs with logistic
regression, RBF SVM, and ExtraTrees. With `--backbones`, it also compares a
rotation/translation-invariant whole-chain shape descriptor (a 3D geometric
baseline, not a knot invariant). It aggregates per-view probabilities into
one prediction per chain before scoring. Metrics include per-class recall,
macro-F1, balanced accuracy, log loss, Brier score, 10-bin ECE, and
chain-bootstrap 95% intervals. View tests use either evenly spaced IDs or a
seeded random subset. Fix the final test set before model or view selection.

The acquisition helper `scripts/fetch_knotprot_cohorts.py` freezes the official
API tables and per-chain coordinate files. It reconstructs the historical 200
chain IDs with a binary `K` (knot) / `S` (slipknot) target, and creates an
external chain cohort from published (`Y`), non-redundant API entries after
removing development PDB-chain IDs. The helper rejects chains with conflicting
K/S labels and records downloads and failed chain IDs in a source manifest.
This target differs from the unavailable historical four-class
chirality/type labels.

```bash
python scripts/fetch_knotprot_cohorts.py
```

KnotProt's API also documents that assigning a knot type to an open chain
requires an explicit closure method (stochastic or direct); its knotoid
workflow is a different target. Since both cohorts come from KnotProt, this is
external-chain validation, not an independent database replication. Report
PDB accession overlap and cluster homologous sequences when sequence metadata
is available.

## Additional completed analyses and remaining checks

- `scripts/evaluate_knotprot_cnn.py` trains an image-only CNN on the saved
  random chain split; `scripts/evaluate_knotprot_cnn_cluster_cv.py` trains it
  under each saved RCSB 30%-identity group fold. Both average view
  probabilities to a chain prediction. They are fixed-seed feasibility runs,
  not a multi-seed model comparison.
- `gauss_feats` derives from a heuristic projected crossing-event sequence;
  it is not a verified topological invariant. The optional Topoly 1.1.0
  `scripts/audit_topology_sampling.py` run adds an Alexander-polynomial
  baseline with deterministic MASS_CENTER and stochastic TWO_POINTS closure.
  See `results/topology_sampling_development.json` and
  `results/topology_sampling_audit.json`. Whole-chain topology is not a
  slipknot fingerprint, and the result depends on closure; KnotProt labels may
  also derive from the same Topoly family, so the near-perfect agreement is
  not independent label verification.
- The whole-chain shape descriptor is a 3D baseline, but it is not a trusted
  topological invariant. Projection features still depend on this repository's
  heuristic crossing detector.
- Topoly recomputation finds deterministic whole-chain type changes after
  stride 3 and a concrete cap-induced trefoil-to-unknot change for external
  chain `6ac6_A`. These checks do not recalculate the K/S fingerprint itself;
  broader closure sensitivity and slipknot fingerprint analysis remain
  important.
- `scripts/audit_sequence_clusters.py` maps chains to the RCSB 30%-identity
  clusters and reruns grouped CV. Nine development chains are unmapped, 22
  accession lookups failed, and the earlier random split shares nine clusters
  across train and test. The previous external set shares ten clusters with
  development training.
- The current external cohort is from KnotProt, not an independent database.
  AlphaKnot 2.0's public API returned HTTP 503 during retrieval; the available
  Hugging Face knot/unknot release is access-gated and has a different target.
  Independent-database validation remains open.
- The original four-class inputs remain absent; current results concern a
  separately sourced K/S reconstruction only.

The repository's original four-class inputs are still absent. The frozen
KnotProt API cohorts support a separate K/S reconstruction and external-chain
evaluation, subject to the source and label caveats above.
