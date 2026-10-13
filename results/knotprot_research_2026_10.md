# KnotProt K/S Feasibility Evaluation

## Scope and target

The original four-class experiment cannot be reconstructed: its labels and
feature archive were not committed. The historical manifest identifies 200
chains, and the current KnotProt API assigns a consistent binary whole-chain
status to them: `K` (knot) or `S` (slipknot). This report therefore evaluates
that distinct, explicitly sourced target; it does not validate the old
four-class checkpoint.

Source tables and coordinate bytes were frozen on 2026-10-12 in
`data/knotprot/research_2026_10/source_manifest.json`. The development cohort
contains 200 historical chains (66 K, 134 S). The external cohort contains
154 published, non-redundant API chains (71 K, 83 S); three coordinate requests
failed and are listed in the source manifest. There are zero overlapping PDB
chain IDs and zero overlapping PDB accessions. Both cohorts come from KnotProt,
so this is external-chain validation within one database, not an independent
database replication.

## Protocol

The saved evaluation in `data/knotprot/research_2026_10/evaluation.json` uses
seed 23, five grouped folds over 160 development chains, and an untouched
40-chain holdout. It records each fold's exact train/validation IDs, the final
train/test IDs, their hashes, source hashes, per-class recall, macro-F1,
balanced accuracy, Brier score, log loss, 10-bin ECE, and stratified
chain-bootstrap 95% intervals.

Inputs are 32 Fibonacci-sphere projections per chain built with stride 3 and a
300-point cap. The image, projected crossing-feature, and concatenated hybrid
modalities are evaluated with logistic regression, RBF SVM, and ExtraTrees.
A whole-chain 3D shape descriptor is also included as a non-topological
baseline. At each view count, predictions are averaged per chain before
scoring. View counts are 1, 4, 8, and all 32; subsets use uniform or seeded
random selection.

## Results

Mean five-fold metrics using all 32 views with uniform selection:

| Input | Model | Accuracy | Macro-F1 | Balanced accuracy | ECE |
|---|---|---:|---:|---:|---:|
| Image | RBF SVM | 0.881 | 0.858 | 0.850 | 0.120 |
| Hybrid | RBF SVM | 0.875 | 0.853 | 0.847 | 0.115 |
| 3D shape | ExtraTrees | 0.938 | 0.929 | 0.929 | 0.085 |
| Projected features | RBF SVM | 0.725 | 0.643 | 0.640 | 0.187 |

The 3D shape model is the strongest cross-validation result, but its balanced
accuracy falls to 0.734 on the external chains (95% CI 0.664–0.805). The
external image RBF SVM reaches 0.715 balanced accuracy (95% CI
0.641–0.779); hybrid RBF SVM reaches 0.718 (0.644–0.786); projection-feature
RBF SVM reaches 0.683 (0.612–0.752). External results are substantially lower
than chain-level CV, especially for the 3D shape model. This points to dataset
shift and/or redundancy effects; it does not support a strong generalization
claim.

Using RBF SVM, mean CV balanced accuracy rises from one view to all views from
0.793 to 0.850 for images and from 0.769 to 0.847 for hybrid inputs. The
crossing-feature-only model remains weak: 0.641 at one view and 0.640 at all
views. The source's crossing-event histogram does not behave like a sufficient
topological representation in this experiment.

On the untouched holdout, the CV-selected 3D shape ExtraTrees model reaches
0.848 balanced accuracy (95% CI 0.714–0.962); on the external cohort it reaches
0.734. The holdout interval is wide because it contains only 40 chains.

## Integrity findings and limits

- Chain labels are inherited from KnotProt's full-chain annotations. The
  experiment does **not** recompute topology after subsampling, so it cannot
  establish that stride 3 preserves the knot/slipknot label.
- The 300-point cap never activates in the 200-chain development cohort: the
  longest chain has 875 C-alpha points, which becomes 292 after stride 3. It
  activates for 4 of 154 external chains (maximum original length 1,597).
- We ran paired external representation audits against the same source labels.
  Removing the cap (stride 3, 300-point cap versus stride 3, no cap) changes
  four chains; none of the image, projection-feature, or hybrid logistic
  predictions flips on those four chains. Mean projection-feature cosine
  similarity on the affected chains is at least 0.9987. The complete audit is
  in `results/sampling_audit_cap.json`.
- Changing stride 3 to full-resolution (no cap) changes all 154 external
  representations. Under the same development-trained logistic models, 58
  image, 59 projection-feature, and 59 hybrid chain predictions flip. For
  image/projection/hybrid respectively, external balanced accuracy changes
  from 0.718/0.711/0.706 to 0.597/0.530/0.604. Projection-feature cosine
  similarity has median 0.986 and minimum 0.660 across chains. This is strong
  sensitivity to the preprocessing choice, but still **not** evidence that
  subsampling changes the mathematical knot label: no topology was recomputed
  on reduced coordinates. See `results/sampling_audit_stride.json`.
- The added raw-PDB parser explicitly selects the first model, resolves
  alternate locations by occupancy (blank wins ties), preserves insertion
  codes, and records residue-number gaps. The downloaded experimental
  coordinates are KnotProt's processed XYZ files, so these PDB parser rules
  were regression-checked but were not the source of these two cohorts.
- The 3D descriptor uses lengths, compactness, covariance eigenvalues, virtual
  angles, and turns. It is rotation/translation invariant but is **not** a
  topological invariant. The Topoly analysis below adds a separate topology
  baseline with explicit open-chain closures.
- The historical development chains are redundant. Chain-level folds prevent
  projection-view leakage but not homolog leakage; see the sequence-cluster
  audit below.
- The CNN comparison below is a single fixed-seed run; multiple seeds and
  training-only threshold/calibration selection are still needed.

These are feasibility results for a new binary K/S task, not evidence of an
ICML/NeurIPS-ready contribution. The sequence-cluster and sampling audits
below reveal substantial family and preprocessing effects that need to shape
the final study design.

## Follow-up experiments

### CNN comparison on the frozen holdout

The compact image-only ProjectionCNN was trained for 12 fixed epochs on all
views from the exact 160-chain training partition and evaluated once on the
same untouched 40-chain holdout. Inputs were area-resized to 32×32, matching
the classical image-feature resolution. It reaches 0.556 balanced accuracy
(95% chain bootstrap CI 0.500–0.630), macro-F1 0.360, and accuracy 0.400. Knot
recall is 1.000 while slipknot recall is 0.111. The 3D shape ExtraTrees
baseline from the same holdout reaches 0.848 balanced accuracy (CI
0.714–0.962). In the 30%-identity sequence-grouped folds, this CNN averages
0.579 balanced accuracy (fold SD 0.112), close to the image RBF SVM's 0.590.
These fixed-seed runs show no CNN advantage here; multiple seeds and
training-only calibration would be needed for a publishable comparison.

### Topology recomputation and invariant baseline

We installed and pinned Topoly 1.1.0 as an optional research dependency and
recomputed Alexander knot types on source coordinates, every-third-residue
coordinates, and stride-3 coordinates capped at 300 points. The primary paired
check uses the deterministic MASS_CENTER closure. It changes the whole-chain
knot type for 5/200 development chains and 12/154 external chains between
full-resolution and stride 3. For stride 3 versus stride 3 plus the cap, type
changes for 0/200 development chains and 1/154 external chains. The latter is
`6ac6_A`: its result is trefoil (`3_1`) before the cap and unknot (`0_1`)
afterward.

The deterministic invariant baseline maps a resolved nontrivial whole-chain
Alexander type to K and an unknot to S; unresolved `TMC`/`Unknown` outcomes are
excluded. It resolves all 40 untouched holdout chains and reaches balanced
accuracy 1.000 there. On the external cohort, 152/154 resolve and balanced
accuracy is 0.988 (95% chain-bootstrap CI 0.970–1.000). At stride 3, balanced
accuracy is 0.981 on the untouched development holdout and 0.933 on external
chains; with the cap these are 0.981 and 0.926. Across all development chains,
the stride-3 score is 0.959. This is a valid topology-based baseline but
not independent confirmation of the labels: KnotProt annotations may use the
same Topoly-family closure analysis. Whole-chain Alexander type also does not
identify a slipknot fingerprint by itself. Stochastic TWO_POINTS distributions
are retained as descriptive Monte Carlo results, not used as paired topology
change tests. Full per-chain data, closure settings, and hashes are in
`results/topology_sampling_development.json` and
`results/topology_sampling_audit.json`.

### Sequence-cluster split and independent database status

RCSB PDB 30%-identity clusters map the 200 development chains to 70 groups
(largest group: 34 chains). Nine development chains are unmapped and treated
as singleton groups. In five-fold cluster-grouped CV, image RBF SVM balanced
accuracy falls to 0.590 from 0.850 in prior chain-grouped CV; 3D shape
ExtraTrees reaches 0.853 versus 0.929. The earlier random train/holdout split
has nine overlapping 30%-identity clusters, and the development training
portion overlaps ten such clusters with the external cohort. Thus the previous
external set is PDB-accession-disjoint but not sequence-cluster-disjoint. Exact
fold IDs/hashes and cluster memberships are in
`results/sequence_cluster_audit.json`. Twenty-two accession metadata failures
are recorded and could hide additional homolog overlap.

An independent-database evaluation is still not complete. AlphaKnot 2.0 is the
most appropriate separate source found: it stores AlphaFold-model knot and
slipknot labels, but its public API returned HTTP 503 during retrieval. The
alternative Hugging Face `knots_AF` release requires accepting gated access
conditions and login; its available card describes knot/unknot labels, which
also differ from the current K/S target. Thus the current 154-chain test stays
an external-chain test within KnotProt, not an independent-database test.
