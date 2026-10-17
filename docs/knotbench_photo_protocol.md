# KnotBench-Photo: protocol for real cable, rope and thread photographs

KnotBench-Photo measures the *perception* half of cable topology: reading a
correct diagram (crossings, over/under, how the ends are joined) from photographs.
The topology half — identification, untangling steps, closure and uncertainty —
is computed exactly by `src/knot_recognition/cable/` once a diagram exists.

The synthetic split (`scripts/run_knotbench_photo.py`) uses the same records and
metrics with rendered cables and exact ground truth. It exercises the pipeline;
it does not replace real photographs.

## 1. What to capture

| Factor | Levels (minimum) |
| --- | --- |
| Material | coax (RG6 / RG58), patch cable, rope (6–10 mm), sewing thread or fishing line |
| Ground-truth type | unknot tangle, 3_1 (both chiralities), 4_1, 5_2, 6_1, granny, square, Hopf link (two cables), two separable cables |
| Background | matte light, matte dark, wood, cluttered desk |
| Lighting | diffuse, single lamp (hard shadows), dim |
| Camera | two phones + one camera; 40–100 cm distance |
| Views per session | 3: top-down, ~30° tilt, ~60° tilt from the opposite side |

Each **physical cable instance** (one tied cable) is photographed in at least
two **sessions** (different place, light or day). Re-tying the same type is a new
cable instance.

## 2. Ground truth

- Tie each knot from a printed diagram; photograph the diagram next to the cable
  in one reference shot (excluded from evaluation).
- **Closed cables:** join the ends with heat-shrink or a connector barrel. The
  type is then fixed.
- **Open cables:** clamp both ends to a closure fixture (a board with two
  clamps at the edge, the cable entering from behind) and fill a
  `ClosureRecord` (`cable/closure.py`): `method="fixture"`, `closing_arc_side`
  = `behind` for a back-panel fixture, `fixture_id`, evidence image. The
  computational closure then replays exactly what the fixture does.
- Verify the type by recomputing it from the annotated diagram
  (`cable.identify.identify_diagram`). Disagreement means re-annotate or re-tie.

## 3. Annotation (per image)

Record format: see the docstring of `cable/photo_bench.py`. Annotate:

1. cable centrelines (polyline per cable, in travel direction),
2. every crossing: position, over-strand tangent, under-strand tangent,
   visibility (`clear`, `partly_occluded`, `occluded`), annotator confidence,
3. free endpoints, and the closure record id.

Two annotators per image; resolve disagreements on over/under by a third look at
another view. Report inter-annotator over/under agreement.

## 4. Splits

`photo_bench.group_split`: test = held-out **cables** imaged in held-out
**sessions**; validation = held-out cables in the remaining sessions; train =
the rest. `photo_bench.check_split` must return no problems. Never split by
image: views of one cable are near-duplicates.

## 5. Metrics

| Metric | Definition |
| --- | --- |
| Crossing precision / recall / F1 | Hungarian matching of predicted to annotated crossings within one cable width (+4 px) |
| Over/under accuracy | on matched crossings; reported per visibility level |
| Topology accuracy | images whose reported type equals the ground truth (abstentions count as wrong) |
| Coverage / abstention rate | fraction of images with / without an answer |
| Selective accuracy | accuracy on answered images |
| Risk–coverage curve, ECE | from the reported type probability (`cable.uncertainty`) |
| Extra-view rate | how often the pipeline asks for another view (`next_best_views`) |

All aggregates carry 95% bootstrap intervals that resample **cables**, not images.

## 6. Pipelines to report

1. `native`: today's traverser PD → identify (abstains on any ambiguous crossing).
2. `adapter`: traverser → `cable.from_traversal` → `cable.uncertainty.propagate`.
3. `adapter_gt_ou`: (2) with ground-truth over/under injected — isolates detection.
4. Non-neural baseline and a learned crossing detector trained on the train split.
5. Multi-view fusion: crossings matched across views; over/under by majority with abstention.

## 7. Known limitations of the skeleton traverser (from the synthetic run)

- Crossings are found as skeleton junctions; thick or outlined cables produce many
  spurious junctions and reject true crossings as "not crossing-like".
- Textbook diagrams with gaps at under-crossings produce no junctions at all
  (the preprocessor opens, never closes, gaps).
- A curl (one crossing closing on itself) is a self-loop edge whose two ends
  share one pairing key; traversal cannot pass through it.
- Its PD code orders arcs by angle and does not encode crossing type; use
  `cable.from_traversal` for any topological answer.

## 8. Knot-aware NanoTransformer (synthetic_v1 results)

`src/knot_recognition/knot_aware.py` keeps the NanoTransformer's encoder blocks but
predicts structure instead of a 250-way label: a soft cable mask, a crossing heatmap,
the over-strand axis at each crossing and the strand axis elsewhere, plus a crossing
count. Input = RGB + an FFT-cleaned grey channel (mirror-padded homomorphic filter +
Wiener shrinkage, `cable/denoise.py`). `cable/neural_parser.py` turns the maps into a
geometric `Diagram`: measure-stable mask (most persistent superlevel set), skeleton cut
at crossings, straightest arm pairing, over/under from the predicted axis, tracing.
Confidence = sigmoid(kappa * margin) shrunk by the consistency of 8 test-time views
(c^gamma), so a view that does not show over/under (a silhouette) cannot produce a
confident type.

Training: 2,000 renders of random braids, plain-light and wood backgrounds only;
cables disjoint from the benchmark. kappa fitted on the benchmark train split;
gamma and parser settings chosen on a separate dev set (`scripts/make_parser_dev_set.py`).

| Sessions | Crossing P / R | Over/under | Pairing | Answered | Wrong answers |
| --- | --- | --- | --- | --- | --- |
| Photo-like (S1-S4), old traverser | 0.11 / 0.16 | 0.49 | -- | 0-2% | 10 without the Euler gate |
| Photo-like (S1-S4), knot-aware | 0.93 / 0.90 | 1.00 | 1.00 | 52% | 0 |
| Unseen backgrounds (S3, S4), knot-aware | 0.85 / 0.81 | 1.00 | 1.00 | 31% | 0 |
| Silhouettes (no over/under visible), knot-aware | 1.00 / 1.00 | 0.57 | -- | 0% | 0 |

Remaining abstentions came mostly from mask topology (skeleton branches and loose
ends where strands pass close together), not from crossings or over/under. Section 9
measures and reduces this.
FFT + measure-stable segmentation alone (no network, `results/knotbench_photo/denoise_ablation.json`)
helps dim and cluttered scenes but hurts wood grain; it is used as an input channel and
post-processing step, not as the segmenter.

## 9. Close strands and the first real-image test (v2, v3)

### 9.1 What was wrong

The skeleton was taken from the cable mask. Where two strands pass within about one
cable width, their masks merge: the skeleton grows a branch between them (gate:
"skeleton branches away from any crossing"), spurs end in the blob ("free cable
ends"), and the face between them disappears (Euler gate). Crossing detection and
over/under were not the problem.

Stress set (`scripts/make_close_strands_set.py`, dev only, seeds disjoint from
training and the benchmark): 100 renders binned by the smallest centre-line gap
between strand pieces away from crossings, in cable widths, 25 per bin.

### 9.2 Changes

| Version | Change |
| --- | --- |
| v1 | mask skeleton (section 8) |
| v2 | **centre-line ridge head** (128 grid): target exp(-d^2/2 sigma^2), d = distance to the nearest strand centre-line; skeleton and Euler count taken from the ridge band, short facing ridge gaps bridged. **Broader styles** (`cable/styles.py`): outlined and hollow ribbons, textbook gaps with thick or thin lines, rope twist, shaded tubes, multicoloured links, widths 5-32 px, new backgrounds; round curves (torus, Lissajous, random Fourier knots) besides braid closures; wider braid wobble so strands pass closer; renders where strands overlap by more than half a width are dropped. Fine-tuned from v1, 3,000 images, 10 epochs. |
| v3 | **full-resolution ridge branch** (256 grid, 2 px at 512) on the frozen v2 trunk, 4 epochs. On the 128 grid two centre-lines about one width apart are 1.5-3 cells apart and merge. Parse with both ridges; report only when every parse that passes all gates gives the same type (`neural_parser.parse_best`). |

The held-out benchmark sessions S3/S4 stay unseen in all versions; kappa is refitted
on the benchmark train split, gamma = 2 (from the parser dev set).

### 9.3 Results

Close-strands stress set: coverage (images answered) by gap, wrong answers in brackets.

| Gap (cable widths) | v1 | v2 | v3 |
| --- | --- | --- | --- |
| 0.6-1.0 | 0% | 0% | 8% |
| 1.0-1.3 | 12% | 36% | 64% |
| 1.3-1.7 | 28% (1) | 44% (1) | 48% (1) |
| >= 1.7 | 64% | 68% | 72% |
| all 100 | 26% (1) | 37% (1) | 48% (1) |

Crossing recall 0.94-0.95 and over/under 0.99-1.00 in every bin and version. The one
wrong answer (same image in all versions, `close074`) is a two-component link whose
projection has two crossings 2.7 px apart: a bigon below image resolution. The
detector sees one crossing, the Euler count agrees (the face between them has no
area), and the parse reads an unknot. No single-view gate can see this; a second view
can.

KnotBench-Photo synthetic_v1 (same images and metrics as section 8):

| | v1 | v2 | v3 |
| --- | --- | --- | --- |
| Crossing P / R (photo-like sessions) | 0.93 / 0.90 | 0.93 / 0.997 | 0.93 / 0.997 |
| Over/under, pairing | 1.00, 1.00 | 1.00, 1.00 | 1.00, 1.00 |
| Answered, all photo-like sessions | 52% | 62% | 69% |
| Answered, unseen sessions S3/S4 | 31% | 28% | 40% |
| Wrong answers | 0 | 0 | 0 |
| Silhouettes answered (must be 0) | 0% | 0% | 0% |

Real knot pictures (`raw_knot/`, 166 images labelled only by type, so the measure is
answered / correct / wrong; `scripts/eval_real_pictures.py`):

| Category | n | old traverser | v1 | v2 | v3 |
| --- | --- | --- | --- | --- | --- |
| drawing | 109 | 0 | 3 | 4 | 4 |
| 3D render | 22 | 0 | 2 | 7 | 8 |
| logo | 8 | 0 | 0 | 0 | 0 |
| photograph of a physical object | 27 | 0 | 0 | 0 | 0 |
| **all (correct / wrong)** | 166 | 0 | 5 / 0 | 11 / 0 | 12 / 0 |

Every answer on real pictures was correct (up to mirror image). Coverage is low.

### 9.4 What this does and does not show

- **Close strands.** Above about one cable width the ridge keeps strands apart and
  coverage rises 2-5x. Below one width (strands touching or overlapping) coverage
  stays near zero. That is the intended behaviour when the image does not separate
  the strands; ask for another view.
- **Real images.** This is the first real-image test, and it is weak evidence: 12
  answers, all on clean drawings and renders. None of the 27 photographs of physical
  rope, cord or sculpture was answered. Most abstentions are crossings without four
  arms and stray branches, i.e. the skeleton still breaks on real textures.
- **Contamination.** The v2 styles (thick ribbons, hollow outlines, shaded tubes) were
  chosen after looking at four failures from this set (a pentagram of circles, a blue
  trefoil render, a 9-crossing outline drawing, a Carrick mat photo). Seven of the 12
  v3 answers come from the blue-tube render family that one of those images belongs
  to. The real-picture numbers are therefore not a blind test. A blind real test
  needs new images.
- **Real cable photographs are still untested.** `raw_knot/` contains no annotated
  cable photographs.

### 9.5 Collecting the real test set

1. Tie and photograph cables following sections 1-2 (at least 20 cables x 2 sessions
   x 3 views).
2. Annotate in `tools/photo_annotator.html` (open in any browser, works offline):
   cable width (two clicks), each crossing (centre, a point on the top strand, a
   point on the bottom strand), visibility, free ends, closure. It writes
   `records.jsonl` in the `cable/photo_bench.py` schema. Served by
   `scripts/annotator_server.py`, it also shows the trained UNetFormer's suggestions (section 14).
3. Put the images under `<folder>/images/` next to `records.jsonl` and run
   `python scripts/eval_real_photos.py <folder>`. Photos are letterboxed (never
   stretched) to 512 px, annotations are mapped with the same transform, and the
   section 5 metrics are reported with cable-level bootstrap intervals, plus a
   multi-view result per cable (answered if any view answers and no two views
   disagree).
4. Do not tune on this folder. Freeze the checkpoint (`checkpoints/knot_aware_nanotransformer.pth`
   = v3; v1 kept as `_v1.pth`) before the first run.

Reproduce: `scripts/make_knot_aware_data.py --styled-frac 0.6 --suffix _v3`;
v2 = `scripts/train_knot_aware.py --data-suffix _v2 --init <v1> --epochs 10 --lr 8e-4` on the
same images without the 256-grid target (the generator now always adds it; drop the
`center_hi` array to rebuild `_v2`); v3 = `--data-suffix _v3 --init <v2> --freeze-trunk --epochs 4 --lr 2e-3`;
`scripts/eval_knot_aware.py <ckpt> knot_aware_v3.json`; `scripts/eval_close_strands.py <ckpt> <out>`;
`KNOT_AWARE_CKPT=<ckpt> SKIP_TRAVERSER=1 OUT_JSON=<out> scripts/eval_real_pictures.py`.

## 10. Held-out evaluation: frozen model, blind set, baselines, ablations

### 10.1 Freezing and the score-once ledger

`results/frozen_model.json` records the SHA-256 of `checkpoints/knot_aware_nanotransformer.pth`
(v3, `6ddb46b4...`) with its parser settings (crossing threshold 0.35, centre-line threshold 0.5,
gamma 2, kappa 26.5, type threshold 0.95). Held-out sets are frozen by the SHA-256 of their manifest
(and of every image). Every scoring run appends to `results/heldout_ledger.jsonl`
(`cable/ledger.py`). The scripts refuse to score a different model as "main" on a set that was
already scored, unless a reason is given and recorded. Baselines and ablations are scored as
"baseline" or "ablation" and never replace the main model.

### 10.2 Blind set blind_real_v1 (scored once)

Built by `scripts/make_blind_set.py` from sources never used or viewed during development; no
image was looked at before scoring:

| Part | Source | Images | Label |
| --- | --- | --- | --- |
| Drawings | knotR (Hankin, GPL-2), diagrams drawn by its own `knotplot()` in R | 98 (5 near-duplicates of raw_knot dropped) | dataset name (Rolfsen <= 10 crossings, square knot, 3_1 # 4_1, unknot) |
| Renders | knotess centrelines (KnotPlot-derived, MIT) of 71 knots and 34 links, POV-Ray: perspective, area light, Phong, optional shadowed backdrop; one near-frontal and one random view each | 210 | exact, from the 3D curve |

Photographs could not be obtained here (Wikimedia Commons, Knot Atlas and arXiv are outside this
environment's network allowlist). `scripts/collect_blind_photos.py` collects Commons photographs on a
machine with normal internet access, writes a review page for human labelling, and freezes the result.

Result of the frozen model (`results/blind/blind_real_v1_knot_aware_nanotransformer.json`):

| Part | Images | Answered | Correct | Wrong |
| --- | --- | --- | --- | --- |
| Drawings | 98 | 42 | 37 | 5 |
| Renders | 210 | 41 | 41 | 0 |
| All | 308 | 83 | 78 | 5 |

These are the first wrong answers in any evaluation. All five are knotR drawings, whose under-strand
gaps are one or two pixels at the working resolution: 7_1 read as 3_1, 8_19 as 8_20, 8_20 as 8_17,
10_123 as 10_124, 3_1 # 4_1 as the square knot. Every crossing in them received over/under
probability 0.96-0.99: all eight test-time views agree on the same reading, so the consistency
shrink, which stops silhouettes, does not fire. Over/under evidence that is systematically absent
(not view-dependent) is not detected. Candidate remedy (to be tested on a NEW blind set): a
network-independent gap check that measures the intensity break of the under-strand at each
crossing, and drawings with near-zero gaps in training labelled as uncertain.

Scoring fix, recorded in the result file: a three-component link (knotess 8.3.2) was first counted
wrong because linking matrices were compared in stored component order. Links are now compared up to
renumbering of components (`cable/compare.py`); the stored answers were rescored without re-running
the model.

### 10.3 Baselines (same training renders, same parser and gates)

| Model | Params | Crossing P / R | Over/under | Answered (photo-like) | Unseen S3/S4 | Wrong | Close strands | Real pictures | Blind: answered / correct / wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250-class image classifier, forced | 0.67M | -- | -- | 100% (12.5% right) | -- | 210 | -- | (training data) | 308 / 2 / 306 |
| same, answering only at p >= 0.95 | 0.67M | -- | -- | 0% | -- | 0 | -- | -- | 0 / 0 / 0 |
| U-Net segmentation + parser (crossings from skeleton junctions, no over/under) | 1.6M | 0.99 / 0.997 | none | 0% | 0% | 0 | -- | -- | 0 / 0 / 0 |
| U-Net segmentation + CenterNet-style keypoint U-Net + parser | 3.1M | 1.00 / 0.98 | 1.00 | **86%** | **74%** | 0 | 50% | 15 / 166 | **138 / 129 / 9** |
| NanoTransformer v3 + parser (frozen main model) | 0.86M | 0.93 / 0.997 | 1.00 | 69% | 40% | 0 | 48% | 12 / 166 | 83 / 78 / 5 |

What this shows:

- Crossing detection is not the bottleneck: a segmentation network finds crossings as well as
  any model, but with no over/under evidence no type ever reaches probability 0.95.
- The formulation does not need the transformer. Two standard U-Nets feeding the same parser
  and gates answer more images, especially on unseen backgrounds, with no wrong answer on the
  benchmark and no answer on silhouettes. On the blind set they answer 138 and are wrong on 9
  (8 of them knotR drawings with nearly invisible gaps, the same failure as the main model).
- Caveats: the U-Net pair has 3.6 times the parameters; it was trained in one stage (12 epochs)
  on the broader-style data, while the NanoTransformer was trained on plain tube renders first
  and then fine-tuned. A size-matched comparison is still to do.
- The classifier is weak even on the pictures it was trained on (18%). A label classifier
  trained on our synthetic renders would be a stronger label baseline; not done yet.

### 10.4 Ablations added

| Variant | Answered | Unseen | Wrong (photo-like) | Silhouettes answered (wrong) | Blind: answered / wrong |
| --- | --- | --- | --- | --- | --- |
| Full v3 (seed 0, CPU) | 69% | 40% | 0 | 0% | 83 / 5 |
| Views averaged, no consistency shrink (gamma = 0) | 69% | 40% | 0 | 62% (25 wrong) | -- |
| Without the FFT channel (full recipe retrained, seed 0) | 63% | 33% | 0 | 0% | 56 / 2 |
| Seed 1 of the full recipe (CPU) | 65% | 32% | 0 | 0% | 72 / 5 |
| Seed 2 of the full recipe (CPU) | 69% | 39% | 0 | 0% | 96 / 6 |
| **3 seeds, mean +- sd** | **68 +- 3%** | **37 +- 5%** | **0** | **0%** | 84 / 5.3 |

Removing the consistency shrink is what lets silhouettes through (25 wrong answers). The model
without the FFT channel (63%, 33% unseen) is within about two standard deviations of the seed
spread, so a benefit of the channel is not established. Seeds 1 and 2 each give one wrong answer on
the 166 real pictures (seed 0 none); on the blind set every seed errs only on drawings with
nearly invisible gaps (5, 5, 6 wrong). `scripts/run_on_mac_gpu.sh` repeats the seeds on an Apple
GPU if wanted; it is not needed for these numbers.

### 10.5 Real cable photographs (capture kit ready)

`data/knotbench_photo/real_v1/`: 20 cables (`cables.csv`, labels computed from the printed
diagrams, chirality included), tie sheets (`capture_kit.pdf`), 120-photo shot list (2 sessions x 3
views per cable, 4 session conditions), README. The annotator fills cable, session, view, type and
material from the file name once `cables.csv` is loaded. `scripts/check_real_photos.py` validates the
set; `scripts/eval_real_photos.py` scores it once.

## 11. Minkowski attention, a Lebesgue loss, gap evidence, and blind set v2

### 11.1 What was added

| Component | Where | What it does |
| --- | --- | --- |
| Lebesgue (layer-cake) loss | `minkowski.layer_cake_loss`, `--layer-cake 1.0` | mean over thresholds t in the parser's band of the area of {p > t} xor {target > t}, which equals the mean of \|clip(p,a,b) - clip(m,a,b)\| / (b - a): mask [0.15, 0.85], ridges [0.3, 0.7] |
| Minkowski position bias | `--attn-bias minkowski` | each head adds -alpha_h \|\|r_i - r_j\|\|_{p_h} on the 16 x 16 token grid; p_h = 1 + softplus(rho_h) (init 2) and alpha_h (init 0.2 ... 0.01) learned per block; unchanged by the 8 test-time views |
| Minkowski content attention | `--attn-content minkowski` | logits -t_h \|\|q - k\|\|_p^p / (p sqrt d), p = 1 on two heads and 2 on two; learned temperature |
| Gap evidence (inference) | `cable/gap_measure.py`, `KA_GAP=fuse` | Lebesgue measure G of the part of each strand's outline that is missing at a crossing (512 px); D = G(under) - G(over); logit p' = logit p + 1.7 D |

Defaults reproduce v1-v3 exactly; old checkpoints load unchanged (`tests/test_minkowski.py`).

### 11.2 Gap evidence: calibration on development data only

`scripts/calibrate_gap_check.py` (results in `results/gap_check/calibration_knot_aware_nanotransformer.json`):
development images = KnotBench train split (72), `parser_dev` (80), and new `gap_dev` renders (120 in five
styles + 40 line drawings with nearly invisible gaps). 856 matched crossings, 85 of them read wrongly by
the network (73 on silhouettes).

| Rule | Development answers (wrong) |
| --- | --- |
| no evidence | 135 (0) |
| fuse, boundary-continuity measure (adopted; tau 0.4, lambda 1.7 by maximum likelihood) | 136 (0) |
| veto p -> 1/2 where D < delta (best grid point) | 131 (0), not adopted |

The cross-section measure (profile correlation) fit the development crossings better (log-likelihood -142
vs -151; -155 without evidence). On the shaded 3D renders of the real-picture set, however, it had the
wrong sign at most crossings the network reads correctly (all 7 on Blue_7_1_Knot), and it cost 6 of 12
real answers. The edge measure was chosen after seeing this (recorded in the calibration file), so its
real-picture numbers are not blind. Blind v2 was frozen before the choice and is the test. A partial run
of the rejected measure on blind v1 (68 images) is in the ledger, with a correction entry.

### 11.3 Blind set v2 (frozen before scoring, never viewed)

`scripts/make_blind_set_v2.py`, manifest `907e2ac4...` (commit 1d2d47e): 207 POV-Ray renders of the
knotess curves from new viewpoints (3 near-duplicates dropped) and 105 matplotlib line drawings with a
halo gap of 0.1-0.8 line widths per side. Labels are exact from the 3D curve or the drawn diagram.
Every run is in the ledger (`results/heldout_ledger.jsonl`). The frozen model is role "main", and
everything else is ablation or baseline.

| Model | Correct | Wrong | of 312 |
| --- | --- | --- | --- |
| Frozen model (plain seed 0) | 54 | 1 | |
| Plain seeds 1, 2 | 52, 51 | 0, 1 | |
| Frozen model + gap evidence | 57 | 1 | drawings 26 -> 30 correct |
| U-Net pair (baseline) | 95 | 1 | |

### 11.4 Minkowski / Lebesgue variants

Training: CPU (cloud, 2 cores) for seed 0 of each variant and seeds 0-1 of `mk_full`. Seed 2 of `mk_full`
was trained on the Mac's Linux VM (CPU, in 3-minute slices, `scripts/train_chunked.py` and
`scripts/recipe_step.sh`; this is bit-identical to an uninterrupted run, which was tested). All three
variants x 3 seeds were also trained on the Mac's GPU (`scripts/run_minkowski_on_mac.sh`, `mps_*`).
Evaluation: `scripts/eval_suite.py` (resumable). Per-model rows: `results/minkowski_summary.json`.

| Group | Benchmark ans. (unseen) | Real ans. | Blind v2 correct / wrong | Paired vs plain, same seed (unseen pts; v2) |
| --- | --- | --- | --- | --- |
| Plain, CPU, 3 seeds | 68% (37%) | 12 | 52.3 / 2 | -- |
| + content attention (mk_full), CPU, 3 seeds | 72% (47%) | 13 | 56.0 / 3 | +0, +15, +15; +5, +1, +5 |
| + position bias (mk_bias), CPU, seed 0 | 70% (43%) | 14 | 60 / 1 | +3; +6 |
| Lebesgue loss only, CPU, seed 0 | 66% (35%) | 8 | 56 / 0 | -5; +2 |
| mk_full, GPU, 3 seeds | 73% (48%) | 12 | 60.3 / 3 | +4, +8, +22; +8, +6, +10 |
| mk_bias, GPU, 3 seeds | 73% (49%) | 12 | 55.3 / 3 | +9, +10, +17; +6, +5, -2 |
| Lebesgue only, GPU, 3 seeds | 71% (46%) | 13 | 55.3 / 2 | -1, +12, +15; +1, +1, +7 |

Reading: compared with the plain model of the same seed, Minkowski content attention never answers less,
and never gets fewer blind-v2 images right. Its benchmark wrong answers stay at 0, it has one more
blind-v2 wrong answer over three seeds, and real pictures do not change consistently. The GPU rows
change the device as well: the Lebesgue-only model gains about as much there, so part of that gain
may be the device. Plain GPU seeds (`scripts/run_on_mac_gpu.sh`) would settle this. Learned exponents
stay at p = 1.89-2.10 and slopes near their initial values, so what helps (if anything) is a multi-scale
locality prior, not a non-Euclidean neighbourhood. Learned values are in `results/minkowski_learned_params.json`.

## 12. Beating the U-Net pair: UNetFormer (matched comparison)

### 12.1 Diagnosis (before any new model)

On KnotBench-Photo both the NanoTransformer and the U-Net pair read crossings (recall 0.997 / 0.98) and
over/under (1.00 / 1.00) almost perfectly. Of the 43 images only the U-Net answers, 29 fail in the
NanoTransformer with "free cable ends", and 37 of the 43 have exactly the right crossing count. Of the 134
free ends on those images, 128 are **off the cable**: the NanoTransformer draws short centre-line and mask
pieces on cluttered and dim backgrounds (mask probability on them has median 0.92). Only 6 are real breaks.

### 12.2 Changes

| Change | Where | Note |
| --- | --- | --- |
| UNetFormer | `src/knot_recognition/unetformer.py`, `--arch unetformer` | U-Net encoder/decoder with skips at all 5 resolutions (widths 16-192) + 4 transformer blocks (width 192, 6 heads) and a count token at the 16 x 16 bottleneck; transformer output added through a zero-initialised projection; predicts all knot-aware heads; 2.9M parameters (the U-Net pair has 3.1M) |
| clDice topology loss | `src/knot_recognition/topology_loss.py`, `--topo-loss 1.0` | soft-skeleton clDice on mask and both ridges |
| Clutter filter (parser) | `cable/neural_parser._drop_open_pieces`, `KA_CLUTTER=1` / `eval_suite.py --clutter` | removes separate centre-line pieces that carry no detected crossing and are open; closed loops kept; no-op with a single piece |
| Widened NanoTransformer | `--embed-dim 192 --depth 8 --heads 6 --stem-dim 64` | 2.9M parameters, size control |

Clutter filter on development images only (`scripts/eval_clutter_filter.py`; new S3/S4-style renders
`data/knot_aware/clutter_dev`, plus parser_dev and gap_dev): frozen model 96 -> 113 answered, 0 wrong;
U-Net pair 143 -> 152, 0 wrong. On KnotBench (not blind for the filter; designed after 12.1): frozen model
69% -> 82% answered (unseen 40% -> 68%), U-Net pair 86% -> 92%; blind v2 +0 and +1.

### 12.3 Matched comparison

`scripts/run_beat_unet.sh` (run on the Mac's GPU): train_v3 data, one stage, 12 epochs, lr 1.5e-3, batch 16,
seeds 0-2 for every model, same device. All evaluated with `eval_suite.py --clutter` on KnotBench, close
strands, real pictures and blind v2 (`scripts/summarize_beat_unet.py`, `results/beat_unet_summary.json`).

| Model | Params | Benchmark answered (unseen) | Close (wrong) | Real answered (wrong) | Blind v2 correct (wrong) |
| --- | --- | --- | --- | --- | --- |
| U-Net pair | 3.1M | 92.1% (85.8%) | 53.0% (6) | 19.3 (2) | 101.3 (2) |
| NanoTransformer widened, one stage | 2.9M | 85.6% (72.8%) | 49.0% (5) | 13.7 (1) | 71.3 (4) |
| **UNetFormer** | 2.9M | 90.4% (82.5%) | 53.0% (5) | **25.3** (3) | **110.7** (3) |
| UNetFormer + clDice | 2.9M | 91.5% (84.7%) | 55.3% (6) | 22.7 (2) | 104.3 (3) |

Means over 3 seeds; wrong answers summed over seeds. UNetFormer minus U-Net pair, same seed: blind v2
+12, +9, +7 correct; real pictures +4, +6, +8 answered; benchmark unseen backgrounds -10.8, +1.7, -0.8
points. Image level, summed over seeds: blind v2 readings correct only for the UNetFormer 43, only
for the U-Net pair 15 (exact McNemar p = 0.02, 0.06, 0.12 per seed). Real pictures: 26 vs 9.

All blind-v2 wrong answers of both models are one drawing (9_25 read as 3_1 # 4_1). All real-picture
wrong answers of both models are one picture that the set holds twice: a single strand with three
figure-eight lobes, read as 4_1 # 4_1 # 4_1, while its folder label is 4_1.

Conclusion: with equal data, recipe, size and device, the UNetFormer beats the U-Net pair on blind v2
and on real pictures at every seed, with about as many wrong answers. The U-Net pair keeps a small
edge on the synthetic benchmark. Size and recipe do not explain the old gap (the widened
NanoTransformer is far behind), and neither does topology: clDice does not help on the sets where
the UNetFormer wins.

## 13. Learned strand order: a next-point pointer head (smoke test)

### 13.1 Idea

The parser orders each strand with hand-written rules: it traces skeleton pixels, bridges short gaps,
and at each crossing pairs the four arms by straightness. Here the network learns the order.
Points are placed along the strand centre-lines about every 4 px of the 256 input, and the parser's
crossing cut (a disc of radius R = 0.9 w + 1.5 on the 256 grid) is removed from both strands. A head
then gives, for every point, a distribution over all other points plus a "none" slot: which points
are next to it along its strand. Reading order and strand order differ:

    1 2 3      the strand visits 1, 2, 3, 6, passes under the strand that covers 5,
    4 5 6      and comes out at 4: next(6) = 4, not 5.

An image does not show which way a strand runs, so targets are symmetric (1/2 on each of a point's
two neighbours; a strand end gives 1/2 to "none"; a point off the cable gives 1 to "none"). Following
the links from any point returns the strand in order, across crossings (`strand_pointer.decode`,
`walks`). Code: `src/knot_recognition/strand_pointer.py`; tests: `tests/test_strand_pointer.py`.

### 13.2 Targets and points

`scripts/make_strand_sidecar.py` re-renders the v3 split with its seeds and keeps each image's ordered
strand paths (`data/knot_aware/{train,val}_v3_strands.pkl`). The val run was checked image by image
against the saved split (all 300 identical). The train run was killed for memory after writing all
3,000 records and before its pixel check, so it was checked against the saved centre-line maps
instead (`scripts/check_strand_sidecar.py`): every record redraws its own image's map to within 13 of
65,536 pixels (sub-pixel rounding); a record of the next image differs by 0.05-0.10 on average.

Training points come half from the true paths (every 4 px, 0.75 px jitter, plus the off-cable points
of the predicted ridge as distractors) and half from the trunk's own predicted centre-line, cut around
its own predicted crossings and labelled by the nearest true strand. Augmentation: the 8 flips and
quarter turns, applied to the image and to the strand paths.

### 13.3 Head

0.51M parameters on the frozen UNetFormer (`checkpoints/beat_unet/mps_uf_s0.pth`). At each point it
reads the UNetFormer feature maps (bottleneck after attention, and the decoder at 32-256), adds
Fourier features of the position, and runs 3 transformer blocks (width 128, 4 heads, attention bias
falling off with distance). Pointer logits are q·k plus a small MLP over pair geometry. The final
version adds three cues the parser's rule uses: the strand's sign-free local direction and its
coherence, the signed direction in which a strand end is open, and the cut radius R (jump lengths
scale with it).

### 13.4 Smoke test

CPU, one seed, 1,500 steps of batch 4 (`scripts/train_strand_pointer.py`). Baseline: the parser's
rules on the same points (`strand_pointer.straightest`): chain nearby points along the predicted
ridge band, pair the four arms at each crossing by straightness, then join remaining ends that run
straight. Links = consecutive points along a strand; jumps = links across a crossing disc.

Development rounds, scored on val images 0-149 (used to diagnose, so not a clean test):

| Round | Change | True points: links / jumps | Predicted points: links / jumps |
| --- | --- | --- | --- |
| 1 | head as in 13.3 without the direction cues; 800 training images | 94.6% / 50.2% | 95.6% / 60.2% |
| 2 | + sign-free direction; all 3,000 images; out-of-reach targets fixed | 94.8% / 48.7% | 95.3% / 58.2% |
| 3 | + open-end direction and cut radius | 96.0% / 67.9% | 96.4% / 72.9% |
| rules | parser's rules on the same points | 92.1% / 51.6% | 94.4% / 59.1% |

Round 1 missed jumps by leaving arm ends unlinked: the head put 0.6-0.84 on the point behind an arm
end and 0.04-0.10 on its partner across the disc. Round 2's sign-free direction cannot tell "ahead
across the gap" from "back along the strand", and its jumps failed most on wide cables (14 px and
more: 11% vs 62% for the rules), whose discs are large; round 3 adds both missing cues.

Held-out test of round 3 on val images 150-299 (not used for training, design or viewing;
`scripts/eval_strand_pointer.py`, `results/strand_pointer/heldout_smoke3_s0.json`), head vs rules,
with the head-minus-rules difference and its 95% bootstrap interval over images:

| Points | Links | Link precision | Jumps (n) | Images with every link right |
| --- | --- | --- | --- | --- |
| true strand points | 96.5% vs 90.9% (+5.6, +4.5..+6.9) | 98.2% vs 93.5% | 71.0% vs 50.3% (+20.8, +17.1..+24.6) of 1,775 | 37 vs 28 (only head 11, only rules 2) |
| predicted centre-line | 95.7% vs 93.5% (+2.2, +1.6..+2.8) | 95.4% vs 93.4% | 72.5% vs 52.8% (+19.7, +15.6..+23.7) of 1,899 | 30 vs 6 (only head 24, only rules 0) |

By crossing (held-out, true points): single crossings with cables under 14 px, head 75-84% vs rules
44-60%; cables of 14 px and more, 26% vs 40% (65 jumps); jumps that span two or more crossings whose
discs merge, 20-50% vs 3-18% (small counts). Both methods also link chains of clutter points on the
predicted ridge (869 vs 1,010 links), which the clutter filter (12.2) is meant to remove.

`scripts/show_strand_order.py` draws the order on the first held-out image with at least four
crossings (`results/strand_pointer/strand_order.png`): in the close-up, strand 1 visits points
1, 2, 5, 9, then 15 across the crossing.

### 13.5 Not done yet

This is a smoke test: one seed, frozen trunk, CPU budget, links between points only. The head is
not yet used by the parser, so nothing here is scored on KnotBench, real pictures or the blind sets.
Next: the GPU run (`scripts/run_strand_pointer_on_mac.sh`, 3 trunk seeds, 6,000 steps of batch 8),
then let the parser take its arm pairing and gap bridging from the head's links and score that with
`eval_suite.py`. Weak spots to work on: wide cables and crossings whose discs merge.

### 13.6 Re-verifying each jump with a separate, non-jump linear test

After the head links an arm end across a crossing, `strand_pointer.check_jumps` walks the straight
segment between the two ends and keeps the jump only if (a) both strands arrive pointing along it
(angle to the segment, measured from each end's other neighbour) and (b) the trunk's centre-line
ridge, which is trained to run on through crossings (under the top strand too), stays high along
it. A failed jump is dropped, so its ends stay open and the parser would abstain there instead of
reading a wrong strand. Thresholds were swept on dev images 0-149 (`scripts/eval_jump_check.py
--sweep`, `results/strand_pointer/jump_check_dev_sweep.json`). Two settings were then fixed: balanced
(70°, ridge mean 0.3) and strict (40°, ridge mean 0.4, ridge never below 0.1). Each was scored once on
held-out images 150-299 (`results/strand_pointer/jump_check_heldout.json`). "Long links" are links
longer than two spacings, nearly all of them jumps.

| Held-out, round-3 head | True points: jumps found | wrong long links | images with none wrong | Predicted points: jumps found | wrong long links | images with none wrong |
| --- | --- | --- | --- | --- | --- | --- |
| head alone | 71.0% | 189 | 83 / 150 | 72.5% | 247 | 71 / 150 |
| + balanced check | 66.9% | 94 | 103 / 150 | 68.8% | 140 | 87 / 150 |
| + strict check | 52.1% | 27 | 132 / 150 | 56.1% | 68 | 110 / 150 |

Links overall change by at most 1 point. The check is a precision dial: balanced halves the wrong
jumps for 4 points of jumps found; strict removes 72-86% of them for 16-19 points. The ridge minimum
costs the most correct jumps, because the ridge dips where a strand runs under the other one. At a
right-angle crossing a wrong turn deviates only 45° from each arm and lies mostly on the ridge, so the
balanced setting lets it through and only the strict setting stops it (`tests/test_strand_pointer.py`).
On dev, offering a rejected end another free end that passes the test ("repair") added 1-2 points of
jumps and 13-23 wrong links back, so it is not used.

### 13.7 A curved re-check that also re-links, and the two checks combined

A second, independent attempt at the same idea (the conversation of 13.6 was lost and the work
redone; both are kept). After the head takes a jump, `strand_pointer.ridge_check` checks it
separately: it follows the smooth curve that leaves the arm end along its arm and enters the partner along the partner's arm
(cubic Hermite), and reads the trunk's centre-line ridge along it. The ridge was trained to run
through every crossing on both strands, so a true jump follows one bar of the X and a turn onto
the other strand cuts the corner, where the ridge is low. On validation images 0-149 (development),
the check separates the head's right and wrong long links better than the head's own confidence
(AUC 0.92 vs 0.84 on true points, 0.84 vs 0.81 on predicted points; median ridge 0.91 on right
jumps, 0.09 on wrong ones). Rejecting alone loses more right jumps than it gains; rejecting and
then re-linking the free ends to the best candidate that passes (`verify_relink`, threshold 0.2
chosen on development images) helps.

Two other neighbourhood ideas were tested first on development images and did not help jumps: a
joint best matching of the free ends around each crossing from the head's probabilities
(67.9% -> 67.4%), and a smaller crossing cut (jumps through merged discs only 19% -> 15% at 0.45 w).
Split by neighbourhood, the head reads 85% of jumps at clean crossings (one crossing, four arm ends;
64% of jumps) but 21% where crossings are so close that their discs merge (19% of jumps).

Held-out (val 150-299, the fixed threshold; `scripts/eval_strand_pointer.py`), re-checked head vs
head, difference with 95% bootstrap interval over images:

| Points | Jumps | Links | Link precision | Images with every link right |
| --- | --- | --- | --- | --- |
| true strand points | 74.9% vs 71.0% (+3.8, +2.1..+5.7) | 96.9% vs 96.5% (+0.4, +0.3..+0.5) | 98.0% vs 98.2% | 50 vs 37 (only re-checked 16, only head 3) |
| predicted centre-line | 73.8% vs 72.5% (+1.3, +0.3..+2.2) | 96.1% vs 95.7% (+0.5, +0.3..+0.6) | 95.0% vs 95.4% | 40 vs 30 (11 vs 1) |

Why not more: most jumps still missed after re-linking have an end already taken by a wrong link
that the ridge check cannot see, because it runs along a ridge: a shortcut along the same strand
(to a point two or more steps away) or a link onto another strand that runs close or touches. A
fifth of them are true jumps whose ridge is weak under the crossing (median 0.09).

Compared on 13.6's measures (held-out val 150-299; `results/strand_pointer/recheck_heldout_long_links.json`
and `recheck_combined_heldout.json`). The combination (curved re-check and re-link, then 13.6's
balanced straight check) was fixed on development images 0-149, where it beat the straight check
alone on every measure, and then scored once:

| Held-out | True points: jumps / wrong long links / images with none wrong / every link right | Predicted points: same |
| --- | --- | --- |
| head alone | 71.0% / 189 / 83 / 37 | 72.5% / 247 / 71 / 30 |
| straight check, balanced (13.6) | 66.9% / 94 / 103 / 37 | 68.8% / 140 / 87 / 29 |
| curved re-check + re-link (this section) | 74.9% / 174 / 89 / 50 | 73.8% / 276 / 66 / 40 |
| **curved re-check + re-link, then straight check** | 69.5% / **78** / **114** / 48 | 69.5% / **134** / 85 / 37 |

The two checks catch different errors. The curved one follows the arms and the weakest stretch of
the ridge, so it rejects turns and finds the right partner for an end the head left open (more
jumps). The straight one also rejects long links that run along the ridge but arrive at the wrong
angle. Together they halve the head's wrong long links for 1.5-3 points of jumps found, and give
11-13 more images with every link right than the straight check alone. This is the default to
carry into the parser (abstaining at an open end is safe; a wrong link is not).

Caveat: the held-out half (150-299) has now scored five fixed variants (the round-3 head, 13.6's two
settings, the curved re-check and the combination). Each was fixed on development images first,
but a fresh set of renders should confirm the final choice.

### 13.8 The UNetFormer trained together with the pointer head (Apple GPU run)

`scripts/train_unetformer_pointer.py` trains the UNetFormer and the pointer head together, from
scratch, with the recipe of 12.3 (train_v3, 12 epochs, batch 16, lr 1.5e-3, AdamW, one-cycle, the
epoch with the best validation loss is kept). The loss is the UNetFormer's loss plus the pointer
loss (weight 1), so the strand order, including every jump across a crossing, shapes the features
the other maps are read from. Each image is seen under one of the 8 flips and quarter turns, applied
to the image, the dense targets and the strand paths (checked against each other in
`tests/test_uf_pointer.py`). Points come from the true paths and, from epoch 4 on, for half of the
images from the model's own predicted centre-line (refreshed every 3 epochs). Batches are prepared
by CPU worker processes and the loss is read back every 50 steps, so the GPU is not kept waiting.
The jump re-checks (13.6, 13.7) have no weights and are applied when the result is scored.

`scripts/run_unetformer_pointer_on_mac.sh` runs seeds 0-2. For each seed it also trains a head on the
frozen plain UNetFormer of the same seed (4,500 steps of 8 images, about as many images as the
joint head sees). It scores both on held-out val 150-299, alone and with the re-checks, and prints
`scripts/summarize_uf_pointer.py`. Options: CONTROL=1 adds the same pipeline without the head;
EVAL_SUITE=1 runs the parser on KnotBench, close strands and real pictures with each new UNetFormer.
First run (Mac, seed 0, stopped in epoch 4): the first 50 steps took 1.2 s each, then 13-40 s,
so an epoch took 1.5-2 hours instead of minutes. The head scored all pairs of points (16 images x
up to 640 x 640 pairs, about 4 GB with gradients) and 18 helper processes held about 7 GB, which
pushed the Mac's shared memory into swap. Fixes: the head scores each point against its 64 nearest
points only (`PointerConfig.candidates`; the true neighbour is outside them for 0.008% of links on
true points and 0.045% on predicted points, 0.15% and 0.06% of crossing jumps; on candidates the
scores equal the all-pairs ones), which cuts the head's peak memory about 3x and its step time
about 2.5x; a neighbour that is not a candidate counts as "none" in the loss; 4 data workers; the
ridge-refresh helpers only exist while refreshing; point counts padded to multiples of 64; the
progress line shows s/step and GPU memory.

Second run (Mac, Apple GPU, all three seeds): 0.43-0.63 s per step, 4.5-6.6 GB of GPU memory,
27 minutes per joint model; the frozen-trunk heads took 16 minutes each. Held-out validation images
150-299 (`results/uf_pointer_summary.json`, `scripts/summarize_uf_pointer.py`); joint = UNetFormer
and head trained together, frozen = head trained on the plain UNetFormer of the same seed (12.3):

| Held-out, mean of 3 seeds | True points: links / jumps | re-checked (`recheck`): jumps / wrong long links (sum) | Predicted points: links / jumps | re-checked: jumps / wrong long links (sum) |
| --- | --- | --- | --- | --- |
| joint | 98.1% / 81.2% | 75.0% / 174 | 97.0% / 80.8% | 75.3% / 385 |
| frozen | 97.8% / 80.3% | 74.2% / 226 | 96.9% / 80.2% | 74.6% / 370 |
| smoke-test head (13.4, CPU, 1,500 steps) | 96.5% / 71.0% | 69.5% / 78 (one seed) | 95.7% / 72.5% | 69.5% / 134 (one seed) |
| parser's rules on the same points | 90.9% / 50.3% | - | 93.5% / 52.8% | - |

Joint minus frozen, per seed: jumps on true points +2.3, -1.5, +2.0 points, on predicted points
+1.2, -1.2, +1.8; images with every link right +11, -18, +5 and +4, -9, +4. Wrong long links of the
head alone on true points 122, 129, 122 vs 167, 148, 148 (fewer at every seed); on predicted
points 203, 188, 199 vs 202, 183, 208. Seed 1's joint UNetFormer trained worse than its plain one
(validation loss 1.944 vs 1.886) and is the seed where the joint head trails.

The UNetFormer's own maps (validation, joint minus plain, mean of 3 seeds): total loss -0.093, but
almost all of it is the crossing-count head (-0.122; 0.17 vs 0.29), which the parser only reports.
The maps the parser reads are unchanged or slightly worse: mask -0.001, crossing heatmap +0.022,
over/under axis +0.009, strand axis and centre-lines within 0.001; crossing recall 0.933 / 0.924 /
0.912 vs 0.935 / 0.930 / 0.930, over/under within 45 degrees 0.955 / 0.954 / 0.955 vs 0.956 / 0.958 / 0.956.

Conclusion: training the strand order into the UNetFormer does not make the UNetFormer better for
the parser (the crossing heatmap gets slightly worse), and the strand order it learns is about the
same as a head trained on the plain UNetFormer (+0.6 to +0.9 jump points on average, mixed across
seeds, fewer wrong long links on true points only). The gain over the smoke test (about +9 jump
points) comes from longer training on the GPU, not from training together. The simpler choice is
the plain UNetFormer with a head trained on top (`checkpoints/strand_pointer/mps_ptr_s<seed>.pth`).
Not done: a control with the same pipeline and no head (CONTROL=1), which would separate the
pipeline's own changes (per-image augmentation, joint clipping) from the head's effect; the parser
end to end with either UNetFormer (EVAL_SUITE=1). The held-out half 150-299 has now scored ten
fixed configurations; fresh renders should confirm the final one.

## 14. Model-assisted annotation

`scripts/annotator_server.py` (double-click `scripts/annotate_photos.command` on a Mac) serves
`tools/photo_annotator.html` on `http://localhost:8765` with a trained UNetFormer (default
`checkpoints/beat_unet/mps_uf_s0.pth`, kappa 26.5 from its evaluation, clutter filter on) and the
strand-order head trained on it (`checkpoints/strand_pointer/mps_ptr_s0.pth`). For each photo,
letterboxed to 512 x 512 as in `scripts/eval_real_photos.py`, it suggests, in the photo's pixels:
- the cable width, as a segment across the cable;
- each crossing: centre, a point on the strand on top, a point on the strand below, and the
  probability of that reading;
- free ends;
- centre-lines in travel direction. These are the parser's traced strands; when the parser
  cannot trace them, they are the strand-order head's ordered points with the jump re-check
  (`strand_pointer.recheck`), and the strand below at each crossing is marked as a guess;
- the system's reading of the type, or why it abstains. This needs SnapPy; without it the
  suggestions still appear.

The page draws suggestions dashed. "Accept all", "Accept crossings only", and per-crossing flip
and delete turn them into annotations. Nothing is saved unless accepted. Each saved item carries
`source` (`manual`, `model`, `model, over/under flipped`, `model (bottom strand guessed)`); the
record carries `assist` (checkpoint and hash, counts by source, the model's reading).
Everything runs locally; opened as a file, the page works by hand as before.

Bias. Accepted suggestions pull the ground truth towards the model. For a set that is scored
(real_v1), take every over/under from the cable or its tie sheet, flip disagreeing suggestions,
and have the second annotator work without suggestions; `source` lets results be split by how
each crossing was annotated. On a KnotBench photo (`c001_S2_desk`) the suggestions put all four
crossings within 0.9 px of the true ones, with the right strand on top, in about 1 s on a CPU
(`tests/test_annotator_server.py`).

### 14.1 On the Hugging Face Space

The suggestions are shared code (`src/knot_recognition/annotate.py`): the local server above and
the KnotAssistant Space (`huggingface_space/app.py`) both use it. The Space has three tabs:
1. the 250-class NanoTransformer with the traverser (unchanged);
2. multi-view cable photo triage, now parsed by the same UNetFormer (clutter filter, kappa 26.5)
   instead of the knot-aware NanoTransformer, with the suggestions drawn on each view;
3. "Annotation assistant": one photo in; the suggestions drawn on it (green = strand on top,
   orange = strand below, dashed when guessed; strands with travel arrows; free ends; width),
   the reading or the reasons it abstains, a crossing table, and the suggestions as a JSON draft
   in the same format the local server sends the page. Three synthetic KnotBench-Photo renders
   are examples (the unknot `c001_S2_desk`, the trefoil `c007_S2_desk` and the figure-eight
   `c019_S4_clutter`; the model reads all three correctly). No real photos are published.

`scripts/build_space.py` copies the modules, the checkpoints, Knotgen, KnotInfo and the examples
into `huggingface_space/` and writes `runtime/MANIFEST.json` (source and sha256 of each file);
`--check` fails when the folder is behind the repository (`tests/test_space.py`).
`scripts/upload_space.command` (double-click; `scripts/upload_space.py`) refreshes the folder,
lists it, asks, uploads it with the user's Hugging Face login and follows the rebuild until the
new commit runs. The GitHub workflow `.github/workflows/cable-space.yml` uses the same build
script, so a push to `main` deploys the same folder.

The Space runs on ZeroGPU. The UNetFormer runs on the CPU in the main process (about 1.5 s per
photo locally, Python 3.11, Gradio 5.49.1), so it uses no GPU quota; only the NanoTransformer tab
keeps the `@spaces.GPU` decorator. SnapPy is imported when the app starts: its PARI library
installs signal handlers on first import, which Python allows only in the main thread, and Gradio
answers in worker threads. (The earlier Space imported it inside a request, so its knot types
could fail with "signal only works in main thread".)
