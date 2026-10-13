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

## 7. Known limitations of the current traverser (from the synthetic run)

- Crossings are found as skeleton junctions; thick or outlined cables produce many
  spurious junctions and reject true crossings as "not crossing-like".
- Textbook diagrams with gaps at under-crossings produce no junctions at all
  (the preprocessor opens, never closes, gaps).
- A curl (one crossing closing on itself) is a self-loop edge whose two ends
  share one pairing key; traversal cannot pass through it.
- Its PD code orders arcs by angle and does not encode crossing type; use
  `cable.from_traversal` for any topological answer.
