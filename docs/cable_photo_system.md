# Cable photograph topology workflow

## What this prototype does

The Gradio Space reuses the repository's skeleton and crossing parser. It produces an overlay, reports unresolved junctions, and computes a planar-diagram Alexander polynomial and determinant only when the parser returns a structurally consistent closed diagram. A census lookup provides candidate labels, not a unique knot identification. Multiple supplied views are processed independently; disagreement or an unresolved crossing requests another angle.

## Explicit assumptions and limits

- The parser was developed for rendered knot diagrams. Coaxial-cable photographs have not been represented in a labeled training set and this Space is not a validated photo classifier.
- A 2D photograph cannot reveal the depth order at every overlap. The parser's image-based over/under heuristic is a proposal; the overlay needs human inspection.
- Multiple views are not a calibrated 3D reconstruction. The current app compares independently parsed diagrams and asks for a new view when results conflict.
- Open cable ends are not silently joined. The app reports that the invariant is withheld. For a future closed-loop workflow, the closure convention must be user-selected and persisted with the result. Different closures can give different knot types.
- Alexander polynomial and determinant are useful invariants but are not complete: equal fingerprints do not establish that two knots are equivalent.
- This is a research prototype for standard single-component knots. It does not classify links, knots with contact/self-overlap ambiguity, or damaged/disconnected cable masks.

## Intended capture and review protocol

1. Photograph the entire tangle against a contrasting matte background with minimal glare.
2. Capture at least two views with a substantial camera/cable rotation; include close-ups of occluded crossings.
3. Inspect every marked crossing. If the over/under relation is not visible or the parser marks it unresolved, provide another view.
4. For open ends, define the physical closure (for example, an actual closure fixture) and record it. Do not use an implicit image-frame closure as a physical claim.
5. Accept a census label only as a candidate requiring independent confirmation. Preserve the input images, annotations, closure record, parser version, invariant fingerprint, and uncertainty notes.

## CI and Hugging Face deployment

`.github/workflows/cable-space.yml` checks syntax and imports on pull requests and pushes. On `main`/`master`, it uploads the validated Space directory to `AbhyudayMishr/KnotAssistant` using:

- Secret `KNOTASSISTANT`: a Hugging Face fine-grained token with write access to the target Space.

The Gradio Space already exists. A push to the deployment branch publishes the app automatically; pull requests run validation only.

## Evaluation required before claiming performance

Build a consented coaxial-cable image set with physical ground truth: known tied knots, open tangles with documented closures, multiple cable types, backgrounds, lighting, cameras, occlusion severity, and repeated independent views. Split by physical cable instance and capture session. Annotate cable masks, centerlines, crossings, over/under state, endpoints, and annotator confidence. Report crossing precision/recall, over/under accuracy, topology accuracy, abstention/extra-view rate, calibration, and chain/cable-level confidence intervals. Include non-neural and 3D scan/video baselines and an external capture setup. Until then, describe this as an assisted diagram-analysis demo, not a reliable recognition product.
