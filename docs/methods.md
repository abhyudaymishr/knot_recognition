# Methods

## Image Preprocessing
- Resize input to a fixed width and preserve aspect ratio.
- Convert to grayscale, equalize histogram, and apply Gaussian blur.
- Perform adaptive thresholding with Otsu fallback.
- Morphological opening removes small artifacts.
- Skeletonize for topology extraction.

## Heuristic Gauss/PD Extraction
- Build a pixel graph from the skeleton.
- Prune short spurs to reduce noise.
- Cluster junction pixels into crossings.
- Simplify the graph by collapsing degree-2 chains.
- Pair edges at crossings using local geometry.
- Compare grayscale continuity along each paired strand. Emit an over/under label only when the continuity margin passes the evidence thresholds; otherwise mark the candidate ambiguous.
- Trace through junctions using the paired edge identity. Ambiguous candidates are excluded from Gauss/PD output and listed in result warnings.
- Construct a PD-like representation only for resolved crossings. A lack of junction candidates is reported as unverified, not as proof that the image has no crossings.

This is a confidence-gated image pass, not a temporal/video tracker. It cannot guarantee a correct label from an ambiguous raster image, and the current skeleton graph can miss a crossing entirely when an under-strand gap disconnects its arms. These cases require better candidate generation or human review; a zero-crossing result must not be treated as verified ground truth.

## CNN Classification
- ResNet backbone with a task-specific final layer.
- Training uses cross-entropy loss and standard augmentations.
