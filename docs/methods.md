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
- Trace the curve to recover crossing order.
- Construct a PD-like representation from cyclic edge order.

## CNN Classification
- ResNet backbone with a task-specific final layer.
- Training uses cross-entropy loss and standard augmentations.
