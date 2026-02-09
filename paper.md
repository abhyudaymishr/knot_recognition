---
title: "Knot Recognition: Image-Based Classification with Heuristic Gauss/PD Extraction"
tags:
  - knot theory
  - topology
  - computer vision
  - image processing
authors:
  - name: abhyudaymishr
    orcid: ""
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2026-02-09
bibliography: paper.bib
---

# Summary
This software provides a scientific pipeline for classifying knot diagrams from images and extracting symbolic representations. It combines a ResNet-based classifier with a structured, heuristic Gauss/PD extractor operating on skeletonized drawings. The codebase is organized to support reproducible experiments and extensible methods research.

# Statement of Need
Image-based knot recognition is a useful tool for topology research, education, and dataset creation. Existing resources often provide symbolic knot codes but lack an integrated, reproducible pipeline for image-based recognition and code extraction. This package provides a foundation for:
- Classifying knot types from images,
- Recovering crossing order and PD-style representations,
- Running systematic experiments with clear documentation and tests.

# Methods Overview
The pipeline uses preprocessing and skeletonization to extract topological structure from input images. A graph-based simplification and traversal recovers crossing order and produces Gauss/PD-like outputs. A CNN classifier provides image-based labels for knot type prediction. These components are integrated into a documented, testable workflow.

# Acknowledgements
The project uses open-source scientific libraries including PyTorch, NumPy, SciPy, scikit-image, and NetworkX.
