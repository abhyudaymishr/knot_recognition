---
title: Cable Knot Photo Triage
emoji: 🪢
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Cable knot photo triage

Experimental, human-reviewed photo workflow built on this repository's diagram parser and Knotgen invariants. It is not a validated coaxial-cable vision model. Open cable ends are not automatically closed; the user must define and record a physical closure before a knot type can be assigned. Invariant matches are candidate lists, since the Alexander polynomial and determinant are not complete knot invariants.

The cable photo tabs run the trained UNetFormer parser (`runtime/mps_uf_s0.pth`, clutter filter on, kappa 26.5). The **Annotation assistant** tab adds its strand-order head (`runtime/mps_ptr_s0.pth`) and shows what it would mark on a photo: each crossing with the strand on top and underneath, the free ends, the strands in travel order and the cable width, downloadable as a JSON draft. These are suggestions for a person to check, not labels; for benchmark ground truth, take which strand is on top from the tie sheet rather than from the model. `runtime/MANIFEST.json` lists the source and sha256 of every model file; the folder is refreshed with `scripts/build_space.py` in the repository.

Project documentation: [cable photo system and deployment](https://github.com/abhyudaymishr/knot_recognition/blob/main/docs/cable_photo_system.md).
