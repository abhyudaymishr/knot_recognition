# KnotBench-Photo real_v1: real cable photographs (to be captured)

Everything except the photographs is ready. Model to score: frozen
(`results/frozen_model.json`, sha256 6ddb46b4...). Do not change it after the first photo is taken.

1. Print `capture_kit.pdf`. Tie the 20 cables from their sheets (a gap = this strand
   passes under). Join the ends of each cable so it is a closed loop. Labels in
   `cables.csv` were computed from the drawings themselves, including chirality.
2. Photograph each cable in its 2 sessions, 3 views each, named exactly as in
   `shot_list.csv` (`cXX_sY_vZ.jpg`, 120 photos). Take one `cXX_reference.jpg` of the
   cable beside its sheet (not evaluated). Re-lay the cable loosely between sessions.
3. Copy the photos to `images/`. Open the annotator: double-click `scripts/annotate_photos.command`
   (or `python scripts/annotator_server.py`), which opens it in the browser with the trained
   UNetFormer suggesting width, crossings, over/under, free ends and centre-lines; or open
   `tools/photo_annotator.html` directly to annotate by hand. Load `cables.csv`, then the photos:
   file names fill cable, session, view, type and material. Accept, flip or delete the dashed
   suggestions, or click the cable width (two edges) and each crossing (centre, a point on the
   top strand, a point on the bottom strand) yourself; set visibility, save each record, and
   download `records.jsonl` into this folder. A second annotator on a subset gives
   inter-annotator agreement.
   This is the test set: take the over/under of every crossing from the cable (its tie sheet),
   not from the model, and flip any suggestion that disagrees. Every record says which items
   came from the model, were corrected or were drawn by hand (`source`, `assist`); the second
   annotator should work without suggestions, so the two can be compared.
4. `python scripts/check_real_photos.py` must report 0 problems.
5. Score once: `python scripts/eval_real_photos.py data/knotbench_photo/real_v1`.
   The run is logged in `results/heldout_ledger.jsonl`; a different main model is refused.
   Baselines: add `--role baseline --ckpt <spec>`.

Expected effort: about 2 hours of tying and photographing, 3-4 hours of annotation.
