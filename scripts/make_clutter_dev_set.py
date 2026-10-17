"""
make_clutter_dev_set.py - Development renders for the parser's clutter filter, disjoint from
KnotBench-Photo and from training.

New random cables (seed range distinct from the benchmark, training and parser_dev), rendered in
the benchmark's unseen session STYLES (S3 dim, S4 clutter). No benchmark image is used. Parser
settings for cluttered backgrounds are checked here; the benchmark's own S3/S4 images and the blind
sets then test them.
Output: data/knot_aware/clutter_dev/records.jsonl + images
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np

from knot_recognition.cable import synth
from knot_recognition.cable.diagram import from_curves_3d
from knot_recognition.cable.identify import identify_diagram, type_key
from make_knot_aware_data import random_word
from run_knotbench_photo import well_posed

OUT = ROOT / "data" / "knot_aware" / "clutter_dev"


def main(n_per_session=40, seed=5005):
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    recs, made = [], {"S3_dim": 0, "S4_clutter": 0}
    while min(made.values()) < n_per_session:
        word = random_word(rng)
        s = int(rng.integers(1 << 30)) + (5 << 31)
        curves = synth.make_instance(word, s, wobble=0.15)
        try:
            gt = type_key(identify_diagram(from_curves_3d(curves)))
        except Exception:  # noqa
            continue
        sess = "S3_dim" if made["S3_dim"] <= made["S4_clutter"] else "S4_clutter"
        colour = list(synth.CABLE_COLOURS)[int(rng.integers(len(synth.CABLE_COLOURS)))]
        img, ann, D = synth.render(curves, sess, colour, int(rng.integers(7, 13)), np.random.default_rng(s))
        if not ann["crossings"] or len(ann["crossings"]) > 16 or not well_posed(D, ann["cable_width_px"]):
            continue
        iid = f"clut{len(recs):03d}"
        img.save(OUT / "images" / f"{iid}.jpg", quality=95)
        recs.append({"image_id": iid, "file": f"images/{iid}.jpg", "style": sess, "cable_id": iid,
                     "session_id": sess, "ground_truth": {"type": gt}, "annotation": ann})
        made[sess] += 1
    with open(OUT / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(made)


if __name__ == "__main__":
    main()
