"""
make_parser_dev_set.py - A dev set for parser settings, disjoint from KnotBench-Photo and from training.

New random cables (seed range distinct from both), rendered in training-style sessions and as
silhouettes (which carry no over/under evidence). Parser hyper-parameters (consistency exponent
gamma, pruning) are chosen here, never on the benchmark.
Output: data/knot_aware/parser_dev/records.jsonl + images
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np

from knot_recognition.cable import synth
from knot_recognition.cable.diagram import from_curves_3d
from knot_recognition.cable.identify import identify_diagram, type_key
sys.path.insert(0, str(ROOT / "scripts"))
from make_knot_aware_data import random_session, random_word
from run_knotbench_photo import well_posed

OUT = ROOT / "data" / "knot_aware" / "parser_dev"


def main(n_per_style=40, seed=3003):
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    recs, made = [], {"photo": 0, "silhouette": 0}
    while min(made.values()) < n_per_style:
        word = random_word(rng)
        s = int(rng.integers(1 << 30)) + (3 << 31)
        curves = synth.make_instance(word, s, wobble=0.15)
        try:
            gt = type_key(identify_diagram(from_curves_3d(curves)))
        except Exception:
            continue
        style = "photo" if made["photo"] <= made["silhouette"] else "silhouette"
        if style == "photo":
            synth.SESSIONS["_dev"] = random_session(rng)
            sess = "_dev"
        else:
            sess = "S0_silhouette"
        colour = list(synth.CABLE_COLOURS)[int(rng.integers(len(synth.CABLE_COLOURS)))]
        img, ann, D = synth.render(curves, sess, colour, int(rng.integers(7, 13)), np.random.default_rng(s))
        if not ann["crossings"] or len(ann["crossings"]) > 16 or not well_posed(D, ann["cable_width_px"]):
            continue
        iid = f"dev{len(recs):03d}"
        img.save(OUT / "images" / f"{iid}.jpg", quality=95)
        recs.append({"image_id": iid, "file": f"images/{iid}.jpg", "style": style, "cable_id": iid,
                     "session_id": style, "ground_truth": {"type": gt}, "annotation": ann})
        made[style] += 1
    with open(OUT / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(made)


if __name__ == "__main__":
    main()
