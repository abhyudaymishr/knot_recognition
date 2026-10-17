"""The annotator's model server: suggestions land on the true crossings and are valid JSON."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

MODEL = ROOT / "checkpoints" / "beat_unet" / "mps_uf_s0.pth"
PHOTO = ROOT / "data" / "knotbench_photo" / "synthetic_v1" / "images" / "c001_S2_desk.jpg"


@pytest.mark.skipif(not (MODEL.exists() and PHOTO.exists()), reason="needs the trained UNetFormer and the benchmark photo")
def test_suggestions_land_on_the_true_crossings():
    from annotator_server import Suggester, clean

    out = Suggester(MODEL, ROOT / "checkpoints" / "strand_pointer" / "mps_ptr_s0.pth").suggest(PHOTO.read_bytes())
    json.dumps(clean(out), allow_nan=False)                                  # what the page receives
    recs = [json.loads(l) for l in open(ROOT / "data/knotbench_photo/synthetic_v1/records.jsonl")]
    gt = next(r for r in recs if r["image_id"] == "c001_S2_desk")["annotation"]
    g = np.array([c["pos_px"] for c in gt["crossings"]])
    p = np.array([c["pos"] for c in out["crossings"]])
    assert len(p) == len(g)
    assert max(np.min(np.linalg.norm(g - q, axis=1)) for q in p) < 0.5 * gt["cable_width_px"]
    for c in out["crossings"]:                                                # the strand on top agrees with the truth
        j = int(np.argmin(np.linalg.norm(g - np.array(c["pos"]), axis=1)))
        vo = np.array(c["o"]) - np.array(c["pos"]); vo /= np.linalg.norm(vo)
        tc = gt["crossings"][j]
        assert abs(vo @ tc["over_tangent_px"]) > abs(vo @ tc["under_tangent_px"])
    assert out["lines"] and out["lines_from"] == "parser"
    assert out["reading"]["answer"] == "0_1"


def test_clean_makes_json_safe():
    from annotator_server import clean

    s = json.dumps(clean({"a": float("nan"), "b": [np.float32(1.5), np.int64(3), float("inf")]}), allow_nan=False)
    assert json.loads(s) == {"a": None, "b": [1.5, 3, None]}
