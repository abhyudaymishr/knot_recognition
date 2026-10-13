"""
eval_denoise.py - Does FFT denoising + measure-stable segmentation help classical cable segmentation?

For every KnotBench-Photo synthetic_v1 image (true crossing count known) compares
  otsu          Otsu threshold on grey (the previous Euler-check mask)
  stable        measure-stable superlevel set of grey (cable.denoise.stable_segmentation)
  fft+stable    mirror-padded homomorphic + Wiener filtering, then the stable set
by mask IoU against the ground-truth mask and the Euler check with the true count.
Output: results/knotbench_photo/denoise_ablation.json
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
from PIL import Image
from knot_recognition.cable import denoise as dn
from knot_recognition.cable.consistency import cable_mask, euler_check
from knot_recognition.cable.synth import gt_mask

DATA = ROOT / "data" / "knotbench_photo" / "synthetic_v1"


def polarity(r):
    q_lo, q_med, q_hi = np.percentile(r, [2, 50, 98])
    return -r if (q_med - q_lo) > (q_hi - q_med) else r


def main():
    recs = [json.loads(l) for l in open(DATA / "records.jsonl")]
    res = defaultdict(lambda: defaultdict(Counter))
    for r in recs:
        img = np.asarray(Image.open(DATA / r["file"]).convert("RGB"))
        ann = r["annotation"]; n = len(ann["crossings"]); w = ann["cable_width_px"]; G = gt_mask(ann)
        g = dn.to_gray(img)
        h = dn.homomorphic(g, d0=1 / 256, g_low=0.5); h, _ = dn.wiener(h)
        masks = {"otsu": cable_mask(img),
                 "stable": dn.stable_segmentation(polarity(g), w)[0],
                 "fft+stable": dn.stable_segmentation(polarity(h), w)[0]}
        for name, m in masks.items():
            c = res[name][r["session_id"]]
            c["n"] += 1
            c["euler_pass"] += bool(euler_check(img, n, w, mask=m)["consistent"])
            c["iou_sum"] += float((m & G).sum() / max((m | G).sum(), 1))
    out = {name: {s: {"images": c["n"], "euler_pass_rate_true_count": c["euler_pass"] / c["n"],
                      "mask_iou": c["iou_sum"] / c["n"]} for s, c in sess.items()} for name, sess in res.items()}
    json.dump(out, open(ROOT / "results" / "knotbench_photo" / "denoise_ablation.json", "w"), indent=1)
    for name, sess in out.items():
        print(f"{name:<11}" + "  ".join(f"{s[:9]} E{v['euler_pass_rate_true_count']:.2f} IoU{v['mask_iou']:.2f}" for s, v in sorted(sess.items())))


if __name__ == "__main__":
    main()
