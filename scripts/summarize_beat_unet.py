"""summarize_beat_unet.py - The matched comparison against the U-Net pair (scripts/run_beat_unet.sh):
same data, one-stage recipe, Apple GPU, clutter filter on in the parser; three seeds each.
Per model: benchmark answered (all / unseen S3-S4), wrong; close strands; real pictures; blind v2.
Paired differences use the U-Net pair trained with the same seed. Output: results/beat_unet_summary.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts")]
from make_cv_tables import _set_files  # noqa: E402
from summarize_variants import metrics  # noqa: E402

MODELS = {"U-Net pair": "bu_unet", "UNetFormer": "bu_uf", "UNetFormer + clDice": "bu_uf_topo",
          "NanoTransformer 2.9M, one stage": "bu_nano3m"}
KEYS = ("kb", "kb_unseen", "kb_wrong", "close", "close_wrong", "real", "real_wrong", "b2", "b2_wrong")


def main():
    out = {}
    for lab, pre in MODELS.items():
        seeds = {s: metrics(_set_files(f"{pre}_s{s}")) for s in (0, 1, 2)}
        seeds = {s: m for s, m in seeds.items() if all(k in m for k in KEYS)}
        out[lab] = {"seeds": seeds}
    base = out["U-Net pair"]["seeds"]
    for lab, d in out.items():
        sd = d["seeds"]
        if not sd:
            continue
        d["mean"] = {k: float(np.mean([m[k] for m in sd.values()])) for k in KEYS}
        d["sum_wrong"] = {k: int(sum(m[k] for m in sd.values())) for k in KEYS if k.endswith("wrong")}
        if lab != "U-Net pair":
            d["paired_vs_unet"] = {k: [round((100 if k in ("kb", "kb_unseen", "close") else 1) * (sd[s][k] - base[s][k]), 1)
                                       for s in sd if s in base] for k in KEYS}
        pr = d.get("paired_vs_unet", {})
        f = lambda k: f"{100 * d['mean'][k]:.1f}%" if k in ("kb", "kb_unseen", "close") else f"{d['mean'][k]:.1f}"
        print(f"{lab:<34} n={len(sd)} KB {f('kb')} (unseen {f('kb_unseen')}, wrong {d['sum_wrong']['kb_wrong']}) "
              f"close {f('close')} ({d['sum_wrong']['close_wrong']}) real {f('real')} ({d['sum_wrong']['real_wrong']}) "
              f"v2 {f('b2')} ({d['sum_wrong']['b2_wrong']})" + (f"\n{'':36}paired vs U-Net: unseen {pr['kb_unseen']}  v2 {pr['b2']}  real {pr['real']}" if pr else ""))
    out["image_level_vs_unet"] = image_level()
    for k, v in out["image_level_vs_unet"].items():
        print(f"{k}: {v}")
    json.dump(out, open(ROOT / "results" / "beat_unet_summary.json", "w"), indent=1)


def _mcnemar(a, b):
    from math import comb
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def image_level():
    """Per seed: images only the UNetFormer gets right vs only the U-Net pair (exact McNemar test)."""
    base = lambda t: None if t is None else t.replace("*", "").replace(" # ", "#")
    res = {}
    for set_name, path, key in (("blind_v2", "results/blind/blind_real_v2_{}.json", "image_id"),
                                ("real", "results/real_knot_pictures_{}.json", "file")):
        rows, A, B = [], 0, 0
        for sd in (0, 1, 2):
            f_u, f_f = ROOT / path.format(f"bu_unet_s{sd}"), ROOT / path.format(f"bu_uf_s{sd}")
            if not (f_u.exists() and f_f.exists()):
                continue
            def ok(r):
                if set_name == "real":
                    return r["knot_aware"] is not None and base(r["knot_aware"]) == base(r["label"])
                return bool(r["correct"])
            cu = {r[key]: ok(r) for r in json.load(open(f_u))["per_image"]}
            cf = {r[key]: ok(r) for r in json.load(open(f_f))["per_image"]}
            a = sum(cf[k] and not cu[k] for k in cf)
            b = sum(cu[k] and not cf[k] for k in cf)
            A, B = A + a, B + b
            rows.append({"seed": sd, "only_unetformer": a, "only_unet": b, "p": round(_mcnemar(a, b), 4)})
        res[set_name] = {"per_seed": rows, "pooled_only_unetformer": A, "pooled_only_unet": B,
                         "pooled_p_(seeds_not_independent)": round(_mcnemar(A, B), 5)}
    return res


if __name__ == "__main__":
    main()
