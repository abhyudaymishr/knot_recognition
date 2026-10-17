"""summarize_variants.py - One line per evaluated configuration (benchmark, close strands, real
pictures, blind v1, blind v2) and per-group means; writes results/minkowski_summary.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts")]
from make_cv_tables import _set_files  # noqa: E402

GROUPS = {
    "plain (CPU)": ["frozen", "seed1", "seed2"],
    "lebesgue (CPU)": ["lebesgue"], "mk_bias (CPU)": ["mk_bias"],
    "mk_full (CPU)": ["mk_full", "mk_full_seed1", "mk_full_seed2"],
    "lebesgue (MPS)": ["mps_lebesgue", "mps_lebesgue_seed1", "mps_lebesgue_seed2"],
    "mk_bias (MPS)": ["mps_mk_bias", "mps_mk_bias_seed1", "mps_mk_bias_seed2"],
    "mk_full (MPS)": ["mps_mk_full", "mps_mk_full_seed1", "mps_mk_full_seed2"],
    "frozen + gap evidence": ["v3_gap"], "plain without FFT": ["nofft"], "U-Net pair": ["unet_seg+kp"],
}


def metrics(d):
    m = {}
    if d["kb"]:
        m.update(kb=d["kb"]["overall_photo_sessions"]["coverage"], kb_unseen=d["kb"]["unseen_sessions_S3_S4"]["coverage"],
                 kb_wrong=d["kb"]["overall_photo_sessions"]["wrong_confident_answers"])
    if d["close"]:
        m.update(close=d["close"]["all"]["coverage"], close_wrong=d["close"]["all"]["wrong_confident_answers"])
    if d["real"]:
        a = d["real"]["by_pipeline"]["knot_aware"]["all"]
        m.update(real=a["answered"], real_wrong=a["wrong"])
    for k in ("b1", "b2"):
        if d[k]:
            a = d[k]["by_category"]["all"]
            m.update({k: a["correct"], k + "_wrong": a["wrong"]})
    return m


def seed_of(name):
    return 1 if name.endswith("seed1") else 2 if name.endswith("seed2") else 0


def paired(out, group, base="plain (CPU)", keys=("kb", "kb_unseen", "real", "b1", "b2", "b2_wrong")):
    """Variant minus plain model trained with the SAME seed (same data order and augmentation)."""
    b = {seed_of(n): m for n, m in out[base]["members"].items()}
    diffs = {k: [] for k in keys}
    for n, m in out[group]["members"].items():
        for k in keys:
            if k in m and k in b.get(seed_of(n), {}):
                d = m[k] - b[seed_of(n)][k]
                diffs[k].append(round(100 * d, 1) if k in ("kb", "kb_unseen") else d)
    return {k: {"diffs": v, "mean": float(np.mean(v)) if v else None, "positive": sum(x > 0 for x in v),
                "negative": sum(x < 0 for x in v)} for k, v in diffs.items()}


def main():
    out = {}
    for g, names in GROUPS.items():
        rows = {n: metrics(_set_files(n)) for n in names}
        rows = {n: r for n, r in rows.items() if r}
        if not rows:
            continue
        keys = sorted({k for r in rows.values() for k in r})
        agg = {}
        for k in keys:
            v = [r[k] for r in rows.values() if k in r]
            agg[k] = {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                      "min": float(min(v)), "max": float(max(v)), "sum": float(sum(v))}
        out[g] = {"members": rows, "aggregate": agg}
        f = lambda k, pct=False: (f"{100 * agg[k]['mean']:.0f}" if pct else f"{agg[k]['mean']:.1f}") if k in agg else "--"
        s = lambda k: f"{agg[k]['sum']:.0f}" if k in agg else "-"
        print(f"{g:<24} n={len(rows)}  KB {f('kb', 1)}% (unseen {f('kb_unseen', 1)}%, wrong {s('kb_wrong')})  close {f('close', 1)}% "
              f"(wrong {s('close_wrong')})  real {f('real')} (wrong {s('real_wrong')})  v1 {f('b1')} (wrong {s('b1_wrong')})  "
              f"v2 {f('b2')} (wrong {s('b2_wrong')})")
    print("\npaired by seed (variant - plain CPU model with the same seed; kb in points):")
    for g in [g for g in out if g != "plain (CPU)" and any(seed_of(n) in (0, 1, 2) for n in out[g]["members"])]:
        if g in ("U-Net pair", "plain without FFT", "frozen + gap evidence"):
            continue
        out[g]["paired_vs_plain_cpu"] = pr = paired(out, g)
        print(f"  {g:<18}", {k: (v["diffs"], f"+{v['positive']}/-{v['negative']}") for k, v in pr.items() if v["diffs"]})
    json.dump(out, open(ROOT / "results" / "minkowski_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
