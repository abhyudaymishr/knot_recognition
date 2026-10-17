"""
summarize_uf_pointer.py - The UNetFormer trained together with the pointer head (joint), against
the plain UNetFormer of the same seed with a pointer head trained on top of it (frozen).

Per seed, on held-out validation images 150-299:
  strand order: links, crossing jumps, images with every link right (head alone); jumps after the
  curved re-check with re-linking; and with the combined re-check (strand_pointer.recheck):
  jumps found, wrong long links, images with every link right.
UNetFormer maps (validation loss, crossing precision/recall, over/under axis) of the joint trunk
against the plain trunk, and, when scripts/eval_suite.py has been run on them, the parser's
benchmark coverage (all / unseen), close strands and real pictures.

  python scripts/summarize_uf_pointer.py [--device mps]
Output: results/uf_pointer_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

R = ROOT / "results" / "uf_pointer"


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def strand(held, rc):
    """Pointer-level numbers from eval_strand_pointer (held) and eval_recheck (rc)."""
    out = {}
    for mode in ("true", "ridge"):
        m = {}
        if held:
            d = held["by_points"][mode]
            m.update(links=d["link_recall"]["head"], jumps=d["jump_recall"]["head"],
                     exact=d["exact_images"]["head"], jumps_relinked=d["recheck"]["jump_recall"]["head_check"])
        if rc:
            c = rc["results"][mode]["curved relink + straight balanced"]
            h = rc["results"][mode]["head alone"]
            m.update(recheck_jumps=c["jumps_found"], recheck_wrong_long=c["wrong_long_links"],
                     recheck_exact=c["images_every_link_right"], wrong_long=h["wrong_long_links"])
        out[mode] = m
    return out


def trunk_val(path):
    p = Path(path)
    if not p.exists():
        return None
    v = torch.load(p, map_location="cpu", weights_only=False).get("extra", {}).get("val", {})
    return {k: v.get(k) for k in ("loss", "crossing_precision", "crossing_recall", "over_axis_within_45deg", "center_hi", "mask")}


def parser(name):
    from make_cv_tables import _set_files
    from summarize_variants import metrics
    m = metrics(_set_files(name))
    return m or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", help="prefix of the checkpoints (the device they were trained on)")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    D = a.device
    res = {}
    for s in map(int, a.seeds.split(",")):
        joint = strand(jload(R / f"heldout_{D}_ufptr_s{s}.json"), jload(R / f"recheck_{D}_ufptr_s{s}.json"))
        frozen = strand(jload(R / f"heldout_{D}_ptr_s{s}.json"), jload(R / f"recheck_{D}_ptr_s{s}.json"))
        res[s] = {"strand_order": {"joint": joint, "frozen": frozen},
                  "unetformer_val": {"joint": trunk_val(ROOT / f"checkpoints/uf_pointer/{D}_ufptr_s{s}.pth"),
                                     "plain": trunk_val(ROOT / f"checkpoints/beat_unet/mps_uf_s{s}.pth"),
                                     "control": trunk_val(ROOT / f"checkpoints/uf_pointer/{D}_ufctl_s{s}.pth")},
                  "parser": {"joint": parser(f"ufptr_s{s}"), "plain": parser(f"bu_uf_s{s}"), "control": parser(f"ufctl_s{s}")}}

    pct = lambda v: "   -  " if v is None else f"{100 * v:5.1f}%"
    num = lambda v: "  - " if v is None else f"{v:4d}"
    print("Strand order on held-out validation images 150-299 (joint = UNetFormer trained with the head;")
    print("frozen = head trained on the plain UNetFormer of the same seed)")
    for mode, label in (("true", "points on the true strands"), ("ridge", "points on the predicted centre-line")):
        print(f"\n  {label}:")
        print(f"  {'seed':>4} {'model':>7} {'links':>7} {'jumps':>7} {'exact':>6} | {'re-checked: jumps':>17} {'wrong long':>10} {'exact':>6}")
        for s, r in res.items():
            for who in ("joint", "frozen"):
                m = r["strand_order"][who][mode]
                if not m:
                    continue
                print(f"  {s:>4} {who:>7} {pct(m.get('links'))} {pct(m.get('jumps'))} {num(m.get('exact'))}   | "
                      f"{pct(m.get('recheck_jumps')):>17} {num(m.get('recheck_wrong_long')):>10} {num(m.get('recheck_exact')):>6}")
        for who in ("joint", "frozen"):
            vals = [r["strand_order"][who][mode] for r in res.values() if r["strand_order"][who][mode].get("jumps") is not None]
            if vals:
                print(f"  mean {who:>7} {pct(np.mean([v['jumps'] for v in vals]))} jumps; re-checked "
                      f"{pct(np.mean([v.get('recheck_jumps', np.nan) for v in vals]))}, wrong long "
                      f"{np.sum([v.get('recheck_wrong_long', 0) for v in vals])} (n = {len(vals)})")
    print("\nUNetFormer maps on validation (loss; crossing precision / recall; over/under axis within 45 deg):")
    for s, r in res.items():
        for who in ("joint", "plain", "control"):
            v = r["unetformer_val"][who]
            if v:
                print(f"  seed {s} {who:>7}: loss {v['loss']:.4f}; crossings {v['crossing_precision']:.3f} / "
                      f"{v['crossing_recall']:.3f}; over/under {v['over_axis_within_45deg']:.4f}")
    if any(r["parser"]["joint"] for r in res.values()):
        print("\nParser with each UNetFormer (clutter filter on): benchmark answered (unseen), close strands, real pictures")
        for s, r in res.items():
            for who in ("joint", "plain", "control"):
                m = r["parser"][who]
                if m and "kb" in m:
                    print(f"  seed {s} {who:>7}: {pct(m['kb'])} ({pct(m['kb_unseen'])}), wrong {m['kb_wrong']}; "
                          f"close {pct(m.get('close'))} ({m.get('close_wrong')}); real {m.get('real')} ({m.get('real_wrong')})")
    out = ROOT / "results" / "uf_pointer_summary.json"
    out.write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1, default=float))
    print(f"\n[saved] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
