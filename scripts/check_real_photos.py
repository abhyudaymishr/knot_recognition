"""
check_real_photos.py - Validate the real cable photo set before scoring it once.

Checks data/knotbench_photo/real_v1 (or the folder given):
  * every photo in shot_list.csv exists in images/ and has a record in records.jsonl
  * every record passes cable.photo_bench.validate_record
  * each record's ground-truth type, cable and session agree with cables.csv / its file name
  * each record has a cable width and at least one crossing unless its cable is an unknot loop
Prints a summary; exits non-zero if anything blocks scoring. Does not run any model.

Usage: python scripts/check_real_photos.py [folder]
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from knot_recognition.cable.photo_bench import validate_record


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "knotbench_photo" / "real_v1"
    cables = {r["cable_id"]: r for r in csv.DictReader(open(folder / "cables.csv"))}
    shots = list(csv.DictReader(open(folder / "shot_list.csv")))
    recs = {}
    if (folder / "records.jsonl").exists():
        for l in open(folder / "records.jsonl"):
            if l.strip():
                r = json.loads(l)
                recs[Path(r["file"]).name] = r
    problems, warnings = [], []
    for s in shots:
        f = s["file"]
        if not (folder / "images" / f).exists():
            problems.append(f"missing photo images/{f}")
        if f not in recs:
            problems.append(f"no annotation for {f}")
    for name, r in recs.items():
        for e in validate_record(r):
            problems.append(f"{name}: {e}")
        m = re.match(r"^(c\d+)_(s\w+?)_v(\d+)", name)
        if not m:
            warnings.append(f"{name}: file name does not follow cXX_sY_vZ")
            continue
        cid, sid = m.group(1), m.group(2)
        if r.get("cable_id") != cid or r.get("session_id") != sid:
            problems.append(f"{name}: cable/session fields disagree with the file name")
        want = cables.get(cid, {}).get("ground_truth_type")
        if want and r["ground_truth"]["type"] != want:
            problems.append(f"{name}: type {r['ground_truth']['type']} but cables.csv says {want}")
        ann = r.get("annotation", {})
        if not ann.get("cable_width_px"):
            problems.append(f"{name}: no cable width")
        if not ann.get("crossings") and want != "0_1":
            warnings.append(f"{name}: no crossings annotated")
        if not (folder / r["file"]).exists():
            problems.append(f"{name}: record points to missing {r['file']}")
    n_cables = len({r.get("cable_id") for r in recs.values()})
    n_sessions = len({r.get("session_id") for r in recs.values()})
    print(f"{len(recs)} annotated photos of {n_cables} cables in {n_sessions} sessions; "
          f"{len(shots)} planned; {len(problems)} problem(s), {len(warnings)} warning(s)")
    for p in problems[:40]:
        print("  PROBLEM", p)
    for w in warnings[:20]:
        print("  warning", w)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
