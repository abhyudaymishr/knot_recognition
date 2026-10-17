"""
collect_blind_photos.py - Collect candidate PHOTOGRAPHS of knots for the blind real-picture set.

Run on a machine with normal internet access (Wikimedia Commons is outside this project's
cloud allowlist, so this script was written but could not be run against the live API here).

Steps
  1. collect: search Wikimedia Commons (file namespace, JPEG only, i.e. mostly photographs)
     for each knot name, download a 1024 px version, record title, URL, author and licence,
     and drop exact or near duplicates of raw_knot/ (difference hash).
  2. review: open data/blind_photos_v1/review.html, tick each picture that shows ONE physical
     knot of the proposed type (fix the type if needed), untick everything else (drawings,
     several knots, decorative knots of unclear type), and save decisions.json into the folder.
     Reviewers label; they never see model output.
  3. freeze: `python scripts/collect_blind_photos.py freeze` keeps accepted photos, writes
     manifest.jsonl + MANIFEST_SHA256 in the format scripts/eval_blind_set.py expects.
  4. score once: `python scripts/eval_blind_set.py --set data/blind_photos_v1`

Usage: python scripts/collect_blind_photos.py collect --raw raw_knot [--per-type 25]
       python scripts/collect_blind_photos.py freeze
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "blind_photos_v1"
API = "https://commons.wikimedia.org/w/api.php"
UA = "KnotBench-blind-set/1.0 (research; contact: repository owner)"

# search phrase -> proposed label (type_key format; reviewers confirm or correct)
TARGETS = {
    '"trefoil knot"': "3_1", '"overhand knot"': "3_1", '"figure-eight knot" OR "figure eight knot"': "4_1",
    '"cinquefoil knot" OR "pentafoil knot"': "5_1", '"three-twist knot"': "5_2", '"stevedore knot"': "6_1",
    '"granny knot"': "3_1 # 3_1", '"reef knot" OR "square knot"': "3_1 # 3_1*",
    '"Hopf link"': "link[0_1,0_1|interlinked|lk=[[0, 1], [1, 0]]]", '"unknot"': "0_1",
}
LABELS = ["0_1", "3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "6_3", "7_1", "3_1 # 3_1", "3_1 # 3_1*", "3_1 # 4_1",
          "link[0_1,0_1|interlinked|lk=[[0, 1], [1, 0]]]", "link[0_1,0_1|separable|lk=[[0, 0], [0, 0]]]"]


def api(**params):
    params.update(format="json")
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def dhash(img, n=8) -> int:
    import numpy as np
    from PIL import Image

    g = np.asarray(img.convert("L").resize((n + 1, n), Image.LANCZOS), float)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def collect(raw: Path, per_type: int):
    from PIL import Image

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    raw_names = {p.name.lower() for p in raw.rglob("*")}
    raw_hash = [dhash(Image.open(p)) for p in raw.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif")]
    cands, seen = [], set()
    for query, label in TARGETS.items():
        res = api(action="query", list="search", srnamespace=6, srlimit=100,
                  srsearch=f"{query} filemime:image/jpeg")
        titles = [h["title"] for h in res.get("query", {}).get("search", [])]
        kept = 0
        for i in range(0, len(titles), 20):
            info = api(action="query", prop="imageinfo", titles="|".join(titles[i:i + 20]),
                       iiprop="url|mime|size|extmetadata", iiurlwidth=1024)
            for page in info.get("query", {}).get("pages", {}).values():
                if kept >= per_type or "imageinfo" not in page:
                    continue
                t, ii = page["title"], page["imageinfo"][0]
                name = t.split(":", 1)[1]
                if t in seen or name.lower() in raw_names or ii.get("mime") != "image/jpeg":
                    continue
                req = urllib.request.Request(ii.get("thumburl") or ii["url"], headers={"User-Agent": UA})
                data = urllib.request.urlopen(req, timeout=60).read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                if min(bin(dhash(img) ^ h).count("1") for h in raw_hash) <= 8:
                    continue
                iid = "commons_" + hashlib.sha1(t.encode()).hexdigest()[:12]
                img.save(OUT / "images" / f"{iid}.jpg", quality=95)
                md = ii.get("extmetadata", {})
                cands.append({"image_id": iid, "file": f"images/{iid}.jpg", "title": t, "url": ii["descriptionurl"] if "descriptionurl" in ii else ii["url"],
                              "author": md.get("Artist", {}).get("value", ""), "license": md.get("LicenseShortName", {}).get("value", ""),
                              "proposed_label": label, "query": query})
                seen.add(t)
                kept += 1
            time.sleep(1)
        print(f"{query}: {kept}")
    with open(OUT / "candidates.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")
    write_review(cands)
    print(f"{len(cands)} candidates; open {OUT / 'review.html'}")


def write_review(cands):
    opts = "".join(f"<option>{l}</option>" for l in LABELS)
    cards = "".join(
        f'<div class="c" data-id="{c["image_id"]}"><img src="{c["file"]}"><label><input type="checkbox"> one physical knot, type:</label>'
        f'<select>{opts.replace(f"<option>{c["proposed_label"]}</option>", f"<option selected>{c["proposed_label"]}</option>")}</select>'
        f'<small>{c["title"]}</small></div>' for c in cands)
    (OUT / "review.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Blind photo review</title>
<style>body{{font:13px system-ui;margin:16px}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}}
.c{{border:1px solid #ccc;padding:6px}}img{{width:100%;height:220px;object-fit:contain;background:#eee}}small{{display:block;color:#666;word-break:break-all}}</style>
<h1>Blind photo review</h1><p>Tick a picture only if it is a photograph of ONE physical knot whose type you can confirm; set the type. Then save.</p>
<button id=s>Save decisions.json</button><div class=g>{cards}</div>
<script>document.getElementById('s').onclick=()=>{{const d={{}};document.querySelectorAll('.c').forEach(c=>{{d[c.dataset.id]={{accept:c.querySelector('input').checked,label:c.querySelector('select').value}}}});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(d,null,1)],{{type:'application/json'}}));a.download='decisions.json';a.click();}};</script>""")


def freeze():
    cands = [json.loads(l) for l in open(OUT / "candidates.jsonl")]
    dec = json.load(open(OUT / "decisions.json"))
    rows = []
    for c in cands:
        d = dec.get(c["image_id"], {})
        if not d.get("accept"):
            continue
        p = OUT / c["file"]
        rows.append({"image_id": c["image_id"], "file": c["file"], "category": "photo", "source": c["url"],
                     "license": c["license"], "author": c["author"], "label": d["label"], "label_from": "human review",
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    with open(OUT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    msha = hashlib.sha256((OUT / "manifest.jsonl").read_bytes()).hexdigest()
    (OUT / "MANIFEST_SHA256").write_text(msha + "\n")
    print(f"frozen {len(rows)} photos; manifest {msha[:12]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["collect", "freeze"])
    ap.add_argument("--raw", default=str(ROOT / "raw_knot"))
    ap.add_argument("--per-type", type=int, default=25)
    a = ap.parse_args()
    collect(Path(a.raw), a.per_type) if a.step == "collect" else freeze()
