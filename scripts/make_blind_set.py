"""
make_blind_set.py - Build and freeze the blind real-picture set (blind_v1) WITHOUT viewing it.

Sources are independent of everything used during development, and nothing in them was
looked at before the model was frozen (results/frozen_model.json):

  drawings  Hankin's knotR diagrams (GitHub RobinHankin/knotR, GPL-2), drawn by knotR's own
            knotplot() in R. Labels from the package's dataset names (Rolfsen table <= 10
            crossings, the square knot, 3_1 # 4_1, and two unknot diagrams).
  renders   KnotPlot-derived 3D centrelines of all prime knots <= 9 crossings and the
            2-component links (GitHub prideout/knotess, MIT), ray-traced with POV-Ray
            (perspective camera, area light, Phong shading, optional backdrop with shadow),
            2 views each
            (one near the principal-plane normal, one random). Labels computed exactly from the 3D curve.

Photographs: none could be obtained in this environment (Wikimedia Commons, Knot Atlas and
arXiv are outside its network allowlist). See scripts/collect_blind_photos.py, to be run
on a machine with normal internet access.

Near-duplicates of raw_knot/ (difference hash, Hamming distance <= 8) are dropped
automatically. The manifest (with SHA-256 of every image) is frozen; its hash identifies
the set in results/heldout_ledger.jsonl.

Usage: python scripts/make_blind_set.py --knotr <knotR clone> --knotess <knotess clone> --raw <raw_knot>
Output: data/blind_real_v1/{images/, manifest.jsonl, MANIFEST_SHA256}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
from PIL import Image

OUT = ROOT / "data" / "blind_real_v1"

# knotR dataset name -> label (type_key format). Ambiguous art pieces are excluded.
SPECIAL = {"reefknot": "3_1 # 3_1*", "unknot": "0_1", "ochiai": "0_1"}
EXCLUDE_PREFIX = ("k11", "k12", "k_infinity")


def dhash(img: Image.Image, n: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.LANCZOS), float)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ knotR drawings
def knotr_items(knotr_dir: Path):
    names = subprocess.check_output(
        ["Rscript", "-e", 'library(knotR); d <- data(package="knotR")$results; cat(paste(d[,"Item"], collapse="\\n"))'],
        text=True).split("\n")
    items = []
    for item in names:
        obj = item.split(" (")[0]
        ds = item.split("(")[1].rstrip(")") if "(" in item else item
        if obj.startswith(EXCLUDE_PREFIX):
            continue
        if obj.startswith("k") and obj[1].isdigit():
            core = obj[1:].split("_")
            label = f"{core[0]}_{''.join(ch for ch in core[1] if ch.isdigit())}"
            if int(core[0]) > 10:
                continue
            items.append((ds, obj, label))
        elif obj in SPECIAL:
            items.append((ds, obj, SPECIAL[obj]))
        elif obj == "sum_31_41":
            items.append((ds, obj, "3_1 # 4_1"))
    return items


def render_knotr(items, img_dir: Path):
    script = ["library(knotR)"]
    for ds, obj, _ in items:
        f = img_dir / f"knotr_{obj}.png"
        script.append(
            f'e <- new.env(); data(list="{ds}", envir=e); o <- get(ls(e)[1], envir=e); '
            f'png("{f}", width=640, height=640); par(mar=c(1,1,1,1)); '
            f'tryCatch(knotplot(o), error=function(err) message("FAIL {obj}")); dev.off()')
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as t:
        t.write("\n".join(script))
    subprocess.run(["Rscript", t.name], check=True, capture_output=True)


# ------------------------------------------------------------------ knotess renders
def knotess_links(knotess_dir: Path):
    src = (knotess_dir / "index.js").read_text()
    body = src.split("const Links = ", 1)[1].split(";", 1)[0]
    links = json.loads(body.replace("'", '"'))
    spines = np.frombuffer((knotess_dir / "centerlines.bin").read_bytes(), dtype="<f4")
    out = {}
    for entry in links:
        lid, ranges = entry[0], entry[1:]
        if not ranges:
            continue
        comps = []
        for off, cnt in ranges:
            pts = spines[off * 3: off * 3 + cnt * 3].reshape(-1, 3).astype(float)
            comps.append(bezier_closed(pts))
        out[lid] = comps
    return out


def bezier_closed(P: np.ndarray, slices: int = 12) -> np.ndarray:
    """knotess' closed spline: for each control point b with neighbours a, c, a cubic Bezier
    from (a+b)/2 to (b+c)/2 pulled 1/3 towards b."""
    n = len(P)
    pts = []
    ts = np.linspace(0, 1, slices, endpoint=False)
    for i in range(n):
        a, b, c = P[i], P[(i + 1) % n], P[(i + 2) % n]
        v1, v4 = (a + b) / 2, (b + c) / 2
        v2, v3 = v1 + (b - v1) / 3, v4 + (b - v4) / 3
        for t in ts:
            u = 1 - t
            pts.append(u ** 3 * v1 + 3 * u * u * t * v2 + 3 * u * t * t * v3 + t ** 3 * v4)
    return np.array(pts)


def exact_label(comps) -> str:
    from knot_recognition.cable.diagram import from_curves_3d
    from knot_recognition.cable.identify import identify_diagram, type_key

    return type_key(identify_diagram(from_curves_3d(comps)))


def pov_scene(comps, rng, size=512, view: str = "random") -> str:
    """view="front": within 25 degrees of the curve's principal plane normal (how knots are
    usually presented); view="random": any direction."""
    allp = np.vstack(comps)
    ctr = allp.mean(0)
    R = float(np.max(np.linalg.norm(allp - ctr, axis=1)))
    if view == "front":
        nrm = np.linalg.svd(allp - ctr, full_matrices=False)[2][-1]
        while True:
            d = nrm + np.tan(np.radians(25)) * rng.uniform(0, 1) * rng.normal(size=3)
            d /= np.linalg.norm(d)
            if abs(d @ nrm) > np.cos(np.radians(25)):
                break
        d = d * rng.choice([-1, 1])
    else:
        d = rng.normal(size=3); d /= np.linalg.norm(d)
    ang = 32.0
    dist = R / np.sin(np.radians(ang / 2)) * rng.uniform(1.08, 1.3)
    cam = ctr + d * dist
    sky = np.cross(d, rng.normal(size=3)); sky /= np.linalg.norm(sky)
    r = R * rng.uniform(0.035, 0.075)
    col = rng.uniform(0.05, 0.95, 3)
    if col.max() - col.min() < 0.3:
        col[rng.integers(3)] = 0.9
    bg = rng.uniform(0.75, 1.0) * np.ones(3) if rng.random() < 0.6 else rng.uniform(0.1, 0.95, 3)
    light = ctr + (d + rng.normal(scale=0.6, size=3)) * dist * 1.5
    lines = [
        "#version 3.7;", "global_settings { assumed_gamma 1.0 }",
        f"camera {{ perspective location <{cam[0]:.4f},{cam[1]:.4f},{cam[2]:.4f}> sky <{sky[0]:.4f},{sky[1]:.4f},{sky[2]:.4f}> "
        f"up <0,1,0> right <1,0,0> angle {ang} look_at <{ctr[0]:.4f},{ctr[1]:.4f},{ctr[2]:.4f}> }}",
        f"background {{ color rgb <{bg[0]:.3f},{bg[1]:.3f},{bg[2]:.3f}> }}",
        f"light_source {{ <{light[0]:.3f},{light[1]:.3f},{light[2]:.3f}> color rgb 1.1 "
        f"area_light <{R:.3f},0,0>, <0,{R:.3f},0>, 4, 4 adaptive 1 jitter }}",
        f"light_source {{ <{cam[0]:.3f},{cam[1]:.3f},{cam[2]:.3f}> color rgb 0.35 shadowless }}",
    ]
    if rng.random() < 0.5:  # backdrop plane behind the knot, receives the shadow
        pc = ctr - d * R * 1.3
        lines.append(f"plane {{ <{d[0]:.4f},{d[1]:.4f},{d[2]:.4f}>, {float(d @ pc):.4f} "
                     f"pigment {{ rgb <{bg[0]:.3f},{bg[1]:.3f},{bg[2]:.3f}> }} finish {{ diffuse 0.9 ambient 0.25 }} }}")
    tex = (f"texture {{ pigment {{ rgb <{col[0]:.3f},{col[1]:.3f},{col[2]:.3f}> }} "
           f"finish {{ ambient 0.12 diffuse 0.75 phong {rng.uniform(0.3, 0.9):.2f} phong_size 40 specular 0.2 }} }}")
    for c in comps:
        step = max(1, len(c) // 360)
        P = c[::step]
        objs = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            objs.append(f"sphere {{ <{a[0]:.4f},{a[1]:.4f},{a[2]:.4f}>, {r:.4f} }}")
            objs.append(f"cylinder {{ <{a[0]:.4f},{a[1]:.4f},{a[2]:.4f}>, <{b[0]:.4f},{b[1]:.4f},{b[2]:.4f}>, {r:.4f} }}")
        lines.append("union { " + " ".join(objs) + " " + tex + " }")
    return "\n".join(lines)


def render_pov(scene: str, out: Path, size=512):
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "s.pov"
        f.write_text(scene)
        subprocess.run(["povray", f"+I{f}", f"+O{out}", f"+W{size}", f"+H{size}", "+A0.2", "+AM2", "-D", "+Q9", "Display=Off"],
                       check=True, capture_output=True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knotr", required=True)
    ap.add_argument("--knotess", required=True)
    ap.add_argument("--raw", required=True, help="raw_knot folder (for near-duplicate removal)")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--seed", type=int, default=9091)
    a = ap.parse_args()
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    items = knotr_items(Path(a.knotr))
    render_knotr(items, img_dir)
    for ds, obj, label in items:
        f = img_dir / f"knotr_{obj}.png"
        if f.exists() and np.asarray(Image.open(f).convert("L")).min() < 128:
            rows.append({"image_id": f"knotr_{obj}", "file": f"images/{f.name}", "category": "drawing",
                         "source": "knotR (Hankin), knotplot()", "license": "GPL-2", "label": label,
                         "label_from": "knotR dataset name"})

    rng = np.random.default_rng(a.seed)
    links = knotess_links(Path(a.knotess))
    for lid, comps in links.items():
        try:
            label = exact_label(comps)
        except Exception as e:  # noqa
            print("skip", lid, str(e)[:60])
            continue
        if label.startswith("unknown"):
            continue
        for v in range(a.views):
            iid = f"knotess_{lid.replace('.', '_')}_v{v}"
            out = img_dir / f"{iid}.png"
            view = "front" if v == 0 else "random"
            render_pov(pov_scene(comps, rng, view=view), out)
            rows.append({"image_id": iid, "file": f"images/{out.name}", "category": "render",
                         "source": "knotess centrelines (KnotPlot-derived), POV-Ray", "license": "MIT (geometry)",
                         "label": label, "label_from": "exact, from the 3D curve", "knotess_id": lid, "view": view})

    # near-duplicates of the development pictures
    raw = [dhash(Image.open(p)) for p in Path(a.raw).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif")]
    kept = []
    for r in rows:
        h = dhash(Image.open(OUT / r["file"]))
        dmin = min(bin(h ^ x).count("1") for x in raw) if raw else 64
        r["dhash_min_distance_to_raw_knot"] = dmin
        if dmin <= 8:
            r["excluded"] = "near-duplicate of a development picture"
        r["sha256"] = sha(OUT / r["file"])
        kept.append(r)
    with open(OUT / "manifest.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    msha = sha(OUT / "manifest.jsonl")
    (OUT / "MANIFEST_SHA256").write_text(msha + "\n")
    n_ok = sum("excluded" not in r for r in kept)
    print(f"{len(kept)} images ({n_ok} kept): "
          f"{sum(r['category'] == 'drawing' for r in kept)} drawings, {sum(r['category'] == 'render' for r in kept)} renders; manifest {msha[:12]}")


if __name__ == "__main__":
    main()
