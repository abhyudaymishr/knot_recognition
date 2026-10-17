"""
annotate.py - The trained UNetFormer's suggestions for annotating one cable photo.

Used by the local photo annotator (scripts/annotator_server.py) and by the Hugging Face Space
(huggingface_space/app.py). For one photo, in that photo's own pixels:
  width        the cable width and a two-point segment across the cable
  crossings    centre, a point on the strand on top, a point on the strand underneath, and the
               probability that this over/under reading is right (under_guessed: the strand below
               was not traced and its direction is a guess)
  ends         strand ends that do not reach a crossing
  lines        each strand in travel direction: the parser's traced strands, or, when the parser
               cannot trace them, the strand-order head's ordered points (links re-checked with
               strand_pointer.recheck)
  reading      the system's answer (type, probability) or the reasons it abstains
The photo is letterboxed to 512 x 512 as in scripts/eval_real_photos.py and parsed with the clutter
filter on and the given kappa (26.5 for the UNetFormer, from its evaluation).

SnapPy (the reading of the type) installs signal handlers when it is first imported, which Python
allows only in the main thread: a server that answers in worker threads must import snappy at start.
"""

from __future__ import annotations

import hashlib
import io
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

DEFAULT_KAPPA = 26.5          # the UNetFormer's kappa (results/suite/bu_uf_s*/meta.json)


def sha12(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def clean(o):
    """JSON-safe: NaN / infinity -> null (browsers reject them), numpy scalars -> numbers."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def open_photo(data) -> Image.Image:
    """bytes, a path or a PIL image -> upright RGB photo (EXIF rotation applied)."""
    if isinstance(data, Image.Image):
        return ImageOps.exif_transpose(data).convert("RGB")
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


class Suggester:
    def __init__(self, model_path, head_path=None, kappa: Optional[float] = None):
        from knot_recognition.knot_aware import load
        from knot_recognition.strand_pointer import load_head

        self.model_path = Path(model_path)
        self.model = load(self.model_path).eval()
        self.model.extra.setdefault("gamma", 2.0)
        self.kappa = kappa or self.model.extra.get("kappa") or DEFAULT_KAPPA
        self.model.extra["kappa"] = self.kappa
        self.head, self.head_path = None, None
        if head_path and Path(head_path).exists():
            self.head = load_head(head_path).eval()
            self.head_path = Path(head_path)
        self.info = {"model": self.model_path.name, "model_sha256": sha12(self.model_path),
                     "arch": self.model.cfg.arch if hasattr(self.model.cfg, "arch") else "knot_aware",
                     "kappa": self.kappa, "clutter_filter": True,
                     "strand_head": self.head_path.name if self.head is not None else None}
        import importlib.util
        # the reading of the type needs SnapPy / spherogram; suggestions do not
        self.can_identify = all(importlib.util.find_spec(m) is not None for m in ("snappy", "spherogram"))
        self.info["identification"] = self.can_identify
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pair_dirs(D, cid):
        """Pixel-space directions of the over and under strands at a parsed crossing
        (as scripts/eval_knot_aware.pair_dirs)."""
        c = next(x for x in D.crossings if x.cid == cid)
        h = D.meta.get("tangent_window")
        to = D.components[c.over.component].tangent_at(c.over.t, h)
        tu = D.components[c.under.component].tangent_at(c.under.t, h)
        return [to[0], -to[1]], [tu[0], -tu[1]], c

    @staticmethod
    def _back(p, s, off):
        """512-square letterbox pixel -> photo pixel."""
        return [float((p[0] - off[0]) / s), float((p[1] - off[1]) / s)]

    def _head_lines(self, sq):
        """Strands in order from the strand-order head (512-square pixels)."""
        from knot_recognition.knot_aware import preprocess
        from knot_recognition.strand_pointer import (Points, cut_radius, decode, heat_peaks, mask_width, recheck,
                                                     ridge_points, targets, walks)
        x = torch.from_numpy(preprocess(sq)).permute(2, 0, 1).float()[None] / 255.0
        with torch.no_grad():
            out = self.model(x, return_feats=True)
            ridge = torch.sigmoid(out["center_hi"])[0].numpy()
            w = mask_width(torch.sigmoid(out["mask"])[0].numpy())
            C = heat_peaks(torch.sigmoid(out["heat"])[0].numpy())
            xy = ridge_points(ridge, C, w, self.head.cfg.spacing)
            if len(xy) < 3:
                return []
            t, v, _, _ = targets([Points(xy, np.full(len(xy), -1), np.full(len(xy), np.nan), [])])
            R = cut_radius(w)
            p = torch.softmax(self.head(out["feats"], t, v, torch.tensor([R]) if self.head.cfg.radius else None), -1)[0].numpy()
        edges = recheck(decode(p), p, xy, ridge, R, self.head.cfg.spacing)
        lines = []
        for st in walks(edges, len(xy)):
            if len(st["order"]) < 8:              # pieces shorter than ~30 px only clutter the page
                continue
            o = st["order"] + ([st["order"][0]] if st["closed"] else [])
            lines.append([[float(xy[i][0] * 2), float(xy[i][1] * 2)] for i in o])      # 256 grid -> 512 square
        return lines

    # ------------------------------------------------------------------ suggest
    def analyse(self, data, identify: bool = True) -> Dict[str, Any]:
        """Everything for one photo: the suggestions ("out", as suggest returns them) and the
        parse behind them (D, rep, the 512 square sq, the propagated type distribution u).
        identify=False skips reading the type (the slow part when a diagram closes)."""
        from knot_recognition.cable.neural_parser import parse, parse_best
        from knot_recognition.knot_aware import letterbox, predict
        pair_dirs = self._pair_dirs

        t0 = time.time()
        photo = open_photo(data)
        sq, s, off = letterbox(photo, 512)
        sq = np.asarray(sq)
        u = None
        with self.lock, torch.no_grad():
            pred = predict(self.model, sq)
            if self.can_identify:      # both centre-line sources must agree on the type, as in the evaluations
                D, rep = parse_best(sq, self.model, pred=pred, kappa=self.kappa, clutter_filter=True)
            else:
                D, rep = parse(sq, self.model, pred=pred, kappa=self.kappa, clutter_filter=True)
            H = sq.shape[0]
            w = float(rep.get("width_px", 8.0))
            L = 1.6 * w
            crossings = []
            if D is not None and D.crossings:
                for c in D.crossings:
                    vo, vu, cc = pair_dirs(D, c.cid)
                    pos = np.array([c.pos[0], H - c.pos[1]])
                    crossings.append({"pos": self._back(pos, s, off), "o": self._back(pos + L * np.asarray(vo), s, off),
                                      "u": self._back(pos + L * np.asarray(vu), s, off),
                                      "p_over": float(cc.confidence), "under_guessed": False})
            else:
                for c in rep.get("crossings", []):
                    pos = np.array(c["pos_px"], float)
                    i, j = (np.clip(pos[::-1] / (H / pred["over"].shape[1]), 0, pred["over"].shape[1] - 1)).astype(int)
                    a = 0.5 * math.atan2(float(pred["over"][1, i, j]), float(pred["over"][0, i, j]))
                    vo = np.array([math.cos(a), math.sin(a)])
                    vu = np.array([-vo[1], vo[0]])                 # the strand below is not traced: a guess
                    crossings.append({"pos": self._back(pos, s, off), "o": self._back(pos + L * vo, s, off),
                                      "u": self._back(pos + L * vu, s, off),
                                      "p_over": c.get("p_over_correct"), "under_guessed": True})
            lines, lines_from = [], None
            if D is not None and D.components:
                for comp in D.components:
                    P = np.asarray(comp.points, float)
                    step = max(1, int(len(P) / max(8, comp.length / max(w, 4))))
                    Q = P[::step]
                    if comp.closed:
                        Q = np.vstack([Q, Q[:1]])
                    lines.append([self._back([x, H - y], s, off) for x, y in Q])
                lines_from = "parser"
            elif self.head is not None:
                lines = [[self._back(p, s, off) for p in Lp] for Lp in self._head_lines(sq)]
                lines_from = "strand-order head"
            ends = [self._back(p, s, off) for p in rep.get("free_end_px", [])]
            width = {"px": w / s, "seg": None}
            longest = max(lines, key=len) if lines else None
            if longest and len(longest) >= 3:
                k = len(longest) // 2
                a_, b_ = np.asarray(longest[k - 1]), np.asarray(longest[k + 1])
                t = (b_ - a_) / (np.linalg.norm(b_ - a_) + 1e-9)
                n = np.array([-t[1], t[0]])
                c0 = np.asarray(longest[k])
                width["seg"] = [list(map(float, c0 - 0.5 * (w / s) * n)), list(map(float, c0 + 0.5 * (w / s) * n))]
            reading = {"answer": None, "most_likely": None, "p": None, "abstain_reasons": list(rep.get("abstain_reasons", []))}
            if not self.can_identify:
                reading["abstain_reasons"].append("no reading: knot identification needs SnapPy (pip install snappy)")
            elif identify and D is not None and D.is_closed and not rep["abstain_reasons"]:
                try:
                    from knot_recognition.cable.uncertainty import propagate
                    u = propagate(D, threshold=0.95, max_uncertain=8)
                    reading.update(most_likely=u["most_likely"], p=float(u["most_likely_probability"]),
                                   answer=u["most_likely"] if u["decision"] == "report" else None)
                    if reading["answer"] is None:
                        reading["abstain_reasons"].append("type probability below 0.95")
                except Exception as e:  # noqa
                    reading["abstain_reasons"].append(f"identification failed: {str(e)[:80]}")
        out = {"width": width, "crossings": crossings, "ends": ends, "lines": lines, "lines_from": lines_from,
               "reading": reading, "image_size": [photo.width, photo.height], "seconds": round(time.time() - t0, 2),
               **self.info}
        return {"out": out, "D": D, "rep": rep, "sq": sq, "u": u, "photo": photo}

    def suggest(self, data):
        return self.analyse(data)["out"]


# ---------------------------------------------------------------------- drawing
TOP, UNDER, END, WIDTH = (20, 170, 60), (235, 120, 20), (220, 30, 30), (200, 0, 200)
LINE_COLOURS = [(30, 110, 230), (140, 60, 200), (0, 150, 160), (200, 160, 0), (90, 90, 90)]


def draw_suggestions(photo: Image.Image, out: Dict[str, Any], max_side: int = 1600) -> Image.Image:
    """The suggestions drawn on the photo (scaled so its longer side is at most max_side):
    centre-lines with travel arrows (one colour per strand), each crossing's strand underneath
    (orange; dashed when guessed) with the strand on top (green) drawn over it, the crossing's
    number and p(over), free ends (red rings) and the width segment (magenta)."""
    k = min(1.0, max_side / max(photo.size))
    im = photo.convert("RGB").resize((max(1, round(photo.width * k)), max(1, round(photo.height * k)))) if k < 1 else photo.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(im.size) / 300))
    P = lambda p: (float(p[0]) * k, float(p[1]) * k)  # noqa: E731

    def halo_line(pts, colour, width):
        d.line(pts, fill=(255, 255, 255), width=width + 2 * max(1, lw // 2), joint="curve")
        d.line(pts, fill=colour, width=width, joint="curve")

    def dashed(a, b, colour, width, n=6):
        a, b = np.asarray(a), np.asarray(b)
        for i in range(0, n, 2):
            halo_line([tuple(a + (b - a) * i / n), tuple(a + (b - a) * (i + 1) / n)], colour, width)

    for li, line in enumerate(out.get("lines") or []):
        pts = [P(p) for p in line if p is not None and None not in p]
        if len(pts) < 2:
            continue
        colour = LINE_COLOURS[li % len(LINE_COLOURS)]
        halo_line(pts, colour, lw)
        step = max(3, len(pts) // 6)                    # travel direction every few points
        for i in range(step, len(pts) - 1, step):
            a, b = np.asarray(pts[i - 1]), np.asarray(pts[i])
            t = (b - a) / (np.linalg.norm(b - a) + 1e-9)
            n = np.array([-t[1], t[0]])
            r = 3.5 * lw
            d.polygon([tuple(b + t * r), tuple(b - t * r + n * r * 0.8), tuple(b - t * r - n * r * 0.8)], fill=colour,
                      outline=(255, 255, 255))
    seg = (out.get("width") or {}).get("seg")
    if seg:
        halo_line([P(seg[0]), P(seg[1])], WIDTH, lw)
    for p in out.get("ends") or []:
        c = P(p)
        r = 5 * lw
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=END, width=lw)
    for i, c in enumerate(out.get("crossings") or [], start=1):
        po = np.asarray(P(c["pos"]))
        vo, vu = np.asarray(P(c["o"])) - po, np.asarray(P(c["u"])) - po
        # arms 1.4 x the suggested length, and never shorter than 8 line widths (thin cables)
        vo, vu = [v / (np.linalg.norm(v) + 1e-9) * max(1.4 * np.linalg.norm(v), 8 * lw) for v in (vo, vu)]
        if c.get("under_guessed"):                                             # underneath first, the top strand over it
            dashed(po - vu, po + vu, UNDER, 2 * lw)
        else:
            halo_line([tuple(po - vu), tuple(po + vu)], UNDER, 2 * lw)
        halo_line([tuple(po - vo), tuple(po + vo)], TOP, 2 * lw + 1)
        r = 3 * lw
        d.ellipse([po[0] - r, po[1] - r, po[0] + r, po[1] + r], fill=(255, 255, 255), outline=(0, 0, 0))
        label = f"{i}" + (f"  p={c['p_over']:.2f}" if isinstance(c.get("p_over"), (int, float)) else "")
        size = max(14, max(im.size) // 45)
        # the label goes in the free quadrant between the arms that points most upwards
        a_, b_ = vo / (np.linalg.norm(vo) + 1e-9), vu / (np.linalg.norm(vu) + 1e-9)
        cands = [q / (np.linalg.norm(q) + 1e-9) for q in (a_ + b_, -a_ - b_, a_ - b_, b_ - a_)]
        q = max(cands, key=lambda v: 0.3 * v[0] - v[1])
        tx, ty = po + q * (4 * lw + 0.4 * size)
        anchor = ("l" if q[0] >= 0 else "r") + ("b" if q[1] < 0 else "t")
        try:
            d.text((tx, ty), label, fill=(0, 0, 0), stroke_width=max(2, size // 7), stroke_fill=(255, 255, 255),
                   font_size=size, anchor=anchor)
        except (TypeError, ValueError):          # Pillow < 10.1 (no font_size) or a bitmap font (no anchor)
            d.text((tx + 2 * lw, ty - 4 * lw), label, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
    return im
