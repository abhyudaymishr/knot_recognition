"""
annotator_server.py - The photo annotator, with the trained UNetFormer suggesting the annotations.

Serves tools/photo_annotator.html at http://localhost:8765 and answers POST /api/suggest with the
model's reading of one photo, in that photo's own pixels:
  cable width        a two-click segment across the cable and its length
  crossings          centre, a point on the strand on top, a point on the strand underneath, and
                     the probability that this over/under reading is right
  free ends          strand ends that do not reach a crossing
  centre-lines       each strand in travel direction: the parser's traced strands, or, when the
                     parser cannot trace them, the strand-order head's ordered points (links
                     re-checked with strand_pointer.recheck)
  reading            the system's answer (type, probability) or the reasons it abstains
The page shows these as suggestions. Nothing is saved until it is accepted, and every saved item
records whether it came from the model, was corrected, or was drawn by hand.

The photo is letterboxed to 512 x 512 as in scripts/eval_real_photos.py; the parser runs with the
settings of the matched comparison (clutter filter on, kappa from the model's evaluation).
Everything runs on this computer; no photo leaves it. Opening tools/photo_annotator.html as a file
still works, without suggestions.

  python scripts/annotator_server.py                      # UNetFormer seed 0 + strand-order head
  python scripts/annotator_server.py --seed 2 --port 8800
  python scripts/annotator_server.py --model checkpoints/uf_pointer/mps_ufptr_s0.pth --head ""
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import torch

from knot_recognition.annotate import Suggester as _Suggester, clean, sha12  # noqa: F401  (re-exported)

PAGE = ROOT / "tools" / "photo_annotator.html"


def kappa_for(model_path):
    """kappa from the model's evaluation (results/suite/bu_<stem>/meta.json), if there is one."""
    meta = ROOT / "results" / "suite" / f"bu_{Path(model_path).stem.replace('mps_', '')}" / "meta.json"
    return json.loads(meta.read_text()).get("kappa") if meta.exists() else None


class Suggester(_Suggester):
    """knot_recognition.annotate.Suggester with kappa looked up from the model's evaluation."""

    def __init__(self, model_path, head_path=None, kappa=None):
        super().__init__(model_path, head_path, kappa or kappa_for(model_path))


def make_handler(sug: Suggester):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html", "/photo_annotator.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._send(200, json.dumps(sug.info).encode())
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            if self.path != "/api/suggest":
                return self._send(404, b'{"error": "not found"}')
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > 80 * 2 ** 20:
                return self._send(400, b'{"error": "send one photo (at most 80 MB) as the request body"}')
            try:
                out = sug.suggest(self.rfile.read(n))
                self._send(200, json.dumps(clean(out), allow_nan=False).encode())
            except Exception as e:  # noqa
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())

        def log_message(self, fmt, *args):
            if "/api/suggest" in (args[0] if args else ""):
                sys.stderr.write("  suggested " + self.headers.get("X-Name", "?") + "\n")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="which trained UNetFormer (checkpoints/beat_unet/mps_uf_s<seed>.pth)")
    ap.add_argument("--model", default=None, help="any knot-aware checkpoint instead")
    ap.add_argument("--head", default=None, help="strand-order head ('' = none; default: the one trained on this UNetFormer)")
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    model = Path(a.model) if a.model else ROOT / "checkpoints" / "beat_unet" / f"mps_uf_s{a.seed}.pth"
    head = a.head if a.head is not None else str(ROOT / "checkpoints" / "strand_pointer" / f"mps_ptr_s{a.seed}.pth")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    sug = Suggester(model, head or None, a.kappa)
    # one request at a time, on the main thread: SnapPy (the reading of the type) installs signal
    # handlers when it is first imported, which Python allows only there
    srv = HTTPServer(("127.0.0.1", a.port), make_handler(sug))
    url = f"http://localhost:{a.port}/"
    print(f"annotator with {sug.info['model']} (kappa {sug.kappa}, strand head: {sug.info['strand_head']}) at {url}")
    print("Ctrl-C to stop. Photos are processed on this computer only.", flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
