"""Human-reviewed cable knot diagram triage demo for Hugging Face Spaces.

This is an experimental diagram parser, not a validated cable-photo classifier.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "runtime" / "knot_recognition"


def _load_diagram_engine():
    # Load the existing vision modules without importing the package's unrelated
    # training/model stack (which otherwise pulls in torch and torchvision).
    pkg = types.ModuleType("knot_recognition")
    pkg.__path__ = [str(SRC)]
    sys.modules.setdefault("knot_recognition", pkg)
    for name in ("utils", "preprocess", "gauss_pd", "traverser"):
        fullname = f"knot_recognition.{name}"
        if fullname in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(fullname, SRC / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
    from knot_recognition.traverser import KnotTraverser
    return KnotTraverser


KnotTraverser = _load_diagram_engine()


def _knotgen():
    # Knotgen is repo-local and uses SymPy for its invariant calculations.
    spec = importlib.util.spec_from_file_location("cable_knotgen", ROOT / "runtime" / "Knotgen" / "knotgen.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _diagram_from_result(pd_crossings):
    """Normalize the traverser's crossing records for Knotgen.from_pd."""
    entries = []
    for _crossing_id, arcs in pd_crossings:
        if len(arcs) != 4 or len(set(arcs)) != 4:
            raise ValueError("Crossing arcs are incomplete or repeated")
        entries.append(tuple(int(v) for v in arcs))
    return entries


@lru_cache(maxsize=1)
def _candidate_index():
    kg = _knotgen()
    census_path = ROOT / "runtime" / "Knotgen" / "census_braids.json"
    census = json.loads(census_path.read_text())
    index = {}
    for name, record in census.items():
        try:
            knot = kg.braid(record["braid"], nstrands=record["nstrands"], name=name)
            signature = (str(knot.alexander()), int(knot.determinant()), len(knot.crossings))
            index.setdefault(signature, []).append(name)
        except Exception:
            continue
    return kg, index


def analyze_views(images, open_ends):
    if not images:
        return None, "Upload at least one clear view of the cable."

    engine = KnotTraverser()
    views = images if isinstance(images, list) else [images]
    reports, overlays, signatures = [], [], []
    try:
        kg, census = _candidate_index()
    except Exception as exc:
        return None, f"Invariant engine could not start: {exc}"

    for i, image in enumerate(views, start=1):
        if image is None:
            continue
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")
        try:
            result = engine.process_image(np.asarray(image))
        except Exception as exc:
            reports.append({
                "view": i,
                "status": "Image processing failed; another view is needed",
                "detail": str(exc),
            })
            continue
        overlays.append(result.annotated_image.to_pil() if result.annotated_image is not None else image)
        ambiguous = [x.crossing_id for x in result.intersections if x.status != "resolved"]
        row = {
            "view": i,
            "crossing_candidates": len(result.intersections),
            "unresolved_crossing_ids": ambiguous,
            "pd_code": result.pd_string,
            "warnings": result.warnings,
        }
        if not result.intersections:
            row["status"] = "No crossing diagram recovered"
        elif ambiguous:
            row["status"] = "Needs another view or manual crossing review"
        elif open_ends:
            row["status"] = "Open ends: invariant classification withheld"
            row["closure"] = "No closure has been applied. Open cable topology depends on the selected closure."
        else:
            try:
                pd = _diagram_from_result(result.pd_code)
                knot = kg.from_pd(pd, name=f"view_{i}")
                sig = (str(knot.alexander()), int(knot.determinant()), len(knot.crossings))
                signatures.append(sig)
                row["status"] = "Resolved diagram; invariant fingerprint computed"
                row["invariants"] = {"alexander_polynomial": sig[0], "determinant": sig[1], "crossings": sig[2]}
                row["census_candidates"] = census.get(sig, [])
            except Exception as exc:
                row["status"] = "Diagram consistency check failed"
                row["detail"] = str(exc)
        reports.append(row)

    if len(views) < 2:
        recommendation = "Capture a second view from a substantially different angle; one photograph can hide crossings and cable segments."
    elif len(signatures) >= 2 and len(set(signatures)) > 1:
        recommendation = "The views disagree. Capture another angle and inspect the marked crossings; no knot label is reported."
    elif any(r["status"] != "Resolved diagram; invariant fingerprint computed" for r in reports):
        recommendation = "At least one view is unresolved. Capture a close-up or a new angle that exposes the uncertain crossing(s)."
    elif signatures:
        labels = next((r.get("census_candidates", []) for r in reports if r.get("invariants")), [])
        recommendation = (
            f"Invariant match candidates: {', '.join(labels)}. These are candidates, not proof of exact identity;"
            " the Alexander polynomial and determinant do not uniquely identify every knot."
            if labels else
            "The diagram has a computed invariant fingerprint, but no matching entry was found in the bundled census."
        )
    else:
        recommendation = "No reliable invariant fingerprint was obtained; capture a clearer view and review the crossing overlay."

    if open_ends:
        recommendation += " Open cable ends are currently left unclosed; specify and record a physical closure before assigning a knot type."
    return overlays, "## Assessment\n" + recommendation + "\n\n```json\n" + json.dumps(reports, indent=2) + "\n```"


with gr.Blocks(title="Cable Knot Photo Triage") as demo:
    gr.Markdown("""# Cable knot photo triage (experimental)

Upload one or more views. The existing image engine proposes crossing locations and over/under states; inspect the overlays. Unclear or inconsistent views are sent back for another photo. A knot name is never treated as certain from the invariant fingerprint alone.

**Capture guidance:** use a plain contrasting background, keep the full tangle in frame, avoid glare, and take a second view after rotating the cable/camera. Indicate whether the cable has open ends. This prototype does not infer hidden 3D geometry from a single photograph and has not been validated on coaxial-cable photographs.
""")
    images = gr.Gallery(label="Cable views", type="pil", columns=2, height="auto")
    open_ends = gr.Checkbox(label="The cable has open ends", value=True)
    run = gr.Button("Analyze views", variant="primary")
    overlays = gr.Gallery(label="Detected crossing overlays", columns=2, height="auto")
    report = gr.Markdown()
    run.click(analyze_views, inputs=[images, open_ends], outputs=[overlays, report])

if __name__ == "__main__":
    demo.launch()
