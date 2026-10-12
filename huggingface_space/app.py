"""Human-reviewed cable knot diagram triage and active NanoTransformer demo for Hugging Face Spaces.

Combines the locally trained 250-class Vision NanoTransformer for instant active
knot classification and similarity retrieval with the topological KnotTraverser.
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

try:
    import spaces
except ImportError:  # The local GitHub CI runner is CPU-only.
    class _SpacesCompat:
        @staticmethod
        def GPU(fn):
            return fn
    spaces = _SpacesCompat()

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "runtime" / "knot_recognition"


def _load_diagram_engine():
    # Load the runtime vision & model modules
    pkg = types.ModuleType("knot_recognition")
    pkg.__path__ = [str(SRC)]
    sys.modules.setdefault("knot_recognition", pkg)
    for name in ("utils", "preprocess", "gauss_pd", "nanotransformer", "traverser"):
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


@lru_cache(maxsize=1)
def _load_nanotransformer_model():
    ckpt_path = ROOT / "runtime" / "nanotransformer_knot250.pth"
    if not ckpt_path.exists():
        print(f"Warning: Checkpoint not found at {ckpt_path}")
        return None
    try:
        from knot_recognition.nanotransformer import load_nanotransformer
        model = load_nanotransformer(ckpt_path, device="cpu")
        print(f"Loaded NanoTransformer with {len(model.classes)} classes.")
        return model
    except Exception as exc:
        print(f"Warning: failed to load NanoTransformer: {exc}")
        return None


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
    index_path = ROOT / "runtime" / "Knotgen" / "census_invariant_index.json"
    rows = json.loads(index_path.read_text())
    index = {
        (row["alexander_polynomial"], int(row["determinant"]), int(row["crossings"])): row["labels"]
        for row in rows
    }
    return kg, index


# ==============================================================================
# Tab 1: NanoTransformer Active Classification & Similar Knot Retrieval
# ==============================================================================
@spaces.GPU
def classify_and_retrieve_similar(image, top_k=5):
    if image is None:
        return None, "Please upload a clear image of a knot or cable tangle.", [], []

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")

    model = _load_nanotransformer_model()
    engine = KnotTraverser(nanotransformer=model)
    fetch_k = max(15, int(top_k))
    res = engine.traverse_and_classify(image, top_k=fetch_k)

    overlay = res.annotated_image.to_pil() if res.annotated_image is not None else image

    pred_info = res.classification or {}
    top_label = pred_info.get("top_label", "unknown")
    top_prob = pred_info.get("top_probability", 0.0)

    # Build Assessment Markdown
    confidence_badge = "High" if top_prob > 0.30 else ("Moderate" if top_prob > 0.10 else "Low / Uncertain")
    report_lines = [
        f"### Predicted Knot Topology: **`{top_label}`** ({confidence_badge} Confidence)",
        f"- **Model Probability:** `{top_prob*100:.2f}%` (Uniform chance over 250 classes is `~0.40%`)",
        f"- **Traverser Detected Crossings:** `{len([x for x in res.intersections if x.status == 'resolved'])}` (Total junction candidates: `{len(res.intersections)}`)",
        f"- **Extracted Planar Diagram (PD) Code:** `{res.pd_string or 'PD[]'}`",
        f"- **Extracted Gauss Code:** `{res.gauss_string or '(none)'}`",
    ]

    # Confidence & Photo/Texture Alert
    if top_prob < 0.05:
        report_lines.append(
            "\n> ⚠️ **Low Model Confidence Alert:** The top prediction has near-uniform probability "
            f"({top_prob*100:.2f}% vs random 0.40%). This typically indicates the image is a 3D photograph, "
            "relief carving, or has stone/background textures rather than a high-contrast line diagram. "
            "Check the full candidate rankings below."
        )

    # Topological Relationship: 4_1 (Figure-8) and 8_18 (Double Figure-8)
    all_cand_labels = [p.get("label") for p in pred_info.get("predictions", [])]
    if "8_18" in all_cand_labels and "4_1" in all_cand_labels:
        report_lines.append(
            "\n> 💡 **Topological Family Note:** Knot **`8_18`** is the exact **2-periodic double of the Figure-Eight knot (`4_1`)**. "
            "Their braid representations are `[1, -2, 1, -2, 1, -2, 1, -2]` (8_18) and `[1, -2, 1, -2]` (4_1), "
            "sharing the identical 4-fold alternating symmetry. Relief carvings with grooved, braided, or dual-strand textures "
            "frequently activate features corresponding to `8_18`."
        )

    if res.consensus:
        report_lines.append("\n#### Model & Traverser Cross-Examination")
        for note in res.consensus.get("notes", []):
            report_lines.append(f"- {note}")

    if len(res.intersections) > 25 and len([x for x in res.intersections if x.status == "resolved"]) == 0:
        report_lines.append(
            f"\n> 📸 **Photo Texture Clutter Warning:** The traverser detected "
            f"`{len(res.intersections)}` spurious junctions across background cracks, moss, or relief textures with 0 resolved crossings. "
            "The raster traverser is designed for 2D line diagrams. For relief carvings or photos, crop closely around the central medallion or adjust contrast."
        )

    if res.warnings:
        report_lines.append("\n#### Parsing Warnings")
        for w in res.warnings:
            report_lines.append(f"- ⚠️ {w}")

    # Top-K Predictions List of Lists for Dataframe
    preds_table = []
    for p in pred_info.get("predictions", [])[:int(top_k)]:
        preds_table.append([
            p.get("rank"),
            p.get("label"),
            f"{p.get('probability', 0.0)*100:.2f}%",
            p.get("crossings", "-"),
            p.get("alexander_polynomial", "") or "-",
            p.get("determinant", "-"),
        ])

    # Similar Knots List of Lists for Dataframe
    similar_table = []
    for s in res.similar_knots:
        similar_table.append([
            s.get("rank"),
            s.get("label"),
            f"{s.get('similarity', 0.0):.4f}",
            s.get("crossings", "-"),
            s.get("alexander_polynomial", "") or "-",
            s.get("determinant", "-"),
        ])

    return overlay, "\n".join(report_lines), preds_table, similar_table


# ==============================================================================
# Tab 2: Multi-View Cable Photo Triage & Invariant Checking
# ==============================================================================
@spaces.GPU
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


# ==============================================================================
# Gradio Application Interface
# ==============================================================================
with gr.Blocks(title="Knot Recognition & Active NanoTransformer Triage") as demo:
    gr.Markdown("""# 🪢 Knot Recognition & NanoTransformer Active Classifier

Combines a locally trained **Vision NanoTransformer** spanning all 250 prime knots (`0_unknot` to `10_165`) with the topological **Knot Traverser** for fast active knot classification, topological similarity retrieval, and over/under crossing analysis.
""")

    with gr.Tabs():
        with gr.TabItem("⚡ Active Classification & Similar Knot Retrieval"):
            gr.Markdown("""
Upload an image of a knot diagram or cable tangle. The **NanoTransformer** predicts candidate knot types, fetches nearest topological neighbors in its 128-d embedding space, and cross-validates against physical crossings extracted by the **Knot Traverser**.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(label="Knot Image", type="pil")
                    top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K Candidates")
                    classify_btn = gr.Button("⚡ Classify & Retrieve Knots", variant="primary")
                with gr.Column(scale=1):
                    annotated_out = gr.Image(label="Traverser Annotated Crossing Overlay")
                    summary_md = gr.Markdown(label="Summary Report")

            gr.Markdown("### 📊 Top Model Predictions")
            preds_df = gr.Dataframe(
                headers=["Rank", "Knot Label", "Probability", "Crossings", "Alexander Polynomial", "Determinant"],
                datatype=["number", "str", "str", "number", "str", "number"],
                label="Predictions across 250 Prime Knots",
            )

            gr.Markdown("### 🔍 Top Structurally Similar Knots (128-d Embedding Space)")
            similar_df = gr.Dataframe(
                headers=["Rank", "Similar Knot", "Cosine Similarity", "Crossings", "Alexander Polynomial", "Determinant"],
                datatype=["number", "str", "str", "number", "str", "number"],
                label="Nearest Census Neighbors",
            )

            classify_btn.click(
                classify_and_retrieve_similar,
                inputs=[input_img, top_k_slider],
                outputs=[annotated_out, summary_md, preds_df, similar_df],
            )

        with gr.TabItem("📸 Multi-View Cable Photo Triage"):
            gr.Markdown("""
Upload multiple views of a cable tangle. Proposes crossing locations and over/under states, requiring multi-angle agreement before computing topological invariants.
""")
            gallery_in = gr.Gallery(label="Cable Views", type="pil", columns=2, height="auto")
            open_ends_chk = gr.Checkbox(label="The cable has open ends", value=True)
            triage_btn = gr.Button("Analyze Cable Views", variant="primary")
            gallery_out = gr.Gallery(label="Detected Crossing Overlays", columns=2, height="auto")
            triage_report = gr.Markdown()

            triage_btn.click(
                analyze_views,
                inputs=[gallery_in, open_ends_chk],
                outputs=[gallery_out, triage_report],
            )

if __name__ == "__main__":
    demo.launch()
