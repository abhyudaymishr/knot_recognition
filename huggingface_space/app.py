"""Human-reviewed cable knot diagram triage and active NanoTransformer demo for Hugging Face Spaces.

Three tools:
  1. the 250-class Vision NanoTransformer with the topological KnotTraverser (classification and
     similar-knot retrieval);
  2. multi-view cable photo triage with the trained UNetFormer parser (over/under, closure checks,
     the knot type only when every check passes);
  3. the annotation assistant: the same UNetFormer, with its strand-order head, suggests what an
     annotator would mark on a photo (crossings and which strand is on top, free ends, the strands
     in travel order, the cable width) and gives them as a JSON draft to check.
The runtime/ folder is refreshed from the repository with scripts/build_space.py; runtime/MANIFEST.json
lists the source and sha256 of every model file.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
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

# SnapPy (the knot type readings) installs signal handlers when it is first imported, which Python
# allows only in the main thread; Gradio answers in worker threads, so import it here, at start.
os.environ.setdefault("KNOTINFO_PATH", str(ROOT / "runtime" / "knotinfo_full_dataset.json"))
try:
    import snappy  # noqa: F401
    import spherogram  # noqa: F401
except Exception as exc:  # noqa
    print(f"Warning: SnapPy is not available, so no knot types are read: {exc}")

MANIFEST = json.loads((ROOT / "runtime" / "MANIFEST.json").read_text())
UF_MODEL = ROOT / "runtime" / MANIFEST["model"]
UF_HEAD = ROOT / "runtime" / MANIFEST["strand_head"]


def _start_suggester():
    """The trained UNetFormer + strand-order head, loaded once at start (CPU)."""
    try:
        from knot_recognition.annotate import Suggester
        sug = Suggester(UF_MODEL, UF_HEAD, kappa=MANIFEST.get("kappa"))
        print(f"Loaded {sug.info['model']} ({sug.info['model_sha256']}) + strand-order head {sug.info['strand_head']}, "
              f"kappa {sug.kappa}, knot types: {sug.can_identify}")
        return sug, None
    except Exception as exc:  # noqa
        print(f"Warning: the UNetFormer could not be loaded: {exc}")
        return None, f"{type(exc).__name__}: {exc}"


SUGGESTER, SUGGESTER_ERROR = _start_suggester()


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


@lru_cache(maxsize=1)
def _geometric_engine():
    """Cable topology tools for the triage tab (the parser itself is the UNetFormer's: SUGGESTER).

    The traverser's own PD code orders a crossing's arms by angle and does not say
    which strand is on top, so invariants are never computed from it here.
    """
    import knot_recognition.cable.identify as identify
    import knot_recognition.cable.moves as moves
    import knot_recognition.cable.uncertainty as uncertainty
    from knot_recognition.cable.from_traversal import diagram_from_traversal

    return {"identify": identify, "moves": moves, "uncertainty": uncertainty,
            "from_traversal": diagram_from_traversal}


def _to_pil(image):
    if isinstance(image, (tuple, list)):
        image = image[0]
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if not isinstance(image, Image.Image):
        return Image.fromarray(np.asarray(image).astype("uint8"))
    return image


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
# Tab 2: Multi-View Cable Photo Triage & Invariant Checking (trained UNetFormer parser)
# ==============================================================================
def analyze_views(images, open_ends):
    if not images:
        return None, "Upload at least one clear view of the cable."
    if SUGGESTER is None:
        return None, f"The UNetFormer could not be loaded: {SUGGESTER_ERROR}"
    try:
        eng = _geometric_engine()
    except Exception as exc:
        return None, f"Topology engine could not start: {exc}"
    from knot_recognition.annotate import clean, draw_suggestions

    views = images if isinstance(images, list) else [images]
    reports, overlays, answers = [], [], []
    for i, image in enumerate(views, start=1):
        if image is None:
            continue
        row = {"view": i, "parser": f"UNetFormer {SUGGESTER.info['model']} (clutter filter, kappa {SUGGESTER.kappa})"}
        try:
            # letterboxed to 512 inside, never stretched: a stretched photo shears crossing angles
            a = SUGGESTER.analyse(_to_pil(image), identify=not open_ends)
        except Exception as exc:
            row["status"] = "Image processing failed; another view is needed"
            row["detail"] = str(exc)
            reports.append(row)
            continue
        D, rep, u, out = a["D"], a["rep"], a["u"], clean(a["out"])
        reasons = list(rep.get("abstain_reasons", []))
        if open_ends and "free cable ends" not in " ".join(reasons):
            reasons.append("free cable ends: record a closure before naming a type")
        if not SUGGESTER.can_identify:
            reasons.append("no reading: knot identification needs SnapPy")
        row["crossings_detected"] = len(D.crossings) if D is not None else len(out["crossings"])
        row["free_ends_detected"] = len(out["ends"])
        overlay = draw_suggestions(a["photo"], out)
        if D is not None and D.is_closed and not reasons:
            if u is None:
                row["status"] = "Diagram check failed"
                row["detail"] = "; ".join(out["reading"]["abstain_reasons"])
            else:
                try:
                    row["type_distribution"] = u["distribution"][:5]
                    row["most_likely"] = u["most_likely"]
                    row["probability"] = round(u["most_likely_probability"], 3)
                    if u["decision"] == "report":
                        row["status"] = "Type reported"
                        answers.append(u["most_likely"])
                        red = eng["moves"].reduce_diagram(D)
                        row["untangling_steps"] = [s["instruction"] for s in red["steps"]]
                        row["conclusion"] = red["conclusion"]
                        overlay = eng["moves"].overlay_steps(Image.fromarray(a["sq"]), red)
                    else:
                        row["status"] = "Uncertain over/under: another view is needed"
                        row["photograph_crossing_next"] = (u["next_best_views"] or [{}])[0].get("cid")
                except Exception as exc:
                    row["status"] = "Diagram check failed"
                    row["detail"] = str(exc)
        else:
            row["status"] = "Abstained"
            row["reasons"] = reasons
        overlays.append(overlay)
        reports.append(row)

    if len(set(answers)) > 1:
        recommendation = "The views disagree (" + ", ".join(sorted(set(answers))) + "). No type is reported; capture another angle."
    elif answers:
        recommendation = (f"All answering views agree: {answers[0]}. Types come from a geometry-derived diagram; "
                          "the knot name is certified where the report says so.")
    else:
        recommendation = "No view passed every check. Follow the reasons listed per view (another angle, a close-up, or a recorded closure)."
    return overlays, "## Assessment\n" + recommendation + "\n\n```json\n" + json.dumps(reports, indent=2, default=str) + "\n```"


# ==============================================================================
# Tab 3: Annotation assistant (trained UNetFormer + strand-order head)
# ==============================================================================
LEGEND = ("**On the picture:** green = the strand on top at each crossing, orange = the strand underneath "
          "(dashed when the model only guessed it), white dot = crossing centre with its number and p(over), "
          "coloured lines = the strands in travel order (arrows), red rings = free ends, magenta = cable width.")


def suggest_annotations(photo_path):
    if not photo_path:
        return None, "Upload a photo of the cable.", [], None
    if SUGGESTER is None:
        return None, f"The UNetFormer could not be loaded: {SUGGESTER_ERROR}", [], None
    from knot_recognition.annotate import clean, draw_suggestions

    a = SUGGESTER.analyse(Path(photo_path).read_bytes())
    out = clean(a["out"])
    rd = out["reading"]
    if rd["answer"]:
        head = f"### Model reads **`{rd['answer']}`** (p = {rd['p']:.2f})"
    else:
        why = rd["abstain_reasons"][0] if rd["abstain_reasons"] else "no closed diagram"
        head = f"### Model abstains: {why}" + (f" (most likely `{rd['most_likely']}`)" if rd["most_likely"] else "")
    guessed = sum(1 for c in out["crossings"] if c["under_guessed"])
    lines = [
        head,
        f"- **Crossings:** {len(out['crossings'])}" + (f" ({guessed} with the bottom strand guessed: check them)" if guessed else ""),
        f"- **Free ends:** {len(out['ends'])}",
        f"- **Strands:** {len(out['lines'])} centre-line piece(s)" + (f" from the {out['lines_from']}" if out["lines_from"] else ""),
        f"- **Cable width:** {out['width']['px']:.1f} px" if out["width"]["px"] else "- **Cable width:** -",
        f"- **Model:** {out['model']} ({out['model_sha256']}) + strand-order head {out['strand_head']}, "
        f"kappa {out['kappa']}, {out['seconds']} s",
        "",
        LEGEND,
        "",
        "> These are suggestions to check, not labels. When the photos are a benchmark's ground truth, take "
        "which strand is on top from the tie sheet, not from the model, so the labels do not lean towards "
        "what the model already predicts.",
    ]
    if len(rd["abstain_reasons"]) > 1:
        lines += ["", "**Reasons it holds back:** " + "; ".join(rd["abstain_reasons"])]
    rows = [[i, round(c["pos"][0]), round(c["pos"][1]),
             "-" if c["p_over"] is None else f"{c['p_over']:.2f}", "guessed" if c["under_guessed"] else "traced"]
            for i, c in enumerate(out["crossings"], start=1)]
    draft = {"image": Path(photo_path).name, "suggestions": out,
             "note": "Model suggestions in this photo's pixels (x right, y down). Check every item before "
                     "using it as an annotation; record which ones you accepted, flipped or drew by hand."}
    folder = Path(tempfile.mkdtemp(prefix="suggest_"))
    js = folder / f"{Path(photo_path).stem}_suggestions.json"
    js.write_text(json.dumps(draft, indent=1, allow_nan=False))
    return draw_suggestions(a["photo"], out), "\n".join(lines), rows, str(js)


# ==============================================================================
# Gradio Application Interface
# ==============================================================================
EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*.jpg"))

with gr.Blocks(title="Knot Recognition & Active NanoTransformer Triage") as demo:
    gr.Markdown("""# 🪢 Knot Recognition & NanoTransformer Active Classifier

Combines a locally trained **Vision NanoTransformer** spanning all 250 prime knots (`0_unknot` to `10_165`) with the topological **Knot Traverser** for fast active knot classification, topological similarity retrieval, and over/under crossing analysis.
The cable photo tabs use the trained **UNetFormer** parser; the **Annotation assistant** shows what it would mark on a photo (crossings, which strand is on top, free ends, the strands in order) for a person to check.
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
Upload multiple views of a cable tangle. The trained **UNetFormer** proposes crossing locations and over/under states; a type is reported only when the diagram closes, every check passes and the answering views agree.
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

        with gr.TabItem("✏️ Annotation Assistant"):
            gr.Markdown("""
Upload one cable photo. The trained **UNetFormer** and its **strand-order head** suggest what an annotator would mark: each crossing with the strand on top and the strand underneath, the free ends, the strands in travel order (including the jump across each crossing) and the cable width. Download the suggestions as a JSON draft and check every item before using it. Photos are processed for this request only and are not stored by the app.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    photo_in = gr.Image(label="Cable photo", type="filepath")
                    suggest_btn = gr.Button("✏️ Suggest annotations", variant="primary")
                    if EXAMPLES:
                        gr.Examples(examples=[[p] for p in EXAMPLES], inputs=[photo_in], label="Synthetic benchmark photos")
                with gr.Column(scale=1):
                    suggest_img = gr.Image(label="Suggestions", type="pil")
                    suggest_md = gr.Markdown()
            suggest_df = gr.Dataframe(
                headers=["#", "x (px)", "y (px)", "p(over)", "Bottom strand"],
                datatype=["number", "number", "number", "str", "str"],
                label="Suggested crossings",
            )
            suggest_file = gr.File(label="Suggestions (JSON draft)")
            suggest_btn.click(
                suggest_annotations,
                inputs=[photo_in],
                outputs=[suggest_img, suggest_md, suggest_df, suggest_file],
            )

if __name__ == "__main__":
    demo.launch()
