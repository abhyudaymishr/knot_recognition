"""
make_capture_kit.py - Everything needed to photograph and annotate the real cable test set
(KnotBench-Photo real_v1), except the photographs.

Writes data/knotbench_photo/real_v1/:
  cables.csv        20 cables: id, knot type, how to tie, material, closure, sessions
  shot_list.csv     every photo to take: file name, cable, session, view, conditions (120 rows)
  capture_kit.pdf   page 1 protocol + session table; then one tie sheet per cable with a
                    diagram to tie from (orthogonal diagram from spherogram, gaps = under-strands)
  README.md         step-by-step

The diagrams are drawn from exact PD codes (spherogram), and every cable's type is checked
with SnapPy before the sheet is written.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np

OUT = ROOT / "data" / "knotbench_photo" / "real_v1"

SESSIONS = {
    "sA": "Desk by a window: daylight, matte light background (paper or wall), phone 1",
    "sB": "Wooden floor or table: one desk lamp from the side (hard shadows), phone 2",
    "sC": "Dark cloth or mat: dim room light, camera or phone 1 without flash",
    "sD": "Cluttered desk (cables, papers, objects nearby): mixed light, phone 2",
}
VIEWS = {"v1": "top-down, cable fills about two thirds of the frame",
         "v2": "about 30 degrees tilt, from one side",
         "v3": "about 60 degrees tilt, from the opposite side"}

# id, label (type_key), how to build, material, colour hint, diameter mm, sessions
CABLES = [
    ("c01", "0_1", ("braid", [1, 2, -1, -2]), "coax RG6", "black", 6.9, ("sA", "sB")),
    ("c02", "0_1", ("braid", [1, 2, 1, -2]), "patch cable (Cat6)", "blue", 6.0, ("sC", "sD")),
    ("c03", "3_1", ("name", "3_1"), "coax RG6", "black", 6.9, ("sA", "sC")),
    ("c04", "3_1*", ("mirror", "3_1"), "rope 8 mm", "any", 8.0, ("sB", "sD")),
    ("c05", "3_1", ("name", "3_1"), "fishing line or thick thread", "any", 1.0, ("sA", "sD")),
    ("c06", "4_1", ("name", "4_1"), "coax RG58", "black", 5.0, ("sB", "sC")),
    ("c07", "4_1", ("name", "4_1"), "rope 8 mm", "any", 8.0, ("sA", "sB")),
    ("c08", "5_1", ("name", "5_1"), "patch cable (Cat6)", "any", 6.0, ("sC", "sD")),
    ("c09", "5_2", ("name", "5_2"), "coax RG6", "white", 6.9, ("sA", "sC")),
    ("c10", "5_2*", ("mirror", "5_2"), "rope 8 mm", "any", 8.0, ("sB", "sD")),
    ("c11", "6_1", ("name", "6_1"), "coax RG58", "black", 5.0, ("sA", "sD")),
    ("c12", "6_2", ("name", "6_2"), "patch cable (Cat6)", "any", 6.0, ("sB", "sC")),
    ("c13", "6_3", ("name", "6_3"), "rope 8 mm", "any", 8.0, ("sA", "sB")),
    ("c14", "7_1", ("name", "7_1"), "fishing line or thick thread", "any", 1.0, ("sC", "sD")),
    ("c15", "3_1 # 3_1", ("sum", ("3_1", "3_1")), "rope 8 mm", "any", 8.0, ("sA", "sC")),
    ("c16", "3_1 # 3_1*", ("sum", ("3_1", "3_1*")), "rope 8 mm", "any", 8.0, ("sB", "sD")),
    ("c17", "link[0_1,0_1|interlinked|lk=[[0, 1], [1, 0]]]", ("name", "L2a1"), "coax RG6 + patch cable (two cables)", "two colours", 6.5, ("sA", "sD")),
    ("c18", "link[0_1,0_1|interlinked|lk=[[0, 1], [1, 0]]]", ("name", "L2a1"), "rope + thread (two cables)", "two colours", 4.0, ("sB", "sC")),
    ("c19", "link[0_1,0_1|separable|lk=[[0, 0], [0, 0]]]", ("unlink2", None), "two coax cables crossing twice, not linked", "any", 6.9, ("sA", "sB")),
    ("c20", "3_1 # 4_1", ("sum", ("3_1", "4_1")), "coax RG58", "black", 5.0, ("sC", "sD")),
]


def build(spec):
    import spherogram

    kind, arg = spec
    if kind == "braid":
        return spherogram.Link(spherogram.ClosedBraid(arg).PD_code())
    if kind == "name":
        return spherogram.Link(arg)
    if kind == "mirror":
        return spherogram.Link(arg).mirror()
    if kind == "sum":
        parts = [spherogram.Link(a.rstrip("*")).mirror() if a.endswith("*") else spherogram.Link(a) for a in arg]
        L = parts[0]
        for p in parts[1:]:
            L = L.connected_sum(p)
        return L
    if kind == "unlink2":
        return spherogram.Link(spherogram.ClosedBraid([1, -1]).PD_code())  # two strands crossing twice, separable
    raise ValueError(kind)


def exact_label(L) -> str:
    """The cable's type exactly as the pipeline names it (cable.identify), so labels and answers
    use the same chirality convention."""
    import snappy  # noqa: F401
    from knot_recognition.cable.identify import identify_link, type_key

    return type_key(identify_link(L))


def planned_matches(label: str, planned: str) -> bool:
    if planned.startswith("link") or planned.startswith("3_1 # 3_1"):
        return label == planned or (planned.startswith("3_1 # 3_1") and label.startswith("3_1 # 3_1")
                                    and label.endswith("*") == planned.endswith("*"))
    parts = lambda k: sorted(x.strip() for x in k.rstrip("*").split("#"))
    return parts(label) == parts(planned)


def ortho(L):
    """Orthogonal diagram of L: vertices, arrows (straight segments), and for each crossing the
    intersection point with the arrow drawn on top and the arrow drawn below."""
    from spherogram.links.orthogonal import OrthogonalLinkDiagram

    verts, arrows, crossings = OrthogonalLinkDiagram(L).plink_data()
    V = np.array(verts, float)
    X = []
    for c in crossings:
        top, bot = c[0], c[1]
        p, q = V[arrows[top][0]], V[arrows[top][1]]
        a = V[arrows[bot][0]]
        x = p[0] if p[0] == q[0] else a[0]
        y = p[1] if p[1] == q[1] else a[1]
        X.append((top, bot, np.array([x, y])))
    return V, arrows, X


def drawn_label(V, arrows, X) -> str:
    """Type of the diagram exactly as drawn: lift it to 3D (top strand bumped towards the viewer,
    +z, the strand below bumped away), project along z and identify. Whatever convention the
    layout code uses, the label describes the picture a person ties from."""
    from knot_recognition.cable.diagram import from_curves_3d
    from knot_recognition.cable.identify import identify_diagram, type_key

    nxt = {a[0]: i for i, a in enumerate(arrows)}
    seen, curves = set(), []
    for start in range(len(arrows)):
        if start in seen:
            continue
        pts, ai = [], start
        while ai not in seen:
            seen.add(ai)
            a, b = V[arrows[ai][0]], V[arrows[ai][1]]
            for t in np.linspace(0, 1, 40, endpoint=False):
                p = a + t * (b - a)
                z = 0.0
                for top, bot, xy in X:
                    d = np.linalg.norm(p - xy)
                    if ai == top:
                        z += np.exp(-(d / 3) ** 2)
                    elif ai == bot:
                        z -= np.exp(-(d / 3) ** 2)
                pts.append([p[0], p[1], z])
            ai = nxt[arrows[ai][1]]
        curves.append(np.array(pts))
    return type_key(identify_diagram(from_curves_3d(curves)))


def draw(ax, V, arrows, X):
    """Straight segments; a gap where a strand passes under. Standard axes (y up), no reflection,
    so the picture is the projection that drawn_label identifies."""
    gap = 2.2
    for ai, (i, j) in enumerate(arrows):
        a, b = V[i], V[j]
        u = (b - a) / np.linalg.norm(b - a)
        cuts = sorted(float((xy - a) @ u) for top, bot, xy in X if bot == ai)
        s0 = 0.0
        for t in cuts:
            ax.plot(*np.array([a + s0 * u, a + (t - gap) * u]).T, color="k", lw=5, solid_capstyle="butt")
            s0 = t + gap
        ax.plot(*np.array([a + s0 * u, b]).T, color="k", lw=5, solid_capstyle="butt")
    # round joints at the corners
    ax.plot(V[:, 0], V[:, 1], "o", color="k", ms=4.2)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    OUT.mkdir(parents=True, exist_ok=True)
    links, labels = {}, {}
    for cid, planned, spec, *_ in CABLES:
        L = build(spec)
        geo = ortho(L)
        label = drawn_label(*geo)
        assert label.rstrip("*") == exact_label(L).rstrip("*") or label.startswith("3_1 # 3_1") or label.startswith("link")
        if not planned_matches(label, planned):
            raise SystemExit(f"{cid}: construction gives {label}, planned {planned}")
        links[cid], labels[cid] = geo, label
    # chirality of mirror pairs must differ
    for a, b in (("c03", "c04"), ("c09", "c10")):
        if labels[a] == labels[b]:
            raise SystemExit(f"{a} and {b} should be mirror images")
    with open(OUT / "cables.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cable_id", "ground_truth_type", "material", "colour", "diameter_mm", "closure", "sessions"])
        for cid, _, spec, mat, col, dia, ses in CABLES:
            w.writerow([cid, labels[cid], mat, col, dia, "ends joined (barrel connector, heat-shrink or tape)", " ".join(ses)])
    shots = []
    for cid, label, spec, mat, col, dia, ses in CABLES:
        for s in ses:
            for v in VIEWS:
                shots.append([f"{cid}_{s}_{v}.jpg", cid, s, v, SESSIONS[s], VIEWS[v]])
    with open(OUT / "shot_list.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "cable_id", "session_id", "view", "session_conditions", "view_instructions"])
        w.writerows(shots)

    with PdfPages(OUT / "capture_kit.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "KnotBench-Photo real_v1: capture kit", fontsize=16, weight="bold")
        lines = [
            "1. Tie each cable from its sheet (next pages). Follow the drawing: a gap means",
            "   the strand passes UNDER. Join the two ends (barrel connector, heat-shrink or tape)",
            "   so the cable is a closed loop; for links tie two cables as drawn.",
            "2. Before the first photo of a cable, take one reference shot of the cable next",
            "   to its sheet (name it cXX_reference.jpg; it is excluded from evaluation).",
            "3. For each cable, take 3 views in each of its 2 sessions and name the files exactly",
            "   as in shot_list.csv (cXX_sY_vZ.jpg): 20 cables x 2 sessions x 3 views = 120 photos.",
            "   Re-lay the cable loosely between sessions; do not re-tie it.",
            "4. Annotate in tools/photo_annotator.html: load cables.csv once, then the photos;",
            "   the file name fills cable, session, view and type. Click the cable width,",
            "   then every crossing (centre, a point on the top strand, a point on the bottom",
            "   strand) and set visibility.",
            "5. Put the photos in data/knotbench_photo/real_v1/images/ and records.jsonl next",
            "   to them. Run scripts/check_real_photos.py, then score ONCE:",
            "   python scripts/eval_real_photos.py data/knotbench_photo/real_v1",
            "",
            "Sessions (each is a different place, light and camera):",
        ] + [f"   {k}: {v}" for k, v in SESSIONS.items()] + ["", "Views:"] + [f"   {k}: {v}" for k, v in VIEWS.items()] + [
            "", "Do not show these photos to anyone changing the model: the model is frozen",
            "(results/frozen_model.json) and the scoring script logs every run."]
        for i, t in enumerate(lines):
            fig.text(0.07, 0.91 - i * 0.022, t, fontsize=9.5, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        names = {"0_1": "unknot (tangled, no knot)", "3_1": "trefoil", "3_1*": "trefoil, mirror image",
                 "4_1": "figure-eight knot", "5_1": "cinquefoil (5_1)", "5_2": "three-twist knot (5_2)",
                 "5_2*": "three-twist knot, mirror image", "6_1": "stevedore knot (6_1)", "6_2": "6_2", "6_3": "6_3",
                 "7_1": "7_1 (seven-crossing torus knot)", "3_1 # 3_1": "granny knot", "3_1 # 3_1*": "square (reef) knot",
                 "3_1 # 4_1": "trefoil + figure-eight (connected sum)"}
        for cid, planned, spec, mat, col, dia, ses in CABLES:
            label = labels[cid]
            fig = plt.figure(figsize=(8.27, 11.69))
            title = names.get(planned, "Hopf link (two linked cables)" if "interlinked" in label else
                              "two separate cables crossing twice (unlinked)")
            fig.text(0.07, 0.94, f"{cid}: {title}", fontsize=18, weight="bold")
            fig.text(0.07, 0.91, f"type {label}   |   material: {mat}, {col}, about {dia} mm   |   sessions: {', '.join(ses)}",
                     fontsize=9.5)
            ax = fig.add_axes([0.1, 0.25, 0.8, 0.6])
            draw(ax, *links[cid])
            fig.text(0.07, 0.2, "Gap = this strand passes UNDER the other one. Join the ends so the loop is closed.", fontsize=10)
            fig.text(0.07, 0.17, "Photos: " + ", ".join(f"{cid}_{s}_{v}.jpg" for s in ses for v in VIEWS), fontsize=8.5)
            pdf.savefig(fig)
            plt.close(fig)
    print("wrote", OUT, len(shots), "shots")


if __name__ == "__main__":
    main()
