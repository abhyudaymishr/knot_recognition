"""
moves.py - Turn diagram simplification into untangling instructions.

``logged_reduction`` applies Reidemeister I and II moves one at a time, in a
deterministic order, and records each one against the crossings of the original
diagram (and so against pixel positions in the photo):

  R1  "undo the twist loop at crossing #a"                    (removes 1 crossing)
  R2  "slide the top strand off at crossings #a and #b"      (removes 2 crossings)

When no R1/R2 move remains, any crossings left either need moves that rearrange
crossings (R3 and larger moves; no step list is produced) or are a real knot.
``overlay_steps`` draws the numbered steps on the photograph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import snappy
from spherogram.links.simplify import reidemeister_I_and_II


def logged_reduction(pd: Sequence[Sequence[int]], row_cids: Optional[Sequence[int]] = None,
                     row_pos: Optional[Sequence[Tuple[float, float]]] = None,
                     over_strand_hint: Optional[Dict[int, Any]] = None) -> Dict[str, Any]:
    """Apply R1/R2 moves one by one and log them.

    ``row_cids[i]`` and ``row_pos[i]`` give the crossing id and position for PD row i
    (``Diagram.crossing_index()`` and the crossing positions).
    """
    L = snappy.Link([tuple(c) for c in pd]) if pd else snappy.Link([])
    cid = {C: (row_cids[C.label] if row_cids is not None else C.label + 1) for C in L.crossings}
    pos = {C: (row_pos[C.label] if row_pos is not None else None) for C in L.crossings}
    steps: List[Dict[str, Any]] = []
    while True:
        progress = False
        for C in sorted(L.crossings, key=lambda c: cid[c]):
            if C not in L.crossings:
                continue
            elim, _changed = reidemeister_I_and_II(L, C)
            if not elim:
                continue
            ids = sorted(cid[x] for x in elim)
            where = [pos[x] for x in sorted(elim, key=lambda x: cid[x])]
            if len(elim) == 1:
                steps.append({"step": len(steps) + 1, "move": "R1", "crossings": ids, "positions": where,
                              "instruction": f"Undo the twist loop at crossing #{ids[0]}."})
            else:
                steps.append({"step": len(steps) + 1, "move": "R2", "crossings": ids, "positions": where,
                              "instruction": f"Slide the strand lying on top at crossings #{ids[0]} and #{ids[1]} "
                                             "off the other strand (they cross twice and come apart)."})
            progress = True
            break
        if not progress:
            break
    remaining = len(L.crossings)
    L2 = L.copy()
    if remaining:
        L2.simplify("global")
    summary = {
        "steps": steps,
        "crossings_before": len(pd),
        "crossings_after_steps": remaining,
        "crossings_after_global_simplification": len(L2.crossings),
        "remaining_crossing_ids": sorted(cid[C] for C in L.crossings),
    }
    if remaining == 0:
        summary["conclusion"] = "The steps remove every crossing: the cable comes apart (no knot)."
        summary["knotted"] = False
    else:
        summary.update(_verdict(L2, remaining))
    summary["reduced_pd"] = [tuple(c) for c in L.PD_code()] if remaining else []
    return summary


def _verdict(L2, remaining: int) -> Dict[str, Any]:
    """Certified knotted / unknotted verdict for what the R1/R2 steps could not remove."""
    if len(L2.crossings) == 0:
        return {"knotted": False, "conclusion": (
            f"{remaining} crossing(s) remain that need re-routing moves (Reidemeister III or larger) "
            "before they can be undone; no knot is present.")}
    if len(L2.link_components) == 1:
        g = L2.knot_floer_homology()["seifert_genus"]  # knot Floer homology detects the unknot
        if g == 0:
            return {"knotted": False, "conclusion": (
                f"No knot is present (knot Floer genus 0), but the {remaining} remaining crossing(s) need "
                "re-routing moves that this step list does not cover.")}
        return {"knotted": True, "conclusion": (
            f"This is a real knot (Seifert genus {g}, certified by knot Floer homology). No sequence of moves "
            "removes it with both ends fixed: unplug one end and pass it through.")}
    from .identify import identify_link
    r = identify_link(L2)
    linked = r["split"] in ("interlinked", "partly separable")
    return {"knotted": linked if r["split_certainty"] == "certified" else None,
            "conclusion": f"Several cables remain: {r['split']} ({r['split_method']})."}


def reduce_diagram(diagram) -> Dict[str, Any]:
    """``logged_reduction`` on a closed ``Diagram``, with pixel positions for the overlay."""
    pd = diagram.to_pd()
    return logged_reduction(pd, diagram.crossing_index(),
                            [diagram.to_pixel(c.pos) for c in diagram.crossings])


def overlay_steps(image, reduction: Dict[str, Any], radius: int = 14):
    """Draw numbered untangling steps on a PIL image. Returns a new image."""
    from PIL import Image, ImageDraw

    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    colors = {"R1": (230, 120, 0), "R2": (0, 140, 220)}
    for st in reduction["steps"]:
        col = colors[st["move"]]
        pts = [p for p in st["positions"] if p is not None]
        for x, y in pts:
            d.ellipse([x - radius, y - radius, x + radius, y + radius], outline=col, width=3)
        if len(pts) == 2:
            d.line([pts[0], pts[1]], fill=col, width=2)
        if pts:
            x = sum(p[0] for p in pts) / len(pts)
            y = sum(p[1] for p in pts) / len(pts)
            d.rectangle([x + radius - 2, y - radius - 16, x + radius + 22, y - radius + 2], fill=col)
            d.text((x + radius + 3, y - radius - 15), f"{st['step']}", fill=(255, 255, 255))
    return img
