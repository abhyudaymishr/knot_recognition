"""
consistency.py - Image-level check that no crossing was missed.

Uncertainty propagation covers wrong over/under readings, but not crossings the
detector never found. A diagram with a missed crossing usually looks simpler
(often like an unknot) and can be reported with false confidence.

Euler's formula gives an independent count. The projection of a cable is a
4-valent planar graph: for a connected piece with n crossings, V - E + F = 2 with
E = 2V gives F = n + 2 faces, i.e. the cable mask has n + 1 holes. Summing over
the connected pieces of the mask:

    holes = crossings + pieces          (a crossing-free loop is one piece, one hole)

``euler_check`` segments the cable, counts pieces and holes (ignoring specks
smaller than a cable width squared), and compares the implied crossing count
with the detected one. Open cable ends break the formula (an open arc encloses
no hole), so the check reports "not applicable" when free ends are present.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def cable_mask(image: np.ndarray) -> np.ndarray:
    import cv2

    g = image if image.ndim == 2 else cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cable = the minority class
    fg = th > 0
    if fg.mean() > 0.5:
        fg = ~fg
    return fg


def euler_check(image: np.ndarray, n_detected: int, cable_width_px: Optional[float] = None,
                open_ends: int = 0, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    import cv2

    fg = cable_mask(image) if mask is None else mask
    fg_u8 = fg.astype(np.uint8)
    n_fg, lab_fg, st_fg, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    w = cable_width_px or 8.0
    min_area = max(9.0, (0.6 * w) ** 2)
    pieces = int(sum(1 for i in range(1, n_fg) if st_fg[i, cv2.CC_STAT_AREA] >= 4 * min_area))
    n_bg, lab_bg, st_bg, _ = cv2.connectedComponentsWithStats((~fg).astype(np.uint8), connectivity=4)
    H, W = fg.shape
    border = set(np.unique(np.concatenate([lab_bg[0], lab_bg[-1], lab_bg[:, 0], lab_bg[:, -1]])))
    holes = int(sum(1 for i in range(1, n_bg) if i not in border and st_bg[i, cv2.CC_STAT_AREA] >= min_area))
    out = {"pieces": pieces, "holes": holes, "detected_crossings": int(n_detected)}
    if open_ends:
        out.update(applicable=False, consistent=None,
                   note="free cable ends present: Euler count does not apply")
        return out
    implied = holes - pieces
    out.update(applicable=True, implied_crossings=implied, consistent=implied == n_detected)
    if implied != n_detected:
        out["note"] = (f"the image implies {implied} crossing(s) but {n_detected} were detected; "
                       "abstain and request a clearer view")
    return out
