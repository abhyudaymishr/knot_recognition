"""
topology_loss.py - A centre-line topology loss (soft clDice, Shit et al., CVPR 2021).

The parser fails mostly on topology, not on pixels. A centre-line that breaks for one cell
leaves "free cable ends", and the image is abstained on. A per-pixel loss (BCE, Dice, the
Lebesgue layer-cake term) charges a one-cell break almost nothing. clDice compares
skeletons:
  T_prec = |S(P) * G| / |S(P)|   (predicted skeleton inside the true structure)
  T_sens = |S(G) * P| / |S(G)|   (true skeleton covered by the prediction: a break lowers it)
  clDice = 2 T_prec T_sens / (T_prec + T_sens),  loss = 1 - clDice
The soft skeleton S is computed with iterated min/max pooling (soft erosion and opening), so
it is differentiable. It is applied to the cable mask and to both centre-line ridges.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _erode(x):
    p1 = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.min(p1, p2)


def _dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def _open(x):
    return _dilate(_erode(x))


def soft_skeleton(x: torch.Tensor, iters: int) -> torch.Tensor:
    """x: (B, 1, H, W) in [0, 1]."""
    skel = F.relu(x - _open(x))
    for _ in range(iters):
        x = _erode(x)
        delta = F.relu(x - _open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(prob: torch.Tensor, target: torch.Tensor, iters: int, smooth: float = 1.0) -> torch.Tensor:
    """prob, target: (B, H, W); target is binarised at 0.5."""
    p = prob.unsqueeze(1)
    g = (target > 0.5).float().unsqueeze(1)
    sp, sg = soft_skeleton(p, iters), soft_skeleton(g, iters)
    tprec = ((sp * g).sum((1, 2, 3)) + smooth) / (sp.sum((1, 2, 3)) + smooth)
    tsens = ((sg * p).sum((1, 2, 3)) + smooth) / (sg.sum((1, 2, 3)) + smooth)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


# erosion steps per map: enough to thin the widest structure on that grid to its skeleton
TOPO_ITERS = {"mask": 10, "center": 5, "center_hi": 8}
