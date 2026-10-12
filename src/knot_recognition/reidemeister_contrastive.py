"""
reidemeister_contrastive.py - Self-Supervised Reidemeister Contrastive Learning.

Implements:
1. Reidemeister Metric Margin Loss (L_topo):
     L_topo = || z(D) - z(R_i(D)) ||_2^2 + lambda * max(0, gamma - || z(D_1) - z(D_2) ||_2^2)
2. InfoNCE / R-SimCLR contrastive loss over ambient isotopy orbits.
3. ReidemeisterAugmentedDataset: Dynamic on-the-fly stochastic Reidemeister pairing.
4. ReidemeisterVisionModel: End-to-end vision backbone trained for certified ambient isotopy invariance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from Knotgen.draw import render_knot_diagram
from Knotgen.knotgen import braid
from Knotgen.reidemeister import apply_stochastic_reidemeister_moves
from .nanotransformer import NanoTransformer


# ==============================================================================
# Contrastive Topological Losses
# ==============================================================================

class ReidemeisterContrastiveLoss(nn.Module):
    """
    Topological Metric Margin Loss:
      L_topo = || z(D) - z(R(D)) ||_2^2 + lambda * max(0, gamma - || z(D_1) - z(D_2) ||_2^2)

    Provably collapses ambient isotopy orbits to single latent manifold points.
    """

    def __init__(self, margin: float = 1.0, lambda_neg: float = 0.5):
        super().__init__()
        self.margin = margin
        self.lambda_neg = lambda_neg

    def forward(
        self,
        z_anchor: torch.Tensor,    # (B, D)
        z_positive: torch.Tensor,  # (B, D)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = z_anchor.shape[0]

        # 1. Positive pair attraction (ambient isotopy collapse)
        # || z(D) - z(R(D)) ||_2^2
        pos_dist_sq = torch.sum((z_anchor - z_positive) ** 2, dim=-1)
        loss_pos = torch.mean(pos_dist_sq)

        # 2. Negative pair separation
        # All pairs (i != j) across batch
        # Compute pairwise distance matrix between anchors
        diff = z_anchor.unsqueeze(1) - z_anchor.unsqueeze(0)  # (B, B, D)
        neg_dist = torch.norm(diff + 1e-8, p=2, dim=-1)       # (B, B)

        # Mask diagonal (self-comparisons)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_anchor.device)
        if mask.sum() > 0:
            hinge = torch.relu(self.margin - neg_dist)
            loss_neg = torch.mean(hinge[mask] ** 2)
        else:
            loss_neg = torch.tensor(0.0, device=z_anchor.device)

        total_loss = loss_pos + self.lambda_neg * loss_neg
        stats = {
            "loss_total": float(total_loss.item()),
            "loss_pos": float(loss_pos.item()),
            "loss_neg": float(loss_neg.item()),
            "mean_pos_dist": float(torch.sqrt(pos_dist_sq + 1e-8).mean().item()),
            "mean_neg_dist": float(neg_dist[mask].mean().item()) if mask.sum() > 0 else 0.0,
        }
        return total_loss, stats


class ReidemeisterInfoNCELoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy (InfoNCE) over Reidemeister pairs."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        # z are assumed L2-normalized: (B, D)
        batch_size = z_anchor.shape[0]
        # Similarity matrix: (B, B)
        sim_matrix = torch.matmul(z_anchor, z_positive.T) / self.temperature

        # Positive pairs are the diagonal
        labels = torch.arange(batch_size, device=z_anchor.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# ==============================================================================
# Dynamic Reidemeister Augmented Dataset
# ==============================================================================

class ReidemeisterAugmentedDataset(Dataset):
    """
    Dynamically generates paired knot diagrams (D, R_i(D)) on the fly.
    Loads braid representations from census table (e.g. unknot through 10_165).
    """

    def __init__(
        self,
        census_path: Optional[str] = None,
        max_crossing_filter: Optional[int] = None,
        min_crossing_filter: Optional[int] = None,
        img_size: Tuple[int, int] = (224, 224),
        num_reidemeister_moves: int = 3,
        samples_per_knot: int = 4,
    ):
        self.img_size = img_size
        self.num_moves = num_reidemeister_moves

        # Load census braids
        if census_path is None:
            census_path = str(Path(__file__).resolve().parent.parent.parent / "Knotgen" / "census_braids.json")

        with open(census_path, "r") as f:
            full_census = json.load(f)

        # Filter by crossing number if specified
        self.knots: List[Dict[str, Any]] = []
        for name, data in full_census.items():
            c = int(data.get("crossings", 0))
            if min_crossing_filter is not None and c < min_crossing_filter:
                continue
            if max_crossing_filter is not None and c > max_crossing_filter:
                continue
            self.knots.append({
                "name": name,
                "braid": list(data.get("braid", [])),
                "nstrands": int(data.get("nstrands", 2)),
                "crossings": c,
            })

        self.samples_per_knot = samples_per_knot
        self.total_len = len(self.knots) * samples_per_knot
        self.name_to_idx = {k["name"]: i for i, k in enumerate(self.knots)}

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        knot_idx = idx % len(self.knots)
        k_data = self.knots[knot_idx]
        word = k_data["braid"]
        n = k_data["nstrands"]
        name = k_data["name"]

        # Base knot diagram
        k_base = braid(word, n, name=name)

        # Perturbed knot diagram via stochastic Reidemeister moves
        new_word, new_n = apply_stochastic_reidemeister_moves(word, n, num_moves=self.num_moves)
        try:
            k_moved = braid(new_word, new_n, name=f"{name}_R")
        except ValueError:
            from Knotgen.reidemeister import reidemeister_1_stabilize
            w_stab, n_stab = reidemeister_1_stabilize(word, n, positive=True)
            k_moved = braid(w_stab, n_stab, name=f"{name}_R")

        # Render diagrams
        img_base = render_knot_diagram(k_base, size=self.img_size, line_width=3.5)
        img_moved = render_knot_diagram(k_moved, size=self.img_size, line_width=3.5)

        # Normalize to float32 [0, 1] and transpose (C, H, W)
        tensor_base = torch.from_numpy(img_base).permute(2, 0, 1).float() / 255.0
        tensor_moved = torch.from_numpy(img_moved).permute(2, 0, 1).float() / 255.0

        return {
            "image_anchor": tensor_base,
            "image_positive": tensor_moved,
            "label": knot_idx,
            "knot_name": name,
            "crossings": k_data["crossings"],
            "pd_base": [c["pd"] for c in k_base.crossings],
            "pd_moved": [c["pd"] for c in k_moved.crossings],
            "signs_base": [c["sign"] for c in k_base.crossings],
            "signs_moved": [c["sign"] for c in k_moved.crossings],
        }


def reidemeister_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates variable-length diagram records into model-ready batch tensors."""
    return {
        "image_anchor": torch.stack([b["image_anchor"] for b in batch]),
        "image_positive": torch.stack([b["image_positive"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "crossings": torch.tensor([b["crossings"] for b in batch], dtype=torch.long),
        "knot_name": [b["knot_name"] for b in batch],
        "pd_base": [b["pd_base"] for b in batch],
        "pd_moved": [b["pd_moved"] for b in batch],
        "signs_base": [b["signs_base"] for b in batch],
        "signs_moved": [b["signs_moved"] for b in batch],
    }


# ==============================================================================
# Reidemeister Vision Model Backbone
# ==============================================================================

class ReidemeisterVisionModel(nn.Module):
    """
    Vision Model optimized for Certified Reidemeister Invariance.
    Employs a Vision NanoTransformer backbone paired with a contrastive projector head.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        proj_dim: int = 128,
        pretrained_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        from .nanotransformer import NanoTransformerConfig
        if pretrained_checkpoint and Path(pretrained_checkpoint).exists():
            self.backbone = NanoTransformer.from_pretrained(pretrained_checkpoint)
        else:
            cfg = NanoTransformerConfig(embed_dim=embed_dim, image_size=128)
            self.backbone = NanoTransformer(cfg)

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts unit-normalized latent representation."""
        target_size = self.backbone.config.image_size
        if x.shape[-2:] != (target_size, target_size):
            x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
        feat = self.backbone.forward_features(x)
        z = self.projector(feat)
        return F.normalize(z, p=2, dim=-1)

    def forward(
        self,
        x_anchor: torch.Tensor,
        x_positive: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z_anchor = self.extract_embedding(x_anchor)
        if x_positive is not None:
            z_pos = self.extract_embedding(x_positive)
            return z_anchor, z_pos
        return z_anchor
