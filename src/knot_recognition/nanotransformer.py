"""
nanotransformer.py - Vision NanoTransformer for Knot Diagram Classification & Retrieval.

A lightweight Vision Transformer (ViT) architecture engineered specifically for
knot diagram identification, embedding generation, and active similarity retrieval
across all 250 prime knot types (from 0_unknot up to 10_165).

Provides:
- NanoTransformer: Compact (~0.4M - 0.8M parameters) Vision Transformer
- Feature embedding extraction (128-d unit-normalized topological representations)
- Fast active classification and top-k probabilistic ranking
- Cosine-similarity nearest-neighbor fetching of topologically similar knots
- Direct integration with KnotTraverser to supply crossing and invariant priors
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from .utils import imread_any


@dataclass
class NanoTransformerConfig:
    image_size: int = 128
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    num_classes: int = 250
    dropout: float = 0.1
    attention_dropout: float = 0.1


class PatchEmbedding(nn.Module):
    """Splits image into non-overlapping patches and projects to embed_dim."""

    def __init__(self, image_size: int = 128, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = self.proj(x)  # (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with scaled dot-product and dropout."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1, attention_dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLPBlock(nn.Module):
    """Feed-Forward Network with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer Block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1, attention_dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout, attention_dropout=attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLPBlock(embed_dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def denoise_for_nanotransformer(
    pil_img: Image.Image,
    noise_threshold: float = 600.0,
) -> Tuple[Image.Image, float, bool]:
    """Measures texture noise and applies edge-preserving bilateral filtering & aspect normalization."""
    arr = np.array(pil_img)
    if arr.ndim == 2:
        gray = arr
    else:
        import cv2
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    import cv2
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_noisy = lap_var > noise_threshold

    w, h = pil_img.size
    # Saliency crop if wide/tall aspect ratio with central knot
    if w > h * 1.35 or h > w * 1.35:
        cx, cy = w // 2, h // 2
        r = min(w, h) // 2
        cropped = pil_img.crop((cx - r, cy - r, cx + r, cy + r))
    else:
        cropped = pil_img

    if is_noisy:
        c_arr = np.array(cropped)
        if c_arr.ndim == 3:
            c_gray = cv2.cvtColor(c_arr, cv2.COLOR_RGB2GRAY)
            smoothed = cv2.bilateralFilter(c_gray, 9, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(smoothed)
            out_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))
        else:
            smoothed = cv2.bilateralFilter(c_arr, 9, 75, 75)
            out_img = Image.fromarray(smoothed)
    else:
        out_img = cropped

    return out_img, lap_var, is_noisy


class NanoTransformer(nn.Module):
    """
    Vision NanoTransformer for 250-Class Knot Recognition and Similarity Retrieval.
    """

    def __init__(self, config: Optional[NanoTransformerConfig] = None):
        super().__init__()
        self.config = config or NanoTransformerConfig()
        cfg = self.config

        self.patch_embed = PatchEmbedding(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                attention_dropout=cfg.attention_dropout,
            )
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        # Class lookup and prototype embeddings for similarity retrieval
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.prototypes: Optional[torch.Tensor] = None  # (num_classes, embed_dim)
        self.census_metadata: Dict[str, Dict[str, Any]] = {}

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def set_classes(self, classes: Sequence[str]):
        """Sets the ordered class names and builds lookup dictionaries."""
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

    def set_prototypes(self, prototypes: torch.Tensor):
        """Sets the normalized embedding prototypes for the classes."""
        self.prototypes = F.normalize(prototypes, p=2, dim=-1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts token representation and returns normalized CLS embedding."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return cls_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard classification forward pass returning logits (B, num_classes)."""
        feat = self.forward_features(x)
        logits = self.head(feat)
        return logits

    def extract_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Returns feature embedding vector (B, embed_dim), unit-normalized by default."""
        feat = self.forward_features(x)
        if normalize:
            feat = F.normalize(feat, p=2, dim=-1)
        return feat



    @torch.inference_mode()
    def predict_image(
        self,
        img: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5,
        device: Optional[torch.device] = None,
        noise_aware: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs noise-aware image classification, returning top-k predictions and embedding.
        """
        dev = device or next(self.parameters()).device
        self.eval()

        if isinstance(img, (str, Path)):
            img_np = imread_any(str(img))
            pil_img = Image.fromarray(img_np)
        elif isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img

        noise_level = 0.0
        is_noisy = False
        target_img = pil_img

        if noise_aware:
            denoised_img, noise_level, is_noisy = denoise_for_nanotransformer(pil_img)
            target_img = denoised_img

        transform = T.Compose([
            T.Resize((self.config.image_size, self.config.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        tensor = transform(target_img.convert("RGB")).unsqueeze(0).to(dev)

        if noise_aware and is_noisy:
            # Dual-pass embedding fusion: prioritize denoised contour with subtle raw context
            raw_tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(dev)
            feat_clean = self.extract_embedding(tensor, normalize=True)
            feat_raw = self.extract_embedding(raw_tensor, normalize=True)
            feat = F.normalize(0.80 * feat_clean + 0.20 * feat_raw, p=2, dim=-1)
        else:
            feat = self.extract_embedding(tensor, normalize=True)

        logits = self.head(feat)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        predictions = []
        for rank, idx in enumerate(top_indices, start=1):
            class_name = self.idx_to_class.get(int(idx), f"class_{idx}")
            prob = float(probs[idx])
            meta = self.census_metadata.get(class_name, {})
            predictions.append({
                "rank": rank,
                "label": class_name,
                "probability": prob,
                "crossings": meta.get("crossings", _infer_crossings_from_name(class_name)),
                "alexander_polynomial": meta.get("alexander_polynomial", ""),
                "determinant": meta.get("determinant", None),
                "pd_code": meta.get("pd_code", ""),
                "gauss_code": meta.get("gauss_code", ""),
            })

        feat_np = feat[0].cpu().numpy()
        return {
            "predictions": predictions,
            "embedding": feat_np,
            "top_label": predictions[0]["label"] if predictions else "unknown",
            "top_probability": predictions[0]["probability"] if predictions else 0.0,
            "noise_level": noise_level,
            "is_noisy": is_noisy,
        }

    @torch.inference_mode()
    def fetch_similar_knots(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        top_k: int = 5,
        exclude_top_exact: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-k nearest topological neighbors via cosine similarity
        against precomputed knot prototypes.
        """
        if self.prototypes is None:
            return []

        if isinstance(query_embedding, np.ndarray):
            q_emb = torch.from_numpy(query_embedding).float()
        else:
            q_emb = query_embedding.float()

        if q_emb.ndim == 1:
            q_emb = q_emb.unsqueeze(0)
        q_emb = F.normalize(q_emb, p=2, dim=-1)

        dev = self.prototypes.device
        q_emb = q_emb.to(dev)

        # Cosine similarity matrix (1, num_classes)
        sims = (q_emb @ self.prototypes.t())[0].cpu().numpy()

        sorted_indices = np.argsort(sims)[::-1]
        results = []
        start_idx = 1 if exclude_top_exact and len(sorted_indices) > 1 else 0
        for rank, idx in enumerate(sorted_indices[start_idx : start_idx + top_k], start=1):
            class_name = self.idx_to_class.get(int(idx), f"class_{idx}")
            score = float(sims[idx])
            meta = self.census_metadata.get(class_name, {})
            results.append({
                "rank": rank,
                "label": class_name,
                "similarity": score,
                "crossings": meta.get("crossings", _infer_crossings_from_name(class_name)),
                "alexander_polynomial": meta.get("alexander_polynomial", ""),
                "determinant": meta.get("determinant", None),
                "pd_code": meta.get("pd_code", ""),
                "gauss_code": meta.get("gauss_code", ""),
            })
        return results


def _infer_crossings_from_name(name: str) -> int:
    """Infers crossing count from standard knot notation like '7_1' -> 7, '0_unknot' -> 0."""
    if name in ("0_unknot", "0_1", "unknot"):
        return 0
    parts = name.split("_")
    try:
        return int(parts[0])
    except Exception:
        return 0


def load_census_metadata_table(
    mapping_csv: Optional[Union[str, Path]] = None,
    census_braids_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Loads combined topological metadata for all 250 knots."""
    table: Dict[str, Dict[str, Any]] = {}

    # 1. Load from mapping_generated.csv
    if mapping_csv and Path(mapping_csv).exists():
        with open(mapping_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "").strip()
                if not label:
                    continue
                table[label] = {
                    "crossings": _infer_crossings_from_name(label),
                    "pd_code": row.get("pd_code", ""),
                    "gauss_code": row.get("gauss_code", ""),
                }

    # 2. Augment from census_braids.json if available
    if census_braids_json and Path(census_braids_json).exists():
        with open(census_braids_json, "r", encoding="utf-8") as f:
            census = json.load(f)
            for label, data in census.items():
                if label not in table:
                    table[label] = {}
                table[label]["crossings"] = int(data.get("crossings", _infer_crossings_from_name(label)))
                table[label]["braid"] = data.get("braid", [])
                table[label]["nstrands"] = data.get("nstrands", 0)

    # 0_unknot default
    if "0_unknot" not in table:
        table["0_unknot"] = {
            "crossings": 0,
            "pd_code": "PD[]",
            "gauss_code": "",
            "alexander_polynomial": "1",
            "determinant": 1,
        }

    return table


def save_nanotransformer_checkpoint(
    model: NanoTransformer,
    checkpoint_path: Union[str, Path],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves complete NanoTransformer weights, config, classes, prototypes, and metadata."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "config": asdict(model.config),
        "classes": model.classes,
        "class_to_idx": model.class_to_idx,
        "idx_to_class": model.idx_to_class,
        "prototypes": model.prototypes.cpu() if model.prototypes is not None else None,
        "census_metadata": model.census_metadata,
        "extra_meta": extra_meta or {},
    }
    torch.save(state, str(checkpoint_path))


def load_nanotransformer(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> NanoTransformer:
    """Loads a trained NanoTransformer model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    dev = torch.device(device) if device else torch.device("cpu")
    checkpoint = torch.load(str(checkpoint_path), map_location=dev)

    cfg_dict = checkpoint.get("config", {})
    config = NanoTransformerConfig(**cfg_dict)
    model = NanoTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    classes = checkpoint.get("classes", [])
    model.set_classes(classes)
    if checkpoint.get("prototypes") is not None:
        model.set_prototypes(checkpoint["prototypes"].to(dev))
    model.census_metadata = checkpoint.get("census_metadata", {})

    model.to(dev)
    model.eval()
    return model
