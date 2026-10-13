"""
run_honest_protocol.py - Rigorous Scientific Evaluation Protocol for Reidemeister Canonicalization.

Tests the hypothesis:
  "An exact R1/R2 canonicalization layer plus a learned encoder generalizes to unseen knots
   and to new isotopic diagrams better than the same encoder without it."

Methodological Protocol:
  1. Same Input: Signed Planar Diagram (PD) code / 4-valent graph for all graph models.
     Rendered-image ViT evaluated as separate reference row.
  2. High-Perturbation Data: Diagrams generated with R1, R2, R3, and flype moves up to min(c) + 6 crossings.
  3. Knot-Level Split:
     - Train: Knots with c <= 8 (36 prime knots + unknot = 37 knots). Mirror in same split.
     - Test Type A: Held-out unseen diagrams of seen knots (c <= 8).
     - Test Type B: Unseen knots: c = 9 (49 knots) and c = 10 (165 knots).
  4. Task 1: Same-Knot Verification with Hard Negatives (same crossing, same Alexander, same Jones).
     Evaluated with AUROC, Top-1, and Top-5 retrieval.
  5. Task 2: Invariant Prediction on Unseen Knots without Cheap Formulas:
     signature, three_genus, braid_index, unknotting_number.
     Compared directly against Trivial Predictors (Mean and Linear fit on c(K), w(K)).
  6. 5 Random Seeds with equal training budgets.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from Knotgen.flype_generator import KnotDiagram, generate_isotopic_diagrams
from Knotgen.draw import render_knot_diagram
from Knotgen.knotgen import braid
from knot_recognition.canonical_models import (
    AlexanderFeatureBaseline,
    BaseKnotGraphModel,
    CanonicalHashLookupTable,
    build_graph_tensors_from_pd,
)
from knot_recognition.knotinfo_bench import KnotInfoBenchmark, KnotPair, KnotPairSampler
from knot_recognition.nanotransformer import NanoTransformer, NanoTransformerConfig


# ==============================================================================
# Dataset Generation & Caching
# ==============================================================================

def prepare_benchmark_dataset(
    bench: KnotInfoBenchmark,
    num_train_diags: int = 8,
    num_test_diags: int = 4,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generates and caches verified isotopic diagrams for all splits."""
    if cache_path and cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            print(f"[Dataset] Loaded cached diagrams from {cache_path}")
            return data
        except Exception:
            pass

    print("[Dataset] Generating multi-diagram benchmark dataset with R1, R2, R3, and flypes...")
    train_knots, test_9, test_10 = bench.get_splits()

    dataset: Dict[str, Any] = {
        "train": {},          # knot -> list of diagrams (training set)
        "test_seen_a": {},    # knot -> list of diagrams (unseen diagrams of c <= 8)
        "test_unseen_9": {},  # knot -> list of diagrams (c = 9)
        "test_unseen_10": {}, # knot -> list of diagrams (c = 10)
    }

    # 1. Train knots (c <= 8): generate (num_train_diags + num_test_diags) diagrams
    for name in train_knots:
        data = bench.census.get(name, {})
        w = data.get("braid", [])
        n = data.get("nstrands", 2)
        total_diags = generate_isotopic_diagrams(name, w, n, num_diagrams=num_train_diags + num_test_diags, seed=42)
        dataset["train"][name] = [
            {"pd": d.pd_code, "signs": d.signs, "crossings": d.crossing_count, "moves": d.moves_history}
            for d in total_diags[:num_train_diags]
        ]
        dataset["test_seen_a"][name] = [
            {"pd": d.pd_code, "signs": d.signs, "crossings": d.crossing_count, "moves": d.moves_history}
            for d in total_diags[num_train_diags:]
        ]

    # 2. Test 9-crossing knots
    for name in test_9:
        data = bench.census.get(name, {})
        w = data.get("braid", [])
        n = data.get("nstrands", 2)
        diags = generate_isotopic_diagrams(name, w, n, num_diagrams=num_test_diags, seed=42)
        dataset["test_unseen_9"][name] = [
            {"pd": d.pd_code, "signs": d.signs, "crossings": d.crossing_count, "moves": d.moves_history}
            for d in diags
        ]

    # 3. Test 10-crossing knots
    for name in test_10:
        data = bench.census.get(name, {})
        w = data.get("braid", [])
        n = data.get("nstrands", 2)
        diags = generate_isotopic_diagrams(name, w, n, num_diagrams=num_test_diags, seed=42)
        dataset["test_unseen_10"][name] = [
            {"pd": d.pd_code, "signs": d.signs, "crossings": d.crossing_count, "moves": d.moves_history}
            for d in diags
        ]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"[Dataset] Saved diagrams to {cache_path}")

    return dataset


# ==============================================================================
# Model Training Routines
# ==============================================================================

class GraphClassificationHead(nn.Module):
    """Linear classifier on top of GNN unit embeddings for training."""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


def train_graph_model(
    model: BaseKnotGraphModel,
    train_data: Dict[str, List[Dict[str, Any]]],
    knot_to_idx: Dict[str, int],
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    is_augmented: bool = False,
    is_shuffled: bool = False,
    seed: int = 42,
) -> Tuple[BaseKnotGraphModel, float]:
    """Trains a graph model on the train knot classes with equal budget."""
    torch.manual_seed(seed)
    model = model.to(device)
    head = GraphClassificationHead(model.backbone.pool_proj[-1].out_features, len(knot_to_idx)).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)

    # Prepare flat sample list
    samples = []
    for knot_name, diags in train_data.items():
        lbl = knot_to_idx[knot_name]
        for d in diags:
            samples.append((knot_name, d["pd"], d["signs"], lbl))

    if is_shuffled:
        rng = np.random.default_rng(seed)
        shuffled_labels = rng.permutation([s[3] for s in samples])
        samples = [(s[0], s[1], s[2], int(shuffled_labels[i])) for i, s in enumerate(samples)]

    model.train()
    head.train()
    final_loss = 0.0

    batch_size = 16
    if not is_augmented:
        # Precompute canonicalized graph tensors once
        precomputed = []
        for knot_name, pd_list, signs, label in samples:
            red_pd, red_signs = model.canonicalize_pd(pd_list, signs)
            nf, ei, ef = build_graph_tensors_from_pd(red_pd, red_signs, device)
            precomputed.append((nf, ei, ef, label))

        for epoch in range(epochs):
            random.shuffle(precomputed)
            epoch_loss = 0.0
            optimizer.zero_grad()
            batch_accum = 0
            for i, (nf, ei, ef, label) in enumerate(precomputed):
                z = model.backbone(nf, ei, ef)
                logits = head(z.unsqueeze(0))
                target = torch.tensor([label], dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                batch_accum += 1
                epoch_loss += loss.item()
                if batch_accum >= batch_size or (i + 1) == len(precomputed):
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_accum = 0
            final_loss = epoch_loss / max(1, len(precomputed))
    else:
        for epoch in range(epochs):
            random.shuffle(samples)
            epoch_loss = 0.0
            optimizer.zero_grad()
            batch_accum = 0
            for i, (knot_name, pd_list, signs, label) in enumerate(samples):
                cur_pd, cur_signs = list(pd_list), list(signs)
                if random.random() < 0.5 and cur_pd:
                    try:
                        import snappy
                        L = snappy.Link(list(cur_pd))
                        L.backtrack(steps=random.randint(1, 3), prob_type_1=0.5, prob_type_2=0.5)
                        cur_pd = [tuple(c) for c in L.PD_code()]
                        cur_signs = [1] * len(cur_pd)
                    except Exception:
                        pass
                red_pd, red_signs = model.canonicalize_pd(cur_pd, cur_signs)
                node_feats, edge_index, edge_feats = build_graph_tensors_from_pd(red_pd, red_signs, device)
                z = model.backbone(node_feats, edge_index, edge_feats)
                logits = head(z.unsqueeze(0))
                target = torch.tensor([label], dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                batch_accum += 1
                epoch_loss += loss.item()
                if batch_accum >= batch_size or (i + 1) == len(samples):
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_accum = 0
            final_loss = epoch_loss / max(1, len(samples))

    model.eval()
    return model, final_loss


def train_contrastive_model(
    model: BaseKnotGraphModel,
    train_data: Dict[str, List[Dict[str, Any]]],
    epochs: int = 20,
    lr: float = 1e-3,
    tau: float = 0.1,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tuple[BaseKnotGraphModel, float]:
    """Trains ContrastiveGNN using InfoNCE over positive diagram pairs."""
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Pre-build graph tensors for all training diagrams
    cached_tensors: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for k_name, diags in train_data.items():
        cached_tensors[k_name] = []
        for d in diags:
            red_pd, red_signs = model.canonicalize_pd(d["pd"], d["signs"])
            nf, ei, ef = build_graph_tensors_from_pd(red_pd, red_signs, device)
            cached_tensors[k_name].append((nf, ei, ef))

    knot_names = list(train_data.keys())
    model.train()
    final_loss = 0.0
    batch_size = 16

    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(knot_names)
        optimizer.zero_grad()
        batch_accum = 0
        for i, anchor_knot in enumerate(knot_names):
            t_list = cached_tensors[anchor_knot]
            if len(t_list) < 2:
                continue
            idx1, idx2 = random.sample(range(len(t_list)), 2)
            nf1, ei1, ef1 = t_list[idx1]
            nf2, ei2, ef2 = t_list[idx2]

            neg_knots = [k for k in knot_names if k != anchor_knot]
            sampled_negs = random.sample(neg_knots, min(5, len(neg_knots)))
            z1 = model.backbone(nf1, ei1, ef1)
            z2 = model.backbone(nf2, ei2, ef2)
            pos_sim = torch.dot(z1, z2) / tau

            neg_sims = []
            for nk in sampled_negs:
                neg_choice = random.choice(cached_tensors[nk])
                zn = model.backbone(neg_choice[0], neg_choice[1], neg_choice[2])
                neg_sims.append(torch.dot(z1, zn) / tau)

            logits = torch.stack([pos_sim] + neg_sims).unsqueeze(0)
            target = torch.zeros(1, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            batch_accum += 1
            epoch_loss += loss.item()
            if batch_accum >= batch_size or (i + 1) == len(knot_names):
                optimizer.step()
                optimizer.zero_grad()
                batch_accum = 0

        final_loss = epoch_loss / max(1, len(knot_names))

    model.eval()
    return model, final_loss


def train_rendered_vit(
    train_data: Dict[str, List[Dict[str, Any]]],
    bench: KnotInfoBenchmark,
    knot_to_idx: Dict[str, int],
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Tuple[NanoTransformer, float]:
    """Trains RenderedViT reference row on rendered 64x64 images."""
    torch.manual_seed(seed)
    cfg = NanoTransformerConfig(
        image_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=128,
        depth=4,
        num_heads=4,
        num_classes=len(knot_to_idx),
    )
    model = NanoTransformer(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Render images for train set
    samples = []
    for knot_name, diags in train_data.items():
        lbl = knot_to_idx[knot_name]
        data = bench.census.get(knot_name, {})
        w = data.get("braid", [])
        n = data.get("nstrands", 2)
        if w:
            k_obj = braid(w, n, name=knot_name)
            img = render_knot_diagram(k_obj, size=(64, 64))
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            for _ in diags:
                samples.append((img_tensor, lbl))

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        random.shuffle(samples)
        epoch_loss = 0.0
        for img_tensor, label in samples:
            optimizer.zero_grad()
            x = img_tensor.unsqueeze(0).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, torch.tensor([label], dtype=torch.long, device=device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / max(1, len(samples))

    model.eval()
    return model, final_loss


# ==============================================================================
# Evaluation Protocols (Task 1 & Task 2)
# ==============================================================================

def get_model_embeddings(
    model_obj: Any,
    dataset_split: Dict[str, List[Dict[str, Any]]],
    device: torch.device = torch.device("cpu"),
    is_vit: bool = False,
    bench: Optional[KnotInfoBenchmark] = None,
) -> Dict[str, List[np.ndarray]]:
    """Extracts unit embeddings for every diagram in a dataset split."""
    embeddings: Dict[str, List[np.ndarray]] = {}

    for knot_name, diags in dataset_split.items():
        embeddings[knot_name] = []
        for d in diags:
            if is_vit:
                data = bench.census.get(knot_name, {})
                w = data.get("braid", [])
                n = data.get("nstrands", 2)
                if w:
                    k_obj = braid(w, n, name=knot_name)
                    img = render_knot_diagram(k_obj, size=(64, 64))
                    x = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                    with torch.no_grad():
                        z = model_obj.forward_features(x).squeeze(0).cpu().numpy()
                else:
                    z = np.zeros(128, dtype=np.float32)
            elif isinstance(model_obj, AlexanderFeatureBaseline):
                z = model_obj.extract_features(d["pd"], d["signs"], knot_name=knot_name).numpy()
            elif isinstance(model_obj, CanonicalHashLookupTable):
                z = model_obj.predict(d["pd"], d["signs"]).numpy()
            else:
                with torch.no_grad():
                    z = model_obj(d["pd"], d["signs"], device).cpu().numpy()

            norm = np.linalg.norm(z)
            if norm > 0:
                z = z / norm
            embeddings[knot_name].append(z)

    return embeddings


def evaluate_task1_verification(
    embeddings: Dict[str, List[np.ndarray]],
    bench: KnotInfoBenchmark,
    candidate_knots: List[str],
    dataset_split: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Task 1: Same-knot verification with hard negatives using rebuilt KnotPairSampler.
    Positives strictly share a knot identity; negatives strictly never share knot identity,
    with matched crossing count.
    Returns: AUROC, Top-1 accuracy, Top-5 accuracy.
    """
    sampler = KnotPairSampler(bench)
    split_to_use = dataset_split if dataset_split is not None else {
        k: [{"dummy": i} for i in range(len(v))] for k, v in embeddings.items()
    }
    pairs = sampler.sample_pairs_for_dataset(
        split_to_use, candidate_knots, num_negatives_per_anchor=5, seed=seed
    )

    y_true: List[int] = []
    y_scores: List[float] = []
    anchor_scores: Dict[str, Dict[str, Any]] = {}

    for p in pairs:
        emb1_list = embeddings.get(p.knot_1, [])
        emb2_list = embeddings.get(p.knot_2, [])
        if not emb1_list or not emb2_list:
            continue

        emb1 = emb1_list[0]
        emb2 = emb2_list[1] if (p.is_positive and len(emb2_list) > 1) else emb2_list[0]
        sim = float(np.dot(emb1, emb2))

        y_true.append(1 if p.is_positive else 0)
        y_scores.append(sim)

        if p.knot_1 not in anchor_scores:
            anchor_scores[p.knot_1] = {"pos": None, "negs": []}
        if p.is_positive:
            anchor_scores[p.knot_1]["pos"] = sim
        else:
            anchor_scores[p.knot_1]["negs"].append(sim)

    top1_correct = 0
    top5_correct = 0
    total_queries = 0

    for k, sc in anchor_scores.items():
        if sc["pos"] is not None and sc["negs"]:
            total_queries += 1
            pos_sim = sc["pos"]
            candidates = [pos_sim] + sc["negs"]
            ranked = sorted(candidates, reverse=True)
            if ranked[0] == pos_sim:
                top1_correct += 1
            if pos_sim in ranked[:5]:
                top5_correct += 1

    try:
        auroc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auroc = 0.5

    top1 = float(top1_correct / max(1, total_queries))
    top5 = float(top5_correct / max(1, total_queries))

    return {"auroc": auroc, "top1": top1, "top5": top5}


def evaluate_task2_invariants(
    train_embeddings: Dict[str, List[np.ndarray]],
    test_embeddings: Dict[str, List[np.ndarray]],
    bench: KnotInfoBenchmark,
    targets: Tuple[str, ...] = ("signature", "three_genus", "braid_index", "unknotting_number"),
) -> Dict[str, Dict[str, float]]:
    """
    Task 2: Invariant prediction on unseen knots using linear probe on knot embeddings.
    Returns MAE and accuracy for each target invariant.
    """
    X_train, y_train_dict = [], {t: [] for t in targets}
    for knot_name, emb_list in train_embeddings.items():
        if not emb_list:
            continue
        entry = bench.get_knot(knot_name)
        # Average embedding across diagrams
        mean_emb = np.mean(emb_list, axis=0)
        X_train.append(mean_emb)
        for t in targets:
            y_train_dict[t].append(getattr(entry, t))

    X_train = np.array(X_train, dtype=np.float32)

    X_test, y_test_dict = [], {t: [] for t in targets}
    for knot_name, emb_list in test_embeddings.items():
        if not emb_list:
            continue
        entry = bench.get_knot(knot_name)
        mean_emb = np.mean(emb_list, axis=0)
        X_test.append(mean_emb)
        for t in targets:
            y_test_dict[t].append(getattr(entry, t))

    X_test = np.array(X_test, dtype=np.float32)

    results: Dict[str, Dict[str, float]] = {}
    for t in targets:
        y_tr = np.array(y_train_dict[t], dtype=np.float32)
        y_te = np.array(y_test_dict[t], dtype=np.float32)

        clf = Ridge(alpha=1.0)
        clf.fit(X_train, y_tr)
        y_pred = clf.predict(X_test)

        mae = float(np.mean(np.abs(y_te - y_pred)))
        acc = float(np.mean(np.round(y_pred) == np.round(y_te)))
        results[t] = {"mae": mae, "acc": acc}

    return results


# ==============================================================================
# Full Multi-Seed Experiment Suite
# ==============================================================================

class FixedChordFeatureBaseline:
    """Non-neural baseline evaluating mean and max of hand-crafted node features directly."""
    def __init__(self, mode: str = "r1_r2"):
        self.mode = mode
        self.base = BaseKnotGraphModel(mode=mode)

    def __call__(self, pd_list: Sequence[Sequence[int]], signs: Optional[Sequence[int]] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        red_pd, red_s = self.base.canonicalize_pd(pd_list, signs)
        nf, ei, ef = build_graph_tensors_from_pd(red_pd, red_s)
        vec = torch.cat([nf.mean(dim=0), nf.max(dim=0).values], dim=-1)
        norm = torch.norm(vec, p=2)
        return (vec / norm) if norm > 0 else vec


def run_experiment_suite(
    seeds: List[int] = [42, 43, 44, 45, 46],
    epochs: int = 20,
    dataset_cache: Optional[Path] = None,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs the full 5-seed protocol across all models and explicit controls."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Reidemeister Canonicalization Benchmark (Device: {device}) ===")

    bench = KnotInfoBenchmark()
    train_knots, test_9, test_10 = bench.get_splits()
    dataset = prepare_benchmark_dataset(bench, num_train_diags=8, num_test_diags=4, cache_path=dataset_cache)

    knot_to_idx = {k: i for i, k in enumerate(train_knots)}

    # Evaluate Trivial Predictors directly
    triv_9 = bench.evaluate_trivial_predictors(train_knots, test_9)
    triv_10 = bench.evaluate_trivial_predictors(train_knots, test_10)

    # Models list with explicit symmetric controls
    model_configs = [
        ("raw_gnn", "Raw GNN (No Canonicalization)"),
        ("canonical_r1r2", "Canonical GNN (R1/R2)"),
        ("canonical_r1_only", "Canonical GNN (R1-only Ablation)"),
        ("canonical_r2_only", "Canonical GNN (R2-only Ablation)"),
        ("snappy_simplify", "SnapPy Simplify (Upper Bound)"),
        ("augmented_gnn", "Augmented GNN"),
        ("contrastive_gnn", "Contrastive GNN"),
        ("fixed_chord_raw", "Fixed Chord Features (Raw Input)"),
        ("fixed_chord_canon", "Fixed Chord Features (Canonical Input)"),
        ("random_raw", "Random Untrained GNN (Raw Input)"),
        ("random_canon", "Random Untrained GNN (Canonical Input)"),
        ("shuffled_raw", "Shuffled-Label GNN (Raw Input)"),
        ("shuffled_canon", "Shuffled-Label GNN (Canonical Input)"),
        ("lookup_table", "Lookup-Table Hash"),
        ("non_learned_alex", "Non-Learned Baseline (Alexander+c+w)"),
        ("rendered_vit", "Rendered ViT (Reference Row)"),
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in model_configs}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Running Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")

        for m_key, m_label in model_configs:
            t0 = time.time()
            if m_key == "lookup_table":
                lut = CanonicalHashLookupTable()
                for k, diags in dataset["train"].items():
                    for d in diags:
                        lut.register(k, d["pd"], d["signs"])
                trained_model = lut
            elif m_key == "non_learned_alex":
                trained_model = AlexanderFeatureBaseline(bench=bench)
            elif m_key == "fixed_chord_raw":
                trained_model = FixedChordFeatureBaseline(mode="raw")
            elif m_key == "fixed_chord_canon":
                trained_model = FixedChordFeatureBaseline(mode="r1_r2")
            elif m_key == "random_raw":
                trained_model = BaseKnotGraphModel(mode="raw").to(device)
                trained_model.eval()
            elif m_key == "random_canon":
                trained_model = BaseKnotGraphModel(mode="r1_r2").to(device)
                trained_model.eval()
            elif m_key == "shuffled_raw":
                base_m = BaseKnotGraphModel(mode="raw")
                trained_model, _ = train_graph_model(
                    base_m, dataset["train"], knot_to_idx, epochs=epochs, is_shuffled=True, device=device, seed=seed
                )
            elif m_key == "shuffled_canon":
                base_m = BaseKnotGraphModel(mode="r1_r2")
                trained_model, _ = train_graph_model(
                    base_m, dataset["train"], knot_to_idx, epochs=epochs, is_shuffled=True, device=device, seed=seed
                )
            elif m_key == "rendered_vit":
                trained_model, _ = train_rendered_vit(
                    dataset["train"], bench, knot_to_idx, epochs=epochs, device=device, seed=seed
                )
            elif m_key == "contrastive_gnn":
                base_m = BaseKnotGraphModel(mode="raw")
                trained_model, _ = train_contrastive_model(
                    base_m, dataset["train"], epochs=epochs, device=device, seed=seed
                )
            elif m_key == "augmented_gnn":
                base_m = BaseKnotGraphModel(mode="raw")
                trained_model, _ = train_graph_model(
                    base_m, dataset["train"], knot_to_idx, epochs=epochs, is_augmented=True, device=device, seed=seed
                )
            elif m_key == "canonical_r1r2":
                base_m = BaseKnotGraphModel(mode="r1_r2")
                trained_model, _ = train_graph_model(
                    base_m, dataset["train"], knot_to_idx, epochs=epochs, device=device, seed=seed
                )
            elif m_key == "canonical_r1_only":
                base_m = BaseKnotGraphModel(mode="r1_only")
                trained_model, _ = train_graph_model(
                    base_m, dataset["train"], knot_to_idx, epochs=epochs, device=device, seed=seed
                )
            elif m_key == "canonical_r2_only":
                base_m = BaseKnotGraphModel(mode="r2_only")
                trained_model, _ = train_graph_model(
                    base_m, dataset["train"], knot_to_idx, epochs=epochs, device=device, seed=seed
                )
            elif m_key == "snappy_simplify":
                base_m = BaseKnotGraphModel(mode="snappy_simplify")
                trained_model, _ = train_graph_model(
                    base_m, dataset["train"], knot_to_idx, epochs=epochs, device=device, seed=seed
                )
            else:  # raw_gnn
                base_m = BaseKnotGraphModel(mode="raw")
                trained_model, _ = train_graph_model(
                    base_m, dataset["train"], knot_to_idx, epochs=epochs, device=device, seed=seed
                )

            # Compute embeddings
            is_vit = (m_key == "rendered_vit")
            emb_train = get_model_embeddings(trained_model, dataset["train"], device=device, is_vit=is_vit, bench=bench)
            emb_seen_a = get_model_embeddings(trained_model, dataset["test_seen_a"], device=device, is_vit=is_vit, bench=bench)
            emb_9 = get_model_embeddings(trained_model, dataset["test_unseen_9"], device=device, is_vit=is_vit, bench=bench)
            emb_10 = get_model_embeddings(trained_model, dataset["test_unseen_10"], device=device, is_vit=is_vit, bench=bench)

            # Task 1: Same-knot verification with rebuilt KnotPairSampler
            task1_seen_a = evaluate_task1_verification(emb_seen_a, bench, train_knots, dataset_split=dataset["test_seen_a"], seed=seed)
            task1_9 = evaluate_task1_verification(emb_9, bench, test_9, dataset_split=dataset["test_unseen_9"], seed=seed)
            task1_10 = evaluate_task1_verification(emb_10, bench, test_10, dataset_split=dataset["test_unseen_10"], seed=seed)

            # Task 2: Invariant prediction
            task2_9 = evaluate_task2_invariants(emb_train, emb_9, bench)
            task2_10 = evaluate_task2_invariants(emb_train, emb_10, bench)

            elapsed = time.time() - t0
            res = {
                "seed": seed,
                "time_sec": elapsed,
                "task1_seen_a": task1_seen_a,
                "task1_unseen_9": task1_9,
                "task1_unseen_10": task1_10,
                "task2_unseen_9": task2_9,
                "task2_unseen_10": task2_10,
            }
            all_results[m_key].append(res)
            print(f"  [{m_label}] Seen-A AUROC={task1_seen_a['auroc']:.3f}, Unseen-9 AUROC={task1_9['auroc']:.3f}, Sig MAE (c=9)={task2_9['signature']['mae']:.3f} ({elapsed:.1f}s)")

    # Aggregate results (mean +- std err)
    summary_report: Dict[str, Any] = {
        "metadata": {
            "num_seeds": len(seeds),
            "seeds": seeds,
            "train_knots": len(train_knots),
            "test_9_knots": len(test_9),
            "test_10_knots": len(test_10),
        },
        "trivial_predictors": {
            "c_9": triv_9,
            "c_10": triv_10,
        },
        "models": {},
    }

    print("\n" + "="*80)
    print("FINAL BENCHMARK SUMMARY (Mean across 5 seeds)")
    print("="*80)
    print(f"{'Model':<35} | {'Seen-A AUROC':<12} | {'Unseen-9 AUROC':<14} | {'Sig MAE (9)':<12} | {'Unknot MAE (9)':<14}")
    print("-" * 95)

    for m_key, m_label in model_configs:
        runs = all_results[m_key]
        n_r = len(runs)
        seen_auroc = [r["task1_seen_a"]["auroc"] for r in runs]
        unseen9_auroc = [r["task1_unseen_9"]["auroc"] for r in runs]
        sig_mae_9 = [r["task2_unseen_9"]["signature"]["mae"] for r in runs]
        unknot_mae_9 = [r["task2_unseen_9"]["unknotting_number"]["mae"] for r in runs]

        m_seen = np.mean(seen_auroc)
        se_seen = np.std(seen_auroc) / np.sqrt(n_r)
        m_u9 = np.mean(unseen9_auroc)
        se_u9 = np.std(unseen9_auroc) / np.sqrt(n_r)
        m_sig = np.mean(sig_mae_9)
        se_sig = np.std(sig_mae_9) / np.sqrt(n_r)
        m_unk = np.mean(unknot_mae_9)
        se_unk = np.std(unknot_mae_9) / np.sqrt(n_r)

        summary_report["models"][m_key] = {
            "label": m_label,
            "task1_seen_a_auroc": {"mean": float(m_seen), "stderr": float(se_seen)},
            "task1_unseen_9_auroc": {"mean": float(m_u9), "stderr": float(se_u9)},
            "task2_sig_mae_9": {"mean": float(m_sig), "stderr": float(se_sig)},
            "task2_unknot_mae_9": {"mean": float(m_unk), "stderr": float(se_unk)},
            "runs": runs,
        }

        print(f"{m_label:<35} | {m_seen:.3f} ± {se_seen:.3f}  | {m_u9:.3f} ± {se_u9:.3f}   | {m_sig:.3f} ± {se_sig:.3f}   | {m_unk:.3f} ± {se_unk:.3f}")

    print("-" * 95)
    print(f"Trivial Mean Baseline (c=9)          |      ---     |      ---       | {triv_9['signature']['mean_mae']:.3f}        | {triv_9['unknotting_number']['mean_mae']:.3f}")
    print(f"Trivial Linear c(K)+w(K) (c=9)       |      ---     |      ---       | {triv_9['signature']['linear_mae']:.3f}        | {triv_9['unknotting_number']['linear_mae']:.3f}")
    print("=" * 80)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary_report, f, indent=2)
        print(f"\n[Results] Full results written to {output_json}")

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reidemeister Canonicalization Benchmark")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per model")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--cache", type=str, default="data/benchmark_diagrams_cache.json", help="Diagram cache path")
    parser.add_argument("--output", type=str, default="results/honest_reidemeister_benchmark.json", help="Output JSON results")
    args = parser.parse_args()

    run_experiment_suite(
        seeds=args.seeds,
        epochs=args.epochs,
        dataset_cache=Path(args.cache),
        output_json=Path(args.output),
    )
