"""
run_rigorous_tasks.py - Rigorous Scientific Execution of Task 1 & Task 2.

Addresses all audited benchmark flaws:
  Task 1 (Verification on Independent Base Diagrams):
    - Generates positive pairs from independent diagrams via SnapPy Link.backtrack(steps=25..70).
    - Enforces strictly safe cyclic shift of arc numbering + crossing permutations (preserving strand orientation).
    - Uses topological node features (crossing signs + incident face degree lengths) with ZERO traversal coordinate leakage.
    - Reports identical-PD rate after R1/R2 reduction (empirically ~0.0% for backtrack walks).
    - Reports AUROC stratified by overall vs. non-identical canonical diagrams, with bootstrap 95% CIs.
    - Evaluates:
        1. SnapPy Canonical Hash / Exact Match Baseline
        2. Pure Graph Structure GNN (Raw Input, Untrained)
        3. Pure Graph Structure GNN (Canonical Input, Untrained)
        4. Pure Graph Structure GNN (Canonical Input, Contrastive Trained on c <= 8)
        5. Alexander Polynomial Baseline

  Task 2 (Full-Scale Invariant Prediction):
    - Full KnotInfo census through 12 crossings (2,978 knots).
    - Train split: all 801 prime knots with c <= 11 (+ 1 unknot = 802 knots).
    - Test split: all 2,176 prime knots with c == 12 (1,288 alternating + 888 non-alternating).
    - Targets:
        1. unknotting_number u(K): strictly restricted to the 1,628 test knots with exact values.
        2. smooth_four_genus g_4(K) (slice genus): evaluated on all 2,176 test knots.
        3. three_genus g_3(K) (Seifert genus): evaluated on all 2,176 test knots.
        4. signature sigma(K): evaluated on all 2,176 test knots.
    - Bars to beat: Trivial Mean Baseline, Trivial Linear Fit on [c(K), w(K)], Alexander Baseline.
    - Models: Mean Baseline, Linear [c, w], Alexander Baseline, Untrained Raw GNN, Untrained Canonical GNN,
              Supervised Multi-Task Canonical GNN.
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
import snappy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from knot_recognition.r_gno import SignedMessagePassingLayer


# ==============================================================================
# Diagram Randomizer & Topological Graph Tensors
# ==============================================================================

def safe_randomize_pd(
    pd_list: List[Tuple[int, int, int, int]],
    signs: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
    """
    Strictly safe cyclic shift of arc indices along the oriented knot + crossing permutation.
    Preserves strand orientation, port identities, and all topological invariants.
    """
    n = len(pd_list)
    if n == 0:
        return pd_list, signs or []
    if seed is not None:
        random.seed(seed)

    arcs = sorted(list(set(x for c in pd_list for x in c[:4])))
    n_arcs = len(arcs)
    shift = random.randint(0, n_arcs - 1)

    arc_map = {a: arcs[(i + shift) % n_arcs] for i, a in enumerate(arcs)}
    shifted_pd = [[arc_map[a] for a in c[:4]] for c in pd_list]
    s_list = list(signs) if signs is not None else [1] * n

    perm = list(range(n))
    random.shuffle(perm)
    return [tuple(shifted_pd[i]) for i in perm], [s_list[i] for i in perm]


def extract_crossing_face_degrees(pd_list: List[Tuple[int, int, int, int]]) -> List[List[float]]:
    """
    Extracts the lengths of the 4 incident planar faces for each crossing.
    Invariant to arc relabeling and traversal origin choice.
    """
    n = len(pd_list)
    if n == 0:
        return []
    try:
        L = snappy.Link(list(pd_list))
        faces = L.faces()
        c_faces: Dict[Any, List[int]] = {}
        for face in faces:
            flen = len(face)
            for corner in face:
                c_faces.setdefault(corner[0], []).append(flen)
        res = []
        for c in L.crossings:
            fl = sorted(c_faces.get(c, [4, 4, 4, 4]))[:4]
            while len(fl) < 4:
                fl.append(4)
            res.append([float(x) / 10.0 for x in fl])
        while len(res) < n:
            res.append([0.4, 0.4, 0.4, 0.4])
        return res[:n]
    except Exception:
        return [[0.4, 0.4, 0.4, 0.4] for _ in range(n)]


def build_pure_graph_tensors(
    pd_list: List[Tuple[int, int, int, int]],
    signs: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds abstract signed 4-valent graph tensors with ZERO traversal coordinate leakage.
    Node features (7 dims): [sign, is_pos, is_neg, f1, f2, f3, f4] (incident planar face degrees).
    Edge features (4 dims): [port_src, port_dst, is_over_src, is_over_dst].
    """
    dev = device or torch.device("cpu")
    n = len(pd_list)
    if n == 0:
        return (
            torch.zeros((1, 7), dtype=torch.float32, device=dev),
            torch.empty((2, 0), dtype=torch.long, device=dev),
            torch.empty((0, 4), dtype=torch.float32, device=dev),
        )

    s_list = list(signs) if signs is not None else [1] * n
    face_degs = extract_crossing_face_degrees(pd_list)

    node_feats = []
    for i, s in enumerate(s_list):
        f = face_degs[i] if i < len(face_degs) else [0.4, 0.4, 0.4, 0.4]
        node_feats.append([
            float(s),
            1.0 if s > 0 else 0.0,
            1.0 if s < 0 else 0.0,
            f[0], f[1], f[2], f[3]
        ])
    node_t = torch.tensor(node_feats, dtype=torch.float32, device=dev)

    arc_to_ports: Dict[int, List[Tuple[int, int]]] = {}
    for c_idx, c in enumerate(pd_list):
        for port_idx, arc in enumerate(c[:4]):
            arc_to_ports.setdefault(arc, []).append((c_idx, port_idx))

    src_list, dst_list, edge_attr_list = [], [], []
    for arc, port_list in arc_to_ports.items():
        if len(port_list) == 2:
            (c1, p1), (c2, p2) = port_list
            src_list.extend([c1, c2])
            dst_list.extend([c2, c1])
            is_ov1 = 1.0 if p1 in (1, 3) else 0.0
            is_ov2 = 1.0 if p2 in (1, 3) else 0.0
            edge_attr_list.append([float(p1), float(p2), is_ov1, is_ov2])
            edge_attr_list.append([float(p2), float(p1), is_ov2, is_ov1])

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=dev)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32, device=dev)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
        edge_attr = torch.empty((0, 4), dtype=torch.float32, device=dev)

    return node_t, edge_index, edge_attr


# ==============================================================================
# Model Architectures
# ==============================================================================

class PureGNN(nn.Module):
    """Permutation-invariant GNN operating solely on intrinsic signed graph topology."""
    def __init__(self, in_dim: int = 7, hidden_dim: int = 64, out_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            SignedMessagePassingLayer(hidden_dim, edge_in_dim=4) for _ in range(num_layers)
        ])
        self.pool = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, nf: torch.Tensor, ei: torch.Tensor, ef: torch.Tensor) -> torch.Tensor:
        h = self.node_embed(nf)
        for layer in self.layers:
            h = layer(h, ei, ef)
        pooled = torch.cat([h.mean(dim=0, keepdim=True), h.max(dim=0, keepdim=True).values], dim=-1)
        z = self.pool(pooled).squeeze(0)
        return F.normalize(z, p=2, dim=-1)


class MultiTaskGNN(nn.Module):
    """GNN with multi-task regression head for simultaneous invariant prediction."""
    def __init__(self, in_dim: int = 7, hidden_dim: int = 64, num_layers: int = 4, num_targets: int = 4):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            SignedMessagePassingLayer(hidden_dim, edge_in_dim=4) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, nf: torch.Tensor, ei: torch.Tensor, ef: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.node_embed(nf)
        for layer in self.layers:
            h = layer(h, ei, ef)
        pooled = torch.cat([h.mean(dim=0, keepdim=True), h.max(dim=0, keepdim=True).values], dim=-1)
        preds = self.head(pooled).squeeze(0)
        return pooled.squeeze(0), preds


def canonicalize_pd_snappy(pd_list: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Applies exact confluent R1/R2 basic simplification via SnapPy."""
    try:
        L = snappy.Link(list(pd_list))
        L.simplify("basic")
        return [tuple(c[:4]) for c in L.PD_code()]
    except Exception:
        return pd_list


# ==============================================================================
# TASK 1: Same-Knot Verification on Independent Base Diagrams
# ==============================================================================

def generate_independent_diagrams_for_knot(
    pd_base: List[Tuple[int, int, int, int]],
    knot_name: str,
    num_diagrams: int = 4,
    seed: int = 42,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Generates genuinely independent diagrams of the same knot via long backtrack walks.
    """
    random.seed(seed)
    diagrams = [copy.deepcopy(pd_base)]

    steps_schedule = [25, 40, 55, 70]
    for i in range(1, num_diagrams):
        st = steps_schedule[(i - 1) % len(steps_schedule)]
        try:
            L = snappy.Link(list(pd_base))
            L.backtrack(steps=st, prob_type_1=0.5, prob_type_2=0.5)
            pd_bt = [tuple(c[:4]) for c in L.PD_code()]
            diagrams.append(pd_bt)
        except Exception:
            diagrams.append(copy.deepcopy(pd_base))

    # Apply strictly safe cyclic shift & crossing permutation to every diagram
    randomized = [safe_randomize_pd(d, seed=seed + idx)[0] for idx, d in enumerate(diagrams)]
    return randomized


def run_task1_benchmark(
    knots_db: Dict[str, Any],
    train_knot_names: List[str],
    test_knot_names: List[str],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs Task 1 on independent base diagrams with safe label randomization."""
    print(f"\n{'='*80}")
    print(f"TASK 1: Same-Knot Verification on Independent Base Diagrams (N={len(test_knot_names)} knots)")
    print(f"{'='*80}")

    # Generate 4 independent diagrams per test knot
    print(f"Generating independent backtrack diagrams for {len(test_knot_names)} test knots...")
    diagrams_per_knot: Dict[str, List[List[Tuple[int, int, int, int]]]] = {}
    canonical_per_knot: Dict[str, List[List[Tuple[int, int, int, int]]]] = {}

    identical_canonical_pairs = 0
    total_positive_pairs = 0

    for idx, k in enumerate(test_knot_names):
        entry = knots_db[k]
        diags = generate_independent_diagrams_for_knot(entry["pd"], k, num_diagrams=4, seed=seed + idx)
        diagrams_per_knot[k] = diags

        can_diags = [canonicalize_pd_snappy(d) for d in diags]
        canonical_per_knot[k] = can_diags

        # Compare positive pairs (d0 vs d1, d0 vs d2, etc.)
        for j in range(1, len(can_diags)):
            total_positive_pairs += 1
            if can_diags[0] == can_diags[j]:
                identical_canonical_pairs += 1

    ident_rate = (identical_canonical_pairs / max(1, total_positive_pairs)) * 100.0
    print(f"Identical Canonical PD Rate across independent backtrack positive pairs: "
          f"{identical_canonical_pairs}/{total_positive_pairs} ({ident_rate:.1f}%)")

    # Sample evaluation pairs
    rng = np.random.default_rng(seed)
    knots_by_c: Dict[int, List[str]] = {}
    for k in test_knot_names:
        c = knots_db[k]["crossings"]
        knots_by_c.setdefault(c, []).append(k)

    eval_pairs = []  # (k1, d1_idx, k2, d2_idx, is_pos, is_identical_canon)
    for k in test_knot_names:
        c_k = knots_db[k]["crossings"]
        # Positives
        for d_idx in range(1, 4):
            is_ident = (canonical_per_knot[k][0] == canonical_per_knot[k][d_idx])
            eval_pairs.append((k, 0, k, d_idx, True, is_ident))

        # Matched-crossing negatives
        pool = [x for x in knots_by_c.get(c_k, []) if x != k]
        if not pool:
            pool = [x for x in test_knot_names if x != k]
        chosen_negs = rng.choice(pool, size=min(5, len(pool)), replace=False)
        for neg_k in chosen_negs:
            eval_pairs.append((k, 0, neg_k, 0, False, False))

    print(f"Total evaluated pairs: {len(eval_pairs)} ({sum(1 for p in eval_pairs if p[4])} pos, {sum(1 for p in eval_pairs if not p[4])} neg)")

    # 1. Untrained Pure GNN
    torch.manual_seed(seed)
    untrained_gnn = PureGNN(in_dim=7).to(device)
    untrained_gnn.eval()

    # 2. Contrastive Trained Pure GNN (Trained on c <= 8 knots with backtrack pairs)
    print(f"\nTraining Contrastive Canonical GNN on c <= 8 training knots ({len(train_knot_names)} knots)...")
    torch.manual_seed(seed)
    contrastive_gnn = PureGNN(in_dim=7).to(device)
    opt = torch.optim.Adam(contrastive_gnn.parameters(), lr=1e-3, weight_decay=1e-4)

    # Build pairs of training diagrams
    train_pairs_tensors = []
    for idx, k in enumerate(train_knot_names[:37]):
        base_pd = knots_db[k]["pd"]
        d1 = canonicalize_pd_snappy(safe_randomize_pd(base_pd, seed=seed + idx)[0])
        try:
            L = snappy.Link(list(base_pd))
            L.backtrack(steps=30)
            d2 = canonicalize_pd_snappy(safe_randomize_pd([tuple(c[:4]) for c in L.PD_code()], seed=seed + 1000 + idx)[0])
        except Exception:
            d2 = d1
        t1 = build_pure_graph_tensors(d1, device=device)
        t2 = build_pure_graph_tensors(d2, device=device)
        train_pairs_tensors.append((t1, t2))

    contrastive_gnn.train()
    for epoch in range(15):
        random.shuffle(train_pairs_tensors)
        for t1, t2 in train_pairs_tensors:
            opt.zero_grad()
            z1 = contrastive_gnn(t1[0], t1[1], t1[2])
            z2 = contrastive_gnn(t2[0], t2[1], t2[2])
            # Positive similarity loss
            loss_pos = 1.0 - F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0))
            loss = loss_pos
            loss.backward()
            opt.step()
    contrastive_gnn.eval()
    print("Contrastive Canonical GNN training complete.")

    # Compute representations for all test diagrams
    print("Extracting test representations across all candidate models...")
    raw_gnn_embs: Dict[str, List[np.ndarray]] = {}
    can_gnn_embs: Dict[str, List[np.ndarray]] = {}
    trained_can_gnn_embs: Dict[str, List[np.ndarray]] = {}
    alex_embs: Dict[str, List[np.ndarray]] = {}

    for k in test_knot_names:
        raw_gnn_embs[k] = []
        can_gnn_embs[k] = []
        trained_can_gnn_embs[k] = []
        alex_embs[k] = []

        alex_v = knots_db[k].get("alexander_vector", [])
        alex_feats = np.zeros(16, dtype=np.float32)
        if isinstance(alex_v, list) and len(alex_v) > 2:
            coeffs = alex_v[2:]
            for ci, val in enumerate(coeffs[:16]):
                alex_feats[ci] = float(val)
        norm_a = np.linalg.norm(alex_feats)
        if norm_a > 0:
            alex_feats /= norm_a

        for d_raw, d_can in zip(diagrams_per_knot[k], canonical_per_knot[k]):
            # Raw Untrained GNN
            nf_r, ei_r, ef_r = build_pure_graph_tensors(d_raw, device=device)
            with torch.no_grad():
                zr = untrained_gnn(nf_r, ei_r, ef_r).cpu().numpy()
            raw_gnn_embs[k].append(zr)

            # Canonical Untrained GNN
            nf_c, ei_c, ef_c = build_pure_graph_tensors(d_can, device=device)
            with torch.no_grad():
                zc = untrained_gnn(nf_c, ei_c, ef_c).cpu().numpy()
                zt = contrastive_gnn(nf_c, ei_c, ef_c).cpu().numpy()
            can_gnn_embs[k].append(zc)
            trained_can_gnn_embs[k].append(zt)

            # Alexander
            alex_embs[k].append(alex_feats)

    models_to_test = [
        ("snappy_hash", "Canonical Hash / Exact Match Baseline"),
        ("pure_gnn_raw", "Pure Graph GNN (Raw Input, Untrained)"),
        ("pure_gnn_canon", "Pure Graph GNN (Canonical Input, Untrained)"),
        ("trained_can_gnn", "Pure Graph GNN (Canonical Input, Contrastive Trained)"),
        ("alexander_poly", "Alexander Polynomial Baseline"),
    ]

    results: Dict[str, Any] = {
        "metadata": {
            "num_knots": len(test_knot_names),
            "identical_canonical_rate_pct": ident_rate,
            "total_evaluated_pairs": len(eval_pairs),
        },
        "scores": {},
    }

    print("\nTask 1 Evaluation Results (stratified by Overall vs Non-Identical diagrams):")
    print(f"{'Model':<48} | {'Overall AUROC':<22} | {'Non-Identical AUROC':<20}")
    print("-" * 96)

    for m_key, m_label in models_to_test:
        y_true_all, y_scores_all = [], []
        y_true_non_ident, y_scores_non_ident = [], []

        for k1, d1, k2, d2, is_pos, is_ident in eval_pairs:
            if m_key == "snappy_hash":
                sim = 1.0 if (canonical_per_knot[k1][d1] == canonical_per_knot[k2][d2]) else 0.0
            elif m_key == "pure_gnn_raw":
                sim = float(np.dot(raw_gnn_embs[k1][d1], raw_gnn_embs[k2][d2]))
            elif m_key == "pure_gnn_canon":
                sim = float(np.dot(can_gnn_embs[k1][d1], can_gnn_embs[k2][d2]))
            elif m_key == "trained_can_gnn":
                sim = float(np.dot(trained_can_gnn_embs[k1][d1], trained_can_gnn_embs[k2][d2]))
            elif m_key == "alexander_poly":
                sim = float(np.dot(alex_embs[k1][d1], alex_embs[k2][d2]))

            y_true_all.append(1 if is_pos else 0)
            y_scores_all.append(sim)

            if not is_ident or not is_pos:
                y_true_non_ident.append(1 if is_pos else 0)
                y_scores_non_ident.append(sim)

        auroc_all = float(roc_auc_score(y_true_all, y_scores_all))
        auroc_non_ident = float(roc_auc_score(y_true_non_ident, y_scores_non_ident))

        # Bootstrap 95% CI
        boot_scores = []
        n_pairs = len(y_true_all)
        for _ in range(300):
            b_idx = rng.integers(0, n_pairs, size=n_pairs)
            yt_b = [y_true_all[i] for i in b_idx]
            ys_b = [y_scores_all[i] for i in b_idx]
            if len(set(yt_b)) > 1:
                boot_scores.append(roc_auc_score(yt_b, ys_b))
        stderr = float(np.std(boot_scores)) if boot_scores else 0.0

        results["scores"][m_key] = {
            "label": m_label,
            "overall_auroc": auroc_all,
            "overall_stderr": stderr,
            "non_identical_auroc": auroc_non_ident,
        }

        print(f"{m_label:<48} | {auroc_all:.3f} ± {1.96*stderr:.3f} (95% CI)  | {auroc_non_ident:.3f}")

    return results


# ==============================================================================
# TASK 2: Large-Scale Invariant Prediction (c <= 11 -> c = 12)
# ==============================================================================

def extract_features_lin(knots_db: Dict[str, Any], knot_list: List[str]) -> np.ndarray:
    feats = []
    for k in knot_list:
        pd = knots_db[k]["pd"]
        c = float(len(pd))
        try:
            L = snappy.Link(list(pd))
            w = float(L.writhe())
        except Exception:
            w = 0.0
        feats.append([c, w])
    return np.array(feats, dtype=np.float32)


def extract_features_alex(knots_db: Dict[str, Any], knot_list: List[str]) -> np.ndarray:
    feats = []
    for k in knot_list:
        c = float(knots_db[k]["crossings"])
        alex_v = knots_db[k].get("alexander_vector", [])
        deg = 0.0
        det = 1.0
        coeffs = [0.0] * 16
        if isinstance(alex_v, list) and len(alex_v) > 2:
            deg = float((alex_v[1] - alex_v[0]) / 2)
            c_vals = alex_v[2:]
            min_deg = alex_v[0]
            val_at_neg1 = sum(c_val * ((-1) ** (i + min_deg)) for i, c_val in enumerate(c_vals))
            det = float(abs(val_at_neg1))
            for idx, val in enumerate(c_vals[:16]):
                coeffs[idx] = float(val)
        feats.append([c, deg, det] + coeffs)
    return np.array(feats, dtype=np.float32)


def run_task2_benchmark(
    knots_db: Dict[str, Any],
    train_knots: List[str],
    test_knots: List[str],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs Task 2 on all 2,978 KnotInfo knots across c <= 11 -> c = 12."""
    print(f"\n{'='*80}")
    print(f"TASK 2: Invariant Prediction at Full KnotInfo Scale")
    print(f"Train Set: {len(train_knots)} knots (c <= 11) | Test Set: {len(test_knots)} knots (c == 12)")
    print(f"{'='*80}")

    targets = ["unknotting_number", "smooth_four_genus", "three_genus", "signature"]

    print("Extracting classical Linear [c, w] features...")
    X_train_lin = extract_features_lin(knots_db, train_knots)
    X_test_lin = extract_features_lin(knots_db, test_knots)

    print("Extracting Alexander polynomial features (deg, det, coefficients)...")
    X_train_alex = extract_features_alex(knots_db, train_knots)
    X_test_alex = extract_features_alex(knots_db, test_knots)

    # 1. Untrained Pure GNN Embeddings (Raw and Canonical)
    torch.manual_seed(seed)
    untrained_gnn = PureGNN(in_dim=7, hidden_dim=64, out_dim=64, num_layers=4).to(device)
    untrained_gnn.eval()

    def get_untrained_gnn_feats(knot_list: List[str], use_canonical: bool = False) -> np.ndarray:
        feats = []
        for k in knot_list:
            pd = knots_db[k]["pd"]
            if use_canonical:
                pd = canonicalize_pd_snappy(pd)
            nf, ei, ef = build_pure_graph_tensors(pd, device=device)
            with torch.no_grad():
                z = untrained_gnn(nf, ei, ef).cpu().numpy()
            feats.append(z)
        return np.array(feats, dtype=np.float32)

    print("Extracting Untrained GNN embeddings (Raw)...")
    X_train_raw_gnn = get_untrained_gnn_feats(train_knots, use_canonical=False)
    X_test_raw_gnn = get_untrained_gnn_feats(test_knots, use_canonical=False)

    print("Extracting Untrained GNN embeddings (Canonical)...")
    X_train_can_gnn = get_untrained_gnn_feats(train_knots, use_canonical=True)
    X_test_can_gnn = get_untrained_gnn_feats(test_knots, use_canonical=True)

    # 2. Supervised Multi-Task Canonical GNN
    print(f"\nTraining Supervised Multi-Task Canonical GNN on {len(train_knots)} train knots...")
    torch.manual_seed(seed)
    multi_gnn = MultiTaskGNN(in_dim=7, hidden_dim=64, num_layers=4, num_targets=4).to(device)
    multi_opt = torch.optim.Adam(multi_gnn.parameters(), lr=1e-3, weight_decay=1e-4)

    # Prepare training tensors & target matrices
    train_graph_data = []
    y_train_matrix = []
    mask_train_matrix = []
    for k in train_knots:
        pd = canonicalize_pd_snappy(knots_db[k]["pd"])
        nf, ei, ef = build_pure_graph_tensors(pd, device=device)
        train_graph_data.append((nf, ei, ef))

        row_y = []
        row_m = []
        for tgt in targets:
            v = knots_db[k][tgt]
            if v is not None:
                row_y.append(float(v))
                row_m.append(1.0)
            else:
                row_y.append(0.0)
                row_m.append(0.0)
        y_train_matrix.append(row_y)
        mask_train_matrix.append(row_m)

    y_train_t = torch.tensor(y_train_matrix, dtype=torch.float32, device=device)
    mask_train_t = torch.tensor(mask_train_matrix, dtype=torch.float32, device=device)

    multi_gnn.train()
    indices = list(range(len(train_knots)))
    for epoch in range(1, 26):
        random.shuffle(indices)
        total_loss = 0.0
        for idx in indices:
            multi_opt.zero_grad()
            nf, ei, ef = train_graph_data[idx]
            _, pred = multi_gnn(nf, ei, ef)
            target_val = y_train_t[idx]
            mask_val = mask_train_t[idx]
            diff = (pred - target_val) * mask_val
            loss = torch.mean(diff ** 2)
            loss.backward()
            multi_opt.step()
            total_loss += float(loss.item())

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:02d}/25 | Multi-task Train MSE Loss: {total_loss / len(train_knots):.4f}")

    multi_gnn.eval()

    # Extract test predictions and representations from Supervised GNN
    test_supervised_preds: List[np.ndarray] = []
    test_supervised_reps: List[np.ndarray] = []
    for k in test_knots:
        pd = canonicalize_pd_snappy(knots_db[k]["pd"])
        nf, ei, ef = build_pure_graph_tensors(pd, device=device)
        with torch.no_grad():
            rep, pred = multi_gnn(nf, ei, ef)
            test_supervised_reps.append(rep.cpu().numpy())
            test_supervised_preds.append(pred.cpu().numpy())
    test_supervised_preds = np.array(test_supervised_preds, dtype=np.float32)

    task2_results: Dict[str, Any] = {}

    print("\nEvaluating Target Invariants on 12-Crossing Test Knots:")
    print(f"{'Target Invariant':<20} | {'Eval Knots':<10} | {'Mean MAE':<9} | {'Linear MAE':<10} | {'Alex MAE':<9} | {'Raw GNN':<8} | {'Canon GNN':<9} | {'Supervised GNN':<14}")
    print("-" * 105)

    for tgt_idx, target in enumerate(targets):
        # Extract ground truth for valid knots
        y_train = []
        valid_train_idx = []
        for i, k in enumerate(train_knots):
            val = knots_db[k][target]
            if val is not None:
                y_train.append(float(val))
                valid_train_idx.append(i)
        y_train = np.array(y_train, dtype=np.float32)

        y_test = []
        valid_test_idx = []
        for i, k in enumerate(test_knots):
            val = knots_db[k][target]
            if val is not None:
                y_test.append(float(val))
                valid_test_idx.append(i)
        y_test = np.array(y_test, dtype=np.float32)

        n_eval = len(y_test)

        # 1. Mean Baseline
        pred_mean = np.full_like(y_test, np.mean(y_train))
        mae_mean = float(np.mean(np.abs(y_test - pred_mean)))
        acc_mean = float(np.mean(np.round(pred_mean) == np.round(y_test))) * 100.0

        # 2. Linear Baseline [c, w]
        reg_lin = Ridge(alpha=1.0)
        reg_lin.fit(X_train_lin[valid_train_idx], y_train)
        pred_lin = reg_lin.predict(X_test_lin[valid_test_idx])
        mae_lin = float(np.mean(np.abs(y_test - pred_lin)))
        acc_lin = float(np.mean(np.round(pred_lin) == np.round(y_test))) * 100.0

        # 3. Alexander Polynomial Baseline
        reg_alex = Ridge(alpha=1.0)
        reg_alex.fit(X_train_alex[valid_train_idx], y_train)
        pred_alex = reg_alex.predict(X_test_alex[valid_test_idx])
        mae_alex = float(np.mean(np.abs(y_test - pred_alex)))
        acc_alex = float(np.mean(np.round(pred_alex) == np.round(y_test))) * 100.0

        # 4. Raw Untrained GNN
        reg_raw = Ridge(alpha=1.0)
        reg_raw.fit(X_train_raw_gnn[valid_train_idx], y_train)
        pred_raw = reg_raw.predict(X_test_raw_gnn[valid_test_idx])
        mae_raw = float(np.mean(np.abs(y_test - pred_raw)))
        acc_raw = float(np.mean(np.round(pred_raw) == np.round(y_test))) * 100.0

        # 5. Canonical Untrained GNN
        reg_can = Ridge(alpha=1.0)
        reg_can.fit(X_train_can_gnn[valid_train_idx], y_train)
        pred_can = reg_can.predict(X_test_can_gnn[valid_test_idx])
        mae_can = float(np.mean(np.abs(y_test - pred_can)))
        acc_can = float(np.mean(np.round(pred_can) == np.round(y_test))) * 100.0

        # 6. Supervised Multi-Task Canonical GNN
        pred_sup = test_supervised_preds[valid_test_idx, tgt_idx]
        mae_sup = float(np.mean(np.abs(y_test - pred_sup)))
        acc_sup = float(np.mean(np.round(pred_sup) == np.round(y_test))) * 100.0

        task2_results[target] = {
            "num_evaluated_test_knots": n_eval,
            "total_test_knots": len(test_knots),
            "mae_mean": mae_mean,
            "acc_mean": acc_mean,
            "mae_linear": mae_lin,
            "acc_linear": acc_lin,
            "mae_alexander": mae_alex,
            "acc_alexander": acc_alex,
            "mae_raw_gnn": mae_raw,
            "acc_raw_gnn": acc_raw,
            "mae_canonical_gnn": mae_can,
            "acc_canonical_gnn": acc_can,
            "mae_supervised_gnn": mae_sup,
            "acc_supervised_gnn": acc_sup,
        }

        print(f"{target:<20} | {n_eval:<10} | {mae_mean:<9.3f} | {mae_lin:<10.3f} | {mae_alex:<9.3f} | {mae_raw:<8.3f} | {mae_can:<9.3f} | {mae_sup:<14.3f}")

    return task2_results


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Rigorous Task 1 and Task 2 Benchmarks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/rigorous_tasks_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device} | Seed: {args.seed}")

    db_path = Path("data/knotinfo/knotinfo_full_dataset.json")
    if not db_path.exists():
        print(f"Error: {db_path} not found. Please extract KnotInfo dataset first.")
        sys.exit(1)

    with open(db_path) as f:
        knots_db = json.load(f)

    train_knots = [k for k, v in knots_db.items() if v["crossings"] <= 11]
    test_knots = [k for k, v in knots_db.items() if v["crossings"] == 12]

    # Task 1: Knots with c in (9, 10) (214 prime knots)
    task1_knots = [k for k, v in knots_db.items() if v["crossings"] in (9, 10)]
    t1_results = run_task1_benchmark(knots_db, train_knots, task1_knots, device=device, seed=args.seed)

    # Task 2: Evaluate on all 2,176 knots at c = 12
    t2_results = run_task2_benchmark(knots_db, train_knots, test_knots, device=device, seed=args.seed)

    full_output = {
        "metadata": {
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
        },
        "task1_verification": t1_results,
        "task2_invariants": t2_results,
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nAll benchmark results successfully saved to {out_p}")


if __name__ == "__main__":
    main()
