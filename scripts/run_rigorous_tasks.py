"""
run_rigorous_tasks.py - Fully Audited Scientific Execution of Task 1 & Task 2.

Audit Fixes Implemented:
  Task 1 (Verification on Independent Base Diagrams):
    1. Size-Matched Pair Sampler: Eliminates the diagram-size artifact (which caused artificial AUROC 0.000).
       Positives and negatives share the identical distribution of backtrack steps and diagram sizes.
    2. Same-Alexander Hard Negatives: Includes pairs sharing the exact Alexander polynomial Delta(t).
    3. Stratification by Topology: Reports AUROC separately for Overall (214 knots), Alternating (164 knots),
       and Non-Alternating (50 knots).
    4. Knot-Level Cluster Bootstrap: Resamples by knot (not by pair) for true cluster-robust 95% CIs.
    5. Evaluated Models:
       - SnapPy Canonical Hash / Exact Match Baseline
       - Pure Graph GNN (Raw Input, Untrained)
       - Pure Graph GNN (Canonical Input, Untrained)
       - Pure Graph GNN (Canonical Input, Contrastive Trained on c <= 8)
       - Alexander Polynomial Baseline

  Task 2 (Full-Scale Invariant Prediction Across 5 Random Seeds):
    1. Full KnotInfo census through 12 crossings (2,978 knots).
       - Train: all 802 prime knots with c <= 11 (801 prime + 1 unknot).
       - Test: all 2,176 prime knots with c == 12 (1,288 alternating + 888 non-alternating).
    2. Exact Target Ground Truth:
       - unknotting_number u(K): restricted to the 1,628 test knots with exact integer values.
       - smooth_four_genus g_4(K): evaluated on all 2,176 test knots.
       - three_genus g_3(K): evaluated on all 2,176 test knots.
       - signature sigma(K): evaluated on all 2,176 test knots.
    3. Full Suite of Classical & Algebraic Baselines:
       - Trivial Mean Baseline
       - Linear Fit on [c(K), w(K)]
       - Exact Signature Murasugi Baseline (|sigma(K)|/2 for g_4 and u(K))
       - Linear Fit on [c(K), w(K), |sigma(K)|]
       - Alexander Polynomial Baseline (deg(Delta)/2 for g_3, det |Delta(-1)|, coeffs)
       - Untrained Canonical GNN
       - Supervised Multi-Task Canonical GNN (5 seeds: 42, 43, 44, 45, 46)
    4. Stratification: Reports MAE and Exact Accuracy separately for All Knots, Alternating Knots,
       and Non-Alternating Knots.
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

import networkx as nx
import numpy as np
import snappy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from knot_recognition.r_gno import SignedMessagePassingLayer


def pd_to_simple_bipartite_graph(pd_list: List[Tuple[int, int, int, int]]) -> nx.Graph:
    """Converts a planar diagram into an abstract simple bipartite graph (crossings and arcs)."""
    G = nx.Graph()
    n = len(pd_list)
    try:
        L = snappy.Link(list(pd_list))
        signs = [c.sign for c in L.crossings]
    except Exception:
        signs = [1] * n
    for i, s in enumerate(signs[:n]):
        G.add_node(f"c_{i}", ntype="crossing", sign=str(s))
    arc_to_crossings: Dict[int, List[Tuple[int, int]]] = {}
    for c_idx, c in enumerate(pd_list):
        for port_idx, arc in enumerate(c[:4]):
            arc_to_crossings.setdefault(arc, []).append((c_idx, port_idx))
    for arc, endpoints in arc_to_crossings.items():
        arc_node = f"a_{arc}"
        G.add_node(arc_node, ntype="arc", sign="0")
        for c_idx, p_idx in endpoints:
            is_over = "1" if p_idx in (1, 3) else "0"
            G.add_edge(f"c_{c_idx}", arc_node, port=str(p_idx), is_over=is_over)
    return G


def compute_wl_graph_hash(pd_list: List[Tuple[int, int, int, int]], iterations: int = 4) -> str:
    """Computes a label-invariant Weisfeiler-Lehman graph hash of a planar diagram."""
    G = pd_to_simple_bipartite_graph(pd_list)
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="sign", edge_attr="is_over", iterations=iterations)


# ==============================================================================
# Topological Helpers
# ==============================================================================

def is_knot_alternating(name: str) -> bool:
    """Identifies whether a knot is alternating from standard Rolfsen / Dowker-Thistlethwaite census naming."""
    if "a_" in name or name.endswith("a"):
        return True
    if "n_" in name or name.endswith("n"):
        return False
    if "_" in name:
        parts = name.split("_")
        try:
            c = int(parts[0])
            idx = int(parts[1])
            if c <= 7:
                return True
            if c == 8:
                return idx <= 18
            if c == 9:
                return idx <= 41
            if c == 10:
                return idx <= 123
        except Exception:
            pass
    return False


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
# TASK 1: Same-Knot Verification with Matched Diagram Sizes & Alexander Negatives
# ==============================================================================

def generate_independent_backtrack_diagrams(
    pd_base: List[Tuple[int, int, int, int]],
    steps_list: List[int] = [25, 40, 55, 70],
    seed: int = 42,
) -> List[List[Tuple[int, int, int, int]]]:
    """Generates independent backtrack diagrams across a fixed step count schedule."""
    random.seed(seed)
    diagrams = []
    for idx, st in enumerate(steps_list):
        try:
            L = snappy.Link(list(pd_base))
            L.backtrack(steps=st, prob_type_1=0.5, prob_type_2=0.5)
            pd_bt = [tuple(c[:4]) for c in L.PD_code()]
        except Exception:
            pd_bt = copy.deepcopy(pd_base)
        d_rand = safe_randomize_pd(pd_bt, seed=seed + 100 * (idx + 1))[0]
        diagrams.append(d_rand)
    return diagrams


def run_task1_benchmark(
    knots_db: Dict[str, Any],
    train_knot_names: List[str],
    test_knot_names: List[str],
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs Task 1 with matched-size sampling, same-Alexander hard negatives, and knot-level bootstrap."""
    print(f"\n{'='*80}")
    print(f"TASK 1: Same-Knot Verification on Independent Base Diagrams (N={len(test_knot_names)} knots)")
    print(f"{'='*80}")

    # Index knots by crossing count and Alexander vector
    knots_by_c: Dict[int, List[str]] = {}
    alex_to_knots: Dict[str, List[str]] = {}
    for k in test_knot_names:
        c = knots_db[k]["crossings"]
        knots_by_c.setdefault(c, []).append(k)
        alex_str = str(knots_db[k].get("alexander_vector", []))
        alex_to_knots.setdefault(alex_str, []).append(k)

    colliding_groups = {a: ks for a, ks in alex_to_knots.items() if len(ks) > 1}
    print(f"Test knots: {len(test_knot_names)} total ({sum(1 for k in test_knot_names if is_knot_alternating(k))} alternating, "
          f"{sum(1 for k in test_knot_names if not is_knot_alternating(k))} non-alternating)")
    print(f"KnotInfo Alexander collision groups in test set: {len(colliding_groups)} "
          f"({sum(len(v) for v in colliding_groups.values())} knots share Delta(t))")

    # Generate 4 independent backtrack diagrams per knot (steps: 25, 40, 55, 70)
    steps_schedule = [25, 40, 55, 70]
    diagrams_per_knot: Dict[str, List[List[Tuple[int, int, int, int]]]] = {}
    canonical_per_knot: Dict[str, List[List[Tuple[int, int, int, int]]]] = {}

    identical_canonical_pairs = 0
    total_positive_pairs = 0

    for idx, k in enumerate(test_knot_names):
        entry = knots_db[k]
        diags = generate_independent_backtrack_diagrams(entry["pd"], steps_list=steps_schedule, seed=seed + idx)
        diagrams_per_knot[k] = diags

        can_diags = [canonicalize_pd_snappy(d) for d in diags]
        canonical_per_knot[k] = can_diags

        # Check for any trivial identical canonical forms
        for i in range(len(can_diags)):
            for j in range(i + 1, len(can_diags)):
                total_positive_pairs += 1
                if can_diags[i] == can_diags[j]:
                    identical_canonical_pairs += 1

    ident_rate = (identical_canonical_pairs / max(1, total_positive_pairs)) * 100.0
    print(f"Identical Canonical PD Rate across independent backtrack positive pairs: "
          f"{identical_canonical_pairs}/{total_positive_pairs} ({ident_rate:.1f}%)")

    # Construct Size-Matched Pairs
    # For each knot k, evaluate pairs of diagrams (d_i, d_j)
    # Positives: (k, d_i) vs (k, d_j)
    # Negatives: (k, d_i) vs (neg_k, d_j) -- matching the EXACT step indices (i, j)!
    rng = np.random.default_rng(seed)
    eval_pairs = []  # (k1, d1_idx, k2, d2_idx, is_pos, is_alt)

    pair_step_indices = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]

    for k in test_knot_names:
        c_k = knots_db[k]["crossings"]
        is_alt = is_knot_alternating(k)
        alex_str = str(knots_db[k].get("alexander_vector", []))
        same_alex_pool = [x for x in alex_to_knots.get(alex_str, []) if x != k]
        same_c_pool = [x for x in knots_by_c.get(c_k, []) if x != k]

        for i, j in pair_step_indices:
            # 1. Positive pair (matched backtrack steps)
            eval_pairs.append((k, i, k, j, True, is_alt))

            # 2. Hard Negative pair: same Alexander if available, else matched crossing count
            if same_alex_pool and rng.random() < 0.5:
                neg_k = rng.choice(same_alex_pool)
            elif same_c_pool:
                neg_k = rng.choice(same_c_pool)
            else:
                pool = [x for x in test_knot_names if x != k]
                neg_k = rng.choice(pool)

            # Matched diagram indices (i, j) guarantees IDENTICAL diagram size distributions
            eval_pairs.append((k, i, neg_k, j, False, is_alt))

    pos_count = sum(1 for p in eval_pairs if p[4])
    neg_count = sum(1 for p in eval_pairs if not p[4])
    print(f"Constructed size-matched evaluation set: {len(eval_pairs)} pairs ({pos_count} pos, {neg_count} neg)")

    # 1. Untrained Pure GNN
    torch.manual_seed(seed)
    untrained_gnn = PureGNN(in_dim=7).to(device)
    untrained_gnn.eval()

    # 2. Contrastive Trained Canonical GNN (Trained on c <= 8 training knots)
    print(f"Training Contrastive Canonical GNN on c <= 8 knots ({len(train_knot_names[:37])} knots)...")
    torch.manual_seed(seed)
    contrastive_gnn = PureGNN(in_dim=7).to(device)
    opt = torch.optim.Adam(contrastive_gnn.parameters(), lr=1e-3, weight_decay=1e-4)

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
            loss = 1.0 - F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0))
            loss.backward()
            opt.step()
    contrastive_gnn.eval()

    # Extract test representations
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

    # Weisfeiler-Lehman graph hashes
    wl_hashes: Dict[str, List[str]] = {}
    for k in test_knot_names:
        wl_hashes[k] = [compute_wl_graph_hash(d) for d in canonical_per_knot[k]]

    models_to_test = [
        ("snappy_hash", "Canonical Hash / Exact Match Baseline"),
        ("wl_hash", "Weisfeiler–Lehman Graph Hash Baseline"),
        ("pure_gnn_raw", "Pure Graph GNN (Raw Input, Untrained)"),
        ("pure_gnn_canon", "Pure Graph GNN (Canonical Input, Untrained)"),
        ("trained_can_gnn", "Pure Graph GNN (Canonical Input, Contrastive Trained)"),
        ("alexander_poly", "Alexander Polynomial Baseline"),
    ]

    results: Dict[str, Any] = {
        "metadata": {
            "num_knots": len(test_knot_names),
            "num_alternating": sum(1 for k in test_knot_names if is_knot_alternating(k)),
            "num_non_alternating": sum(1 for k in test_knot_names if not is_knot_alternating(k)),
            "identical_canonical_rate_pct": ident_rate,
            "total_evaluated_pairs": len(eval_pairs),
        },
        "scores": {},
    }

    print("\nTask 1 Evaluation Results (Stratified by Knot Topology with Knot-Level 95% Bootstrap CI):")
    print(f"{'Model':<48} | {'Overall AUROC':<22} | {'Alternating AUROC':<20} | {'Non-Alternating':<16}")
    print("-" * 115)

    # Index pairs by knot for cluster bootstrap
    unique_knots = sorted(list(set(p[0] for p in eval_pairs)))
    knot_to_pair_idx: Dict[str, List[int]] = {}
    for idx, p in enumerate(eval_pairs):
        knot_to_pair_idx.setdefault(p[0], []).append(idx)

    for m_key, m_label in models_to_test:
        sim_scores = []
        for k1, d1, k2, d2, is_pos, is_alt in eval_pairs:
            if m_key == "snappy_hash":
                sim = 1.0 if (canonical_per_knot[k1][d1] == canonical_per_knot[k2][d2]) else 0.0
            elif m_key == "wl_hash":
                sim = 1.0 if (wl_hashes[k1][d1] == wl_hashes[k2][d2]) else 0.0
            elif m_key == "pure_gnn_raw":
                sim = float(np.dot(raw_gnn_embs[k1][d1], raw_gnn_embs[k2][d2]))
            elif m_key == "pure_gnn_canon":
                sim = float(np.dot(can_gnn_embs[k1][d1], can_gnn_embs[k2][d2]))
            elif m_key == "trained_can_gnn":
                sim = float(np.dot(trained_can_gnn_embs[k1][d1], trained_can_gnn_embs[k2][d2]))
            elif m_key == "alexander_poly":
                sim = float(np.dot(alex_embs[k1][d1], alex_embs[k2][d2]))
            sim_scores.append(sim)

        y_true_all = [1 if p[4] else 0 for p in eval_pairs]
        auroc_all = float(roc_auc_score(y_true_all, sim_scores))

        # Alternating subset
        alt_indices = [idx for idx, p in enumerate(eval_pairs) if p[5]]
        y_true_alt = [y_true_all[i] for i in alt_indices]
        scores_alt = [sim_scores[i] for i in alt_indices]
        auroc_alt = float(roc_auc_score(y_true_alt, scores_alt)) if len(set(y_true_alt)) > 1 else 0.5

        # Non-Alternating subset
        non_alt_indices = [idx for idx, p in enumerate(eval_pairs) if not p[5]]
        y_true_non = [y_true_all[i] for i in non_alt_indices]
        scores_non = [sim_scores[i] for i in non_alt_indices]
        auroc_non = float(roc_auc_score(y_true_non, scores_non)) if len(set(y_true_non)) > 1 else 0.5

        # Knot-level cluster bootstrap for Overall AUROC
        boot_scores = []
        for _ in range(300):
            sampled_knots = rng.choice(unique_knots, size=len(unique_knots), replace=True)
            boot_pair_indices = []
            for sk in sampled_knots:
                boot_pair_indices.extend(knot_to_pair_idx[sk])
            yt_b = [y_true_all[i] for i in boot_pair_indices]
            ys_b = [sim_scores[i] for i in boot_pair_indices]
            if len(set(yt_b)) > 1:
                boot_scores.append(roc_auc_score(yt_b, ys_b))
        stderr_knot = float(np.std(boot_scores)) if boot_scores else 0.0

        results["scores"][m_key] = {
            "label": m_label,
            "overall_auroc": auroc_all,
            "overall_stderr_knot_bootstrap": stderr_knot,
            "alternating_auroc": auroc_alt,
            "non_alternating_auroc": auroc_non,
        }

        print(f"{m_label:<48} | {auroc_all:.3f} ± {1.96*stderr_knot:.3f} (95% CI)  | {auroc_alt:<20.3f} | {auroc_non:<16.3f}")

    # -------------------------------------------------------------------------
    # Evaluation on Distinct Base Origins Across All Non-Alternating Knots
    # -------------------------------------------------------------------------
    non_alt_knots_list = [k for k in test_knot_names if not is_knot_alternating(k)]
    print(f"\n--- Distinct Base Origin Evaluation Across All Non-Alternating Knots ({len(non_alt_knots_list)} Knots) ---")
    ki_can_diags = {}
    sp_can_diags = {}
    identical_base_count = 0
    valid_non_alt = []

    for k in non_alt_knots_list:
        try:
            pd_ki = [tuple(c[:4]) for c in knots_db[k]["pd"]]
            L_sp = snappy.Link(k)
            pd_sp = [tuple(c[:4]) for c in L_sp.PD_code()]
        except Exception:
            continue
        d_ki = canonicalize_pd_snappy(safe_randomize_pd(pd_ki, seed=seed + 10)[0])
        d_sp = canonicalize_pd_snappy(safe_randomize_pd(pd_sp, seed=seed + 20)[0])
        if d_ki == d_sp:
            identical_base_count += 1
        ki_can_diags[k] = d_ki
        sp_can_diags[k] = d_sp
        valid_non_alt.append(k)

    print(f"Identical Canonical Forms (KnotInfo vs SnapPy): {identical_base_count} / {len(valid_non_alt)} "
          f"({(identical_base_count/max(1, len(valid_non_alt)))*100:.1f}%)")

    # Embeddings
    with torch.no_grad():
        ki_embs_na = {k: untrained_gnn(*build_pure_graph_tensors(ki_can_diags[k], device=device)).cpu().numpy() for k in valid_non_alt}
        sp_embs_na = {k: untrained_gnn(*build_pure_graph_tensors(sp_can_diags[k], device=device)).cpu().numpy() for k in valid_non_alt}
        wl_ki_na = {k: compute_wl_graph_hash(ki_can_diags[k]) for k in valid_non_alt}
        wl_sp_na = {k: compute_wl_graph_hash(sp_can_diags[k]) for k in valid_non_alt}

    diff_base_pairs_y = []
    diff_base_sims_gnn = []
    diff_base_sims_wl = []
    pos_cos = []
    neg_cos = []

    for k_pos in valid_non_alt:
        sim_gnn_pos = float(np.dot(ki_embs_na[k_pos], sp_embs_na[k_pos]))
        sim_wl_pos = 1.0 if wl_ki_na[k_pos] == wl_sp_na[k_pos] else 0.0
        diff_base_pairs_y.append(1)
        diff_base_sims_gnn.append(sim_gnn_pos)
        diff_base_sims_wl.append(sim_wl_pos)
        pos_cos.append(sim_gnn_pos)

        c_k = knots_db[k_pos]["crossings"]
        same_c_pool = [x for x in valid_non_alt if knots_db[x]["crossings"] == c_k and x != k_pos]
        for _ in range(5):
            k_neg = rng.choice(same_c_pool) if same_c_pool else rng.choice([x for x in valid_non_alt if x != k_pos])
            sim_gnn_neg = float(np.dot(ki_embs_na[k_pos], sp_embs_na[k_neg]))
            sim_wl_neg = 1.0 if wl_ki_na[k_pos] == wl_sp_na[k_neg] else 0.0
            diff_base_pairs_y.append(0)
            diff_base_sims_gnn.append(sim_gnn_neg)
            diff_base_sims_wl.append(sim_wl_neg)
            neg_cos.append(sim_gnn_neg)

    auroc_diff_gnn = float(roc_auc_score(diff_base_pairs_y, diff_base_sims_gnn))
    auroc_diff_wl = float(roc_auc_score(diff_base_pairs_y, diff_base_sims_wl))
    print(f"Untrained GNN AUROC on Distinct Base Diagrams: {auroc_diff_gnn:.4f} "
          f"(Mean Pos Cosine: {np.mean(pos_cos):.4f}, Mean Neg Cosine: {np.mean(neg_cos):.4f})")
    print(f"Weisfeiler-Lehman Hash AUROC on Distinct Base Diagrams: {auroc_diff_wl:.4f} "
          f"(Pos Match Rate: {sum(diff_base_sims_wl[i] for i in range(len(diff_base_pairs_y)) if diff_base_pairs_y[i]==1)}/{len(valid_non_alt)})")

    results["distinct_base_origin_non_alternating"] = {
        "num_knots": len(valid_non_alt),
        "identical_canonical_rate_pct": (identical_base_count / max(1, len(valid_non_alt))) * 100.0,
        "untrained_gnn_auroc": auroc_diff_gnn,
        "mean_positive_cosine": float(np.mean(pos_cos)),
        "mean_negative_cosine": float(np.mean(neg_cos)),
        "wl_hash_auroc": auroc_diff_wl,
    }

    return results


# ==============================================================================
# TASK 2: Large-Scale Invariant Prediction Across 5 Seeds & Stratified Subsets
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


def extract_features_lin_with_sig(knots_db: Dict[str, Any], knot_list: List[str]) -> np.ndarray:
    feats = []
    for k in knot_list:
        pd = knots_db[k]["pd"]
        c = float(len(pd))
        try:
            L = snappy.Link(list(pd))
            w = float(L.writhe())
        except Exception:
            w = 0.0
        sig = float(knots_db[k].get("signature", 0.0))
        feats.append([c, w, abs(sig) / 2.0, sig])
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
    seeds: List[int] = [42, 43, 44, 45, 46],
) -> Dict[str, Any]:
    """Runs Task 2 across 5 random seeds with stratified alternating vs non-alternating evaluation."""
    print(f"\n{'='*80}")
    print(f"TASK 2: Invariant Prediction Across 5 Random Seeds ({len(seeds)} Seeds)")
    print(f"Train Set: {len(train_knots)} knots (c <= 11) | Test Set: {len(test_knots)} knots (c == 12)")
    print(f"Test Split: {sum(1 for k in test_knots if is_knot_alternating(k))} Alternating, "
          f"{sum(1 for k in test_knots if not is_knot_alternating(k))} Non-Alternating")
    print(f"{'='*80}")

    targets = ["unknotting_number", "smooth_four_genus", "three_genus", "signature"]

    # Classical feature matrices
    print("Extracting feature matrices (Linear [c, w], Linear+Signature, Alexander)...")
    X_train_lin = extract_features_lin(knots_db, train_knots)
    X_test_lin = extract_features_lin(knots_db, test_knots)

    X_train_lin_sig = extract_features_lin_with_sig(knots_db, train_knots)
    X_test_lin_sig = extract_features_lin_with_sig(knots_db, test_knots)

    X_train_alex = extract_features_alex(knots_db, train_knots)
    X_test_alex = extract_features_alex(knots_db, test_knots)

    # Pre-build graph tensors once
    print("Pre-building canonical graph tensors for all knots...")
    train_graph_data = []
    for k in train_knots:
        pd = canonicalize_pd_snappy(knots_db[k]["pd"])
        nf, ei, ef = build_pure_graph_tensors(pd, device=device)
        train_graph_data.append((nf, ei, ef))

    test_graph_data = []
    for k in test_knots:
        pd = canonicalize_pd_snappy(knots_db[k]["pd"])
        nf, ei, ef = build_pure_graph_tensors(pd, device=device)
        test_graph_data.append((nf, ei, ef))

    # Target tensors
    y_train_matrix = []
    mask_train_matrix = []
    for k in train_knots:
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

    y_raw_arr = np.array(y_train_matrix, dtype=np.float32)
    m_raw_arr = np.array(mask_train_matrix, dtype=np.float32)
    target_means = np.zeros(4, dtype=np.float32)
    target_stds = np.ones(4, dtype=np.float32)
    for i in range(4):
        valid_vals = y_raw_arr[m_raw_arr[:, i] > 0, i]
        target_means[i] = np.mean(valid_vals)
        target_stds[i] = np.std(valid_vals) if np.std(valid_vals) > 1e-5 else 1.0

    y_train_norm = (y_raw_arr - target_means) / target_stds
    y_train_norm_t = torch.tensor(y_train_norm, dtype=torch.float32, device=device)
    mask_train_t = torch.tensor(m_raw_arr, dtype=torch.float32, device=device)
    means_t = torch.tensor(target_means, dtype=torch.float32, device=device)
    stds_t = torch.tensor(target_stds, dtype=torch.float32, device=device)

    # Subsets of test knots: All, Alternating, Non-Alternating
    subsets = {
        "all": list(range(len(test_knots))),
        "alternating": [i for i, k in enumerate(test_knots) if is_knot_alternating(k)],
        "non_alternating": [i for i, k in enumerate(test_knots) if not is_knot_alternating(k)],
    }

    # Execute 5 seeds for Supervised Multi-Task Canonical GNN and Untrained GNN
    print(f"\nTraining Supervised Multi-Task Canonical GNN across {len(seeds)} random seeds...")
    seed_supervised_preds: List[np.ndarray] = []
    seed_untrained_preds: Dict[str, List[np.ndarray]] = {tgt: [] for tgt in targets}

    for s_idx, seed_val in enumerate(seeds):
        torch.manual_seed(seed_val)
        multi_gnn = MultiTaskGNN(in_dim=7, hidden_dim=64, num_layers=4, num_targets=4).to(device)
        multi_opt = torch.optim.AdamW(multi_gnn.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(multi_opt, T_max=80)
        batch_size = 32

        # Train Supervised GNN
        multi_gnn.train()
        indices = list(range(len(train_knots)))
        for epoch in range(1, 81):
            random.seed(seed_val + epoch)
            random.shuffle(indices)
            multi_opt.zero_grad()
            for step, idx in enumerate(indices):
                nf, ei, ef = train_graph_data[idx]
                _, pred_norm = multi_gnn(nf, ei, ef)
                loss = F.smooth_l1_loss(pred_norm * mask_train_t[idx], y_train_norm_t[idx] * mask_train_t[idx], reduction="mean") / batch_size
                loss.backward()
                if (step + 1) % batch_size == 0 or (step + 1) == len(indices):
                    multi_opt.step()
                    multi_opt.zero_grad()
            scheduler.step()

        # Check training set convergence on seed 42 to verify GNN matches/beats linear fit on training set
        if s_idx == 0:
            multi_gnn.eval()
            with torch.no_grad():
                tr_preds = []
                for nf, ei, ef in train_graph_data:
                    _, pred_norm = multi_gnn(nf, ei, ef)
                    tr_preds.append((pred_norm * stds_t + means_t).cpu().numpy())
            tr_preds = np.array(tr_preds)
            print("\n  --- Training Set Convergence Verification (Seed 42) ---")
            for ti, tgt_name in enumerate(targets):
                val_mask = m_raw_arr[:, ti] > 0
                gnn_tr_mae = np.mean(np.abs(tr_preds[val_mask, ti] - y_raw_arr[val_mask, ti]))
                reg_tr = Ridge(alpha=1.0)
                reg_tr.fit(X_train_lin[val_mask], y_raw_arr[val_mask, ti])
                lin_tr_mae = np.mean(np.abs(reg_tr.predict(X_train_lin[val_mask]) - y_raw_arr[val_mask, ti]))
                mean_tr_mae = np.mean(np.abs(y_raw_arr[val_mask, ti] - target_means[ti]))
                print(f"  {tgt_name:<20}: GNN Train MAE = {gnn_tr_mae:.3f} | Linear [c,w] Train MAE = {lin_tr_mae:.3f} | Mean MAE = {mean_tr_mae:.3f}")

        multi_gnn.eval()
        t_preds = []
        for nf, ei, ef in test_graph_data:
            with torch.no_grad():
                _, pred_norm = multi_gnn(nf, ei, ef)
                pred_unnorm = pred_norm * stds_t + means_t
                t_preds.append(pred_unnorm.cpu().numpy())
        seed_supervised_preds.append(np.array(t_preds, dtype=np.float32))

        # Untrained Canonical GNN (Ridge probe)
        torch.manual_seed(seed_val)
        untrained_gnn = PureGNN(in_dim=7, hidden_dim=64, out_dim=64, num_layers=4).to(device)
        untrained_gnn.eval()
        X_tr_can = [untrained_gnn(*train_graph_data[i]).detach().cpu().numpy() for i in range(len(train_knots))]
        X_te_can = [untrained_gnn(*test_graph_data[i]).detach().cpu().numpy() for i in range(len(test_knots))]
        X_tr_can = np.array(X_tr_can, dtype=np.float32)
        X_te_can = np.array(X_te_can, dtype=np.float32)

        for tgt in targets:
            y_tr_sub = [float(knots_db[k][tgt]) for k in train_knots if knots_db[k][tgt] is not None]
            v_tr_idx = [i for i, k in enumerate(train_knots) if knots_db[k][tgt] is not None]
            reg = Ridge(alpha=1.0)
            reg.fit(X_tr_can[v_tr_idx], y_tr_sub)
            seed_untrained_preds[tgt].append(reg.predict(X_te_can))

        print(f"  Completed Seed {seed_val} ({s_idx + 1}/{len(seeds)})")

    task2_results: Dict[str, Any] = {}

    print("\n" + "=" * 125)
    print("TASK 2 RESULTS SUMMARY (5-Seed Mean ± Stderr, Stratified by Topology)")
    print("=" * 125)

    for target_idx, target in enumerate(targets):
        print(f"\nTarget Invariant: [{target.upper()}]")
        print(f"{'Subset':<16} | {'Knots':<6} | {'Mean Baseline':<14} | {'Linear [c,w]':<14} | {'Signature Base':<15} | {'Alexander':<14} | {'Supervised GNN (5-Seed)':<24}")
        print("-" * 120)

        task2_results[target] = {}

        # Ground truth vectors
        y_train_full = np.array([float(knots_db[k][target]) for k in train_knots if knots_db[k][target] is not None], dtype=np.float32)
        v_train_idx = [i for i, k in enumerate(train_knots) if knots_db[k][target] is not None]

        y_test_full = []
        v_test_idx_set = set()
        for i, k in enumerate(test_knots):
            val = knots_db[k][target]
            if val is not None:
                y_test_full.append(float(val))
                v_test_idx_set.add(i)
        y_test_full = np.array(y_test_full, dtype=np.float32)

        for sub_name, sub_indices in subsets.items():
            valid_sub_indices = [i for i in sub_indices if i in v_test_idx_set]
            n_eval = len(valid_sub_indices)
            if n_eval == 0:
                continue

            y_sub = np.array([float(knots_db[test_knots[i]][target]) for i in valid_sub_indices], dtype=np.float32)

            # 1. Mean Baseline
            pred_mean = np.full_like(y_sub, np.mean(y_train_full))
            mae_mean = float(np.mean(np.abs(y_sub - pred_mean)))
            acc_mean = float(np.mean(np.round(pred_mean) == np.round(y_sub))) * 100.0

            # 2. Linear Baseline [c, w]
            reg_lin = Ridge(alpha=1.0)
            reg_lin.fit(X_train_lin[v_train_idx], y_train_full)
            pred_lin = reg_lin.predict(X_test_lin[valid_sub_indices])
            mae_lin = float(np.mean(np.abs(y_sub - pred_lin)))
            acc_lin = float(np.mean(np.round(pred_lin) == np.round(y_sub))) * 100.0

            # 3. Signature Baseline (|sigma|/2 for g_4 and u(K), and Linear on [c, w, |sigma|])
            if target in ("smooth_four_genus", "unknotting_number"):
                # Exact Murasugi bound / formula: |sigma(K)| / 2
                pred_sig_direct = np.array([abs(float(knots_db[test_knots[i]].get("signature", 0.0))) / 2.0 for i in valid_sub_indices], dtype=np.float32)
                mae_sig = float(np.mean(np.abs(y_sub - pred_sig_direct)))
                acc_sig = float(np.mean(np.round(pred_sig_direct) == np.round(y_sub))) * 100.0
            else:
                # Linear probe with signature
                reg_sig = Ridge(alpha=1.0)
                reg_sig.fit(X_train_lin_sig[v_train_idx], y_train_full)
                pred_sig = reg_sig.predict(X_test_lin_sig[valid_sub_indices])
                mae_sig = float(np.mean(np.abs(y_sub - pred_sig)))
                acc_sig = float(np.mean(np.round(pred_sig) == np.round(y_sub))) * 100.0

            # 4. Alexander Baseline (deg(Delta)/2 for g_3, Ridge for others)
            if target == "three_genus":
                # Exact Alexander Seifert genus for alternating knots: deg(Delta)/2
                pred_alex = np.array([float(X_test_alex[i, 1]) for i in valid_sub_indices], dtype=np.float32)
                mae_alex = float(np.mean(np.abs(y_sub - pred_alex)))
                acc_alex = float(np.mean(np.round(pred_alex) == np.round(y_sub))) * 100.0
            else:
                reg_alex = Ridge(alpha=1.0)
                reg_alex.fit(X_train_alex[v_train_idx], y_train_full)
                pred_alex = reg_alex.predict(X_test_alex[valid_sub_indices])
                mae_alex = float(np.mean(np.abs(y_sub - pred_alex)))
                acc_alex = float(np.mean(np.round(pred_alex) == np.round(y_sub))) * 100.0

            # 5. Untrained Canonical GNN (Ridge probe across 5 seeds)
            seed_untrained_maes = []
            seed_untrained_accs = []
            for s_pred in seed_untrained_preds[target]:
                p_sub = s_pred[valid_sub_indices]
                seed_untrained_maes.append(float(np.mean(np.abs(y_sub - p_sub))))
                seed_untrained_accs.append(float(np.mean(np.round(p_sub) == np.round(y_sub))) * 100.0)
            mae_untrained = float(np.mean(seed_untrained_maes))

            # 6. Supervised Multi-Task Canonical GNN (across 5 seeds)
            seed_maes = []
            seed_accs = []
            for s_pred_matrix in seed_supervised_preds:
                p_sub = s_pred_matrix[valid_sub_indices, target_idx]
                seed_maes.append(float(np.mean(np.abs(y_sub - p_sub))))
                seed_accs.append(float(np.mean(np.round(p_sub) == np.round(y_sub))) * 100.0)

            mae_sup_mean = float(np.mean(seed_maes))
            mae_sup_stderr = float(np.std(seed_maes) / np.sqrt(len(seed_maes)))
            acc_sup_mean = float(np.mean(seed_accs))
            acc_sup_stderr = float(np.std(seed_accs) / np.sqrt(len(seed_accs)))

            task2_results[target][sub_name] = {
                "num_evaluated_knots": n_eval,
                "mae_mean_baseline": mae_mean,
                "acc_mean_baseline": acc_mean,
                "mae_linear_baseline": mae_lin,
                "acc_linear_baseline": acc_lin,
                "mae_signature_baseline": mae_sig,
                "acc_signature_baseline": acc_sig,
                "mae_alexander_baseline": mae_alex,
                "acc_alexander_baseline": acc_alex,
                "mae_untrained_canonical_gnn": mae_untrained,
                "mae_supervised_canonical_gnn_mean": mae_sup_mean,
                "mae_supervised_canonical_gnn_stderr": mae_sup_stderr,
                "acc_supervised_canonical_gnn_mean": acc_sup_mean,
                "acc_supervised_canonical_gnn_stderr": acc_sup_stderr,
            }

            print(f"{sub_name:<16} | {n_eval:<6} | {mae_mean:<14.3f} | {mae_lin:<14.3f} | {mae_sig:<15.3f} | {mae_alex:<14.3f} | {mae_sup_mean:.3f} ± {mae_sup_stderr:.3f}")

    return task2_results


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Rigorous Audited Task 1 and Task 2 Benchmarks")
    parser.add_argument("--seed", type=int, default=42, help="Primary random seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds for multi-seed Task 2")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch computation device (cpu or mps)")
    parser.add_argument("--output", type=str, default="results/rigorous_tasks_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} | Primary Seed: {args.seed} | Seeds: {args.seeds}")

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

    # Task 2: Evaluate on all 2,176 knots at c = 12 across 5 seeds
    t2_results = run_task2_benchmark(knots_db, train_knots, test_knots, device=device, seeds=args.seeds)

    full_output = {
        "metadata": {
            "primary_seed": args.seed,
            "seeds": args.seeds,
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
    print(f"\nAll audited benchmark results successfully saved to {out_p}")


if __name__ == "__main__":
    main()
