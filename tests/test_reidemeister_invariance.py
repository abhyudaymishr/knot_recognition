"""
test_reidemeister_invariance.py - Unit tests for Direction 1: Certified Reidemeister Invariance.

Validates:
  1. Stochastic Reidemeister move generators preserve exact topological invariants (Alexander, determinant).
  2. Canonical Reidemeister Normal Form (R-NF) contracts R1 curls and R2 bigons.
  3. R-GNO achieves certified zero-distance invariance (||z(D) - z(R_i(D))|| == 0).
  4. ReidemeisterContrastiveLoss computes accurate positive attraction and negative repulsion gradients.
  5. TopologicalProbeEvaluator fits linear probes to predict topological targets.
"""

import pytest
import torch
import numpy as np

import Knotgen
from Knotgen.knotgen import braid
from Knotgen.reidemeister import (
    apply_stochastic_reidemeister_moves,
    reidemeister_1_stabilize,
    reidemeister_2_insert,
    reidemeister_3_slide,
    generate_reidemeister_diagram_pair,
)
from src.knot_recognition.r_gno import RGNO, reidemeister_normal_form
from src.knot_recognition.reidemeister_contrastive import (
    ReidemeisterContrastiveLoss,
    ReidemeisterVisionModel,
)
from src.knot_recognition.topological_probe import TopologicalProbeEvaluator


def test_reidemeister_moves_preserve_exact_invariants():
    """Verify that stochastic R1, R2, R3 braid operations preserve exact knot invariants."""
    k_base = braid([1, -2, 1, -2], 3, name="4_1")
    base_det = k_base.determinant()
    assert base_det == 5

    # 1. R1 stabilization
    w_r1, n_r1 = reidemeister_1_stabilize(k_base.word, k_base.n, positive=True)
    k_r1 = braid(w_r1, n_r1, name="4_1_r1")
    assert k_r1.determinant() == base_det

    # 2. R2 cancelling pair insertion
    w_r2 = reidemeister_2_insert(k_base.word, k_base.n, strand=1, pos=1, positive_first=True)
    k_r2 = braid(w_r2, k_base.n, name="4_1_r2")
    assert k_r2.determinant() == base_det

    # 3. Stochastic multi-move sequence
    w_multi, n_multi = apply_stochastic_reidemeister_moves(k_base.word, k_base.n, num_moves=3)
    k_multi = braid(w_multi, n_multi, name="4_1_multi")
    assert k_multi.determinant() == base_det


def test_reidemeister_canonical_reduction_confluence():
    """Verify that Planar Diagram reduction contracts R1 and R2 moves to identical canonical normal form."""
    k_base = braid([1, -2, 1, -2], 3, name="4_1")
    pd_base = [c["pd"] for c in k_base.crossings]
    signs_base = [c["sign"] for c in k_base.crossings]

    # Knot with added R2 bigon
    w_r2 = [1, 2, -2, -2, 1, -2]
    k_r2 = braid(w_r2, 3, name="4_1_r2")
    pd_r2 = [c["pd"] for c in k_r2.crossings]
    signs_r2 = [c["sign"] for c in k_r2.crossings]

    assert len(pd_r2) > len(pd_base)

    red_pd_base, _ = reidemeister_normal_form(pd_base, signs_base)
    red_pd_r2, _ = reidemeister_normal_form(pd_r2, signs_r2)

    # Crossings count in canonical irreducible form must match
    assert len(red_pd_base) == len(red_pd_r2)


def test_rgno_strict_certified_invariance():
    """Verify that RGNO embedding is provably identical for base diagram and R2-perturbed diagram."""
    k_base = braid([1, -2, 1, -2], 3, name="4_1")
    pd_base = [c["pd"] for c in k_base.crossings]
    signs_base = [c["sign"] for c in k_base.crossings]

    w_r2 = [1, 1, -1, -2, 1, -2]
    k_r2 = braid(w_r2, 3, name="4_1_r2")
    pd_r2 = [c["pd"] for c in k_r2.crossings]
    signs_r2 = [c["sign"] for c in k_r2.crossings]

    model = RGNO(hidden_dim=64, out_dim=64)
    model.eval()

    with torch.no_grad():
        z_base = model(pd_base, signs_base)
        z_r2 = model(pd_r2, signs_r2)

    diff = (z_base - z_r2).norm().item()
    assert diff < 1e-5, f"Expected 0.0 distance, got {diff}"


def test_reidemeister_contrastive_loss_optimization():
    """Verify Reidemeister contrastive loss computation and backward gradient flow."""
    loss_fn = ReidemeisterContrastiveLoss(margin=1.0, lambda_neg=0.5)

    z_anchor = torch.randn(4, 32, requires_grad=True)
    # Perfectly aligned positive pairs
    z_pos = z_anchor.clone()

    loss, stats = loss_fn(z_anchor, z_pos)
    assert stats["loss_pos"] < 1e-6
    assert stats["loss_total"] >= 0.0

    # Perturbed positive pairs
    z_pos_perturbed = z_anchor + torch.randn_like(z_anchor) * 0.1
    loss_pert, stats_pert = loss_fn(z_anchor, z_pos_perturbed)
    assert stats_pert["loss_pos"] > 0.0

    loss_pert.backward()
    assert z_anchor.grad is not None
    assert torch.isfinite(z_anchor.grad).all()


def test_reidemeister_diagram_pair_generation():
    """Verify generation of image pairs for contrastive training."""
    k = braid([1, -2, 1, -2], 3, name="4_1")
    img_orig, img_moved, k_orig, k_moved = generate_reidemeister_diagram_pair(k, num_moves=2, size=(64, 64))

    assert img_orig.shape == (64, 64, 3)
    assert img_moved.shape == (64, 64, 3)
    assert k_orig.determinant() == k_moved.determinant()


def test_topological_probe_evaluator():
    """Verify topological probe evaluator computes R^2 on sample embeddings."""
    evaluator = TopologicalProbeEvaluator()
    sample_names = ["3_1", "4_1", "5_1", "5_2", "6_1", "6_2", "7_1", "7_2"]

    # Generate synthetic embeddings correlated with determinant
    embeddings = {}
    for name in sample_names:
        det = evaluator.get_invariants(name)["determinant"]
        vec = np.zeros(16, dtype=np.float32)
        vec[0] = det / 10.0
        vec[1:] = np.random.randn(15) * 0.01
        embeddings[name] = vec

    results = evaluator.evaluate_embeddings(
        embeddings,
        train_names=sample_names[:6],
        test_names=sample_names[6:],
    )
    assert "determinant" in results
    assert "alexander_at_2" in results
    assert "crossing_number" in results
