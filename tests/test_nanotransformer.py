import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from knot_recognition.nanotransformer import (
    NanoTransformer,
    NanoTransformerConfig,
    load_nanotransformer,
    save_nanotransformer_checkpoint,
)
from knot_recognition.traverser import KnotTraverser


def test_nanotransformer_forward_and_shapes():
    cfg = NanoTransformerConfig(
        image_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        num_classes=10,
    )
    model = NanoTransformer(cfg)
    model.set_classes([f"knot_{i}" for i in range(10)])

    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 10)

    feats = model.extract_embedding(x, normalize=True)
    assert feats.shape == (2, 64)
    # Check unit norm
    norms = torch.norm(feats, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_nanotransformer_predict_and_similarity_retrieval():
    cfg = NanoTransformerConfig(
        image_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=2,
        num_classes=5,
    )
    model = NanoTransformer(cfg)
    classes = ["0_unknot", "3_1", "4_1", "5_1", "5_2"]
    model.set_classes(classes)

    # Set mock prototypes
    prototypes = torch.randn(5, 64)
    model.set_prototypes(prototypes)

    # Synthetic image
    img = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 200)
    pred_res = model.predict_image(img, top_k=3)

    assert "predictions" in pred_res
    assert len(pred_res["predictions"]) == 3
    assert "embedding" in pred_res
    assert pred_res["embedding"].shape == (64,)

    # Test similarity fetching
    similar = model.fetch_similar_knots(pred_res["embedding"], top_k=3)
    assert len(similar) == 3
    for s in similar:
        assert "label" in s
        assert "similarity" in s
        assert -1.0 <= s["similarity"] <= 1.0


def test_nanotransformer_save_and_load_checkpoint():
    cfg = NanoTransformerConfig(
        image_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=2,
        num_classes=3,
    )
    model = NanoTransformer(cfg)
    classes = ["0_unknot", "3_1", "4_1"]
    model.set_classes(classes)
    model.set_prototypes(torch.randn(3, 64))

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_nanovit.pth"
        save_nanotransformer_checkpoint(model, ckpt_path)
        assert ckpt_path.exists()

        loaded_model = load_nanotransformer(ckpt_path)
        assert loaded_model.classes == classes
        assert loaded_model.prototypes is not None
        assert loaded_model.prototypes.shape == (3, 64)


def test_traverser_integration_with_nanotransformer():
    cfg = NanoTransformerConfig(
        image_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=2,
        num_classes=3,
    )
    model = NanoTransformer(cfg)
    model.set_classes(["0_unknot", "3_1", "4_1"])
    model.set_prototypes(torch.randn(3, 64))

    traverser = KnotTraverser(nanotransformer=model)

    # Empty white image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    res = traverser.traverse_and_classify(img, top_k=2)

    assert res.classification is not None
    assert len(res.classification["predictions"]) == 2
    assert len(res.similar_knots) == 2
    assert res.consensus is not None
    assert "traverser_crossings" in res.consensus
    assert "predicted_label" in res.consensus


def test_nanotransformer_noise_awareness():
    from knot_recognition.nanotransformer import denoise_for_nanotransformer

    # Create artificial noisy image (textured background)
    rng = np.random.default_rng(42)
    noisy_img = rng.integers(0, 255, (200, 400, 3), dtype=np.uint8)
    pil_noisy = Image.fromarray(noisy_img)

    cleaned_pil, noise_val, is_noisy = denoise_for_nanotransformer(pil_noisy)
    assert is_noisy is True
    assert noise_val > 600.0
    # Wide image was cropped/centered
    assert cleaned_pil.size[0] <= pil_noisy.size[0]

    cfg = NanoTransformerConfig(
        image_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=2,
        num_classes=3,
    )
    model = NanoTransformer(cfg)
    model.set_classes(["0_unknot", "3_1", "4_1"])
    model.set_prototypes(torch.randn(3, 64))

    pred = model.predict_image(pil_noisy, top_k=2, noise_aware=True)
    assert pred["is_noisy"] is True
    assert pred["noise_level"] > 600.0
    assert len(pred["predictions"]) == 2

