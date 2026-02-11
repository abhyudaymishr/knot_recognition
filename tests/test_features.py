import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from knot_recognition.features import extract_symmetry_invariant_features


def _make_symmetric_image(size=64):
    img = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    img[:, mid - 1 : mid + 1] = 255
    return img


def test_feature_vector_length_and_finite_grayscale():
    img = _make_symmetric_image()
    feats = extract_symmetry_invariant_features(img, resize_to=64)
    assert feats.shape == (24,)
    assert np.all(np.isfinite(feats))


def test_feature_vector_rgb_and_symmetry_score_range():
    img = _make_symmetric_image()
    rgb = np.stack([img, img, img], axis=-1)
    feats = extract_symmetry_invariant_features(rgb, resize_to=64)
    assert feats.shape == (24,)
    sym = float(feats[-1])
    assert -1.0 <= sym <= 1.0
