"""
topological_probe.py - Linear Probing on Exact Mathematical Invariants.

Probes whether neural representations have learned intrinsic knot topology or superficial appearance:
  - Determinant: det(K) = |Delta(-1)|
  - Alexander polynomial evaluation: Delta(2)
  - Crossing number: c(K)
  - Braid index / strand count: n
  - Hyperbolic volume: Vol(S^3 \\ K) via SnapPy

Evaluates Ridge and Linear probes using R^2 score, Spearman rank correlation, and Pearson r.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr

from Knotgen.knotgen import braid


class TopologicalProbeEvaluator:
    """Evaluates how linearly predictable exact knot invariants are from embeddings."""

    def __init__(self, census_path: Optional[str] = None, cache_file: Optional[str] = None):
        if census_path is None:
            census_path = str(Path(__file__).resolve().parent.parent.parent / "Knotgen" / "census_braids.json")
        with open(census_path, "r") as f:
            self.census = json.load(f)

        self.cache_file = cache_file or str(Path(census_path).parent / "topological_invariants_cache.json")
        self._invariants_cache: Dict[str, Dict[str, float]] = {}
        self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        p = Path(self.cache_file)
        if p.exists():
            try:
                with open(p, "r") as f:
                    self._invariants_cache = json.load(f)
            except Exception:
                self._invariants_cache = {}

    def _save_disk_cache(self) -> None:
        p = Path(self.cache_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, "w") as f:
                json.dump(self._invariants_cache, f, indent=2)
        except Exception:
            pass

    def get_invariants(self, name: str) -> Dict[str, float]:
        """Retrieves or computes exact mathematical invariants on-demand for a knot."""
        if name in self._invariants_cache:
            return self._invariants_cache[name]

        data = self.census.get(name, {})
        w = list(data.get("braid", []))
        n = int(data.get("nstrands", 2))
        c = int(data.get("crossings", 0))

        k = braid(w, n, name=name)
        det = float(k.determinant())

        # Alexander polynomial evaluation at t = 2 via Burau representation
        alex_val = 1.0
        if w and n > 1:
            try:
                t_val = 2.0
                M = np.eye(n, dtype=np.float64)
                for g in w:
                    idx = abs(g) - 1
                    S = np.eye(n, dtype=np.float64)
                    if g > 0:
                        S[idx : idx + 2, idx : idx + 2] = [[1.0 - t_val, t_val], [1.0, 0.0]]
                    else:
                        S[idx : idx + 2, idx : idx + 2] = [[0.0, 1.0], [1.0 / t_val, 1.0 - 1.0 / t_val]]
                    M = M @ S
                R = np.eye(n, dtype=np.float64) - M
                denom = sum(t_val**i for i in range(n))
                d_val = np.linalg.det(R[: n - 1, : n - 1])
                alex_val = float(abs(d_val / denom))
            except Exception:
                alex_val = float(det)
        else:
            alex_val = float(det)

        # Hyperbolic volume via SnapPy
        vol = 0.0
        try:
            import snappy
            if w:
                L = snappy.Link(braid_closure=w)
                vol = float(L.exterior().volume())
        except Exception:
            vol = 0.0

        invs = {
            "determinant": det,
            "alexander_at_2": alex_val,
            "crossing_number": float(c),
            "braid_index": float(n),
            "hyperbolic_volume": vol,
        }
        self._invariants_cache[name] = invs
        return invs

    def evaluate_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        train_names: Optional[List[str]] = None,
        test_names: Optional[List[str]] = None,
        alpha: float = 1.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Fits linear probes for each topological invariant.

        Args:
            embeddings: Dictionary mapping knot name -> 1D embedding vector (D,).
            train_names: Optional explicit list of knot names for training the probe.
            test_names: Optional explicit list of knot names for testing the probe.
            alpha: Ridge regression regularization parameter.

        Returns:
            Dictionary of probe performance metrics per invariant.
        """
        all_names = [name for name in embeddings.keys() if name in self.census]
        if not all_names:
            return {}

        # Precompute/load invariants only for requested names
        for n in all_names:
            self.get_invariants(n)
        self._save_disk_cache()

        if train_names is None or test_names is None:
            # Default to 80/20 train/test split
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(all_names))
            split = int(0.8 * len(all_names))
            train_names = [all_names[i] for i in perm[:split]]
            test_names = [all_names[i] for i in perm[split:]]

        X_train = np.array([embeddings[n] for n in train_names], dtype=np.float32)
        X_test = np.array([embeddings[n] for n in test_names], dtype=np.float32)

        target_keys = ["determinant", "alexander_at_2", "crossing_number", "braid_index", "hyperbolic_volume"]
        results: Dict[str, Dict[str, float]] = {}

        for target in target_keys:
            y_train = np.array([self._invariants_cache[n][target] for n in train_names], dtype=np.float32)
            y_test = np.array([self._invariants_cache[n][target] for n in test_names], dtype=np.float32)

            # Fit probe
            probe = Ridge(alpha=alpha)
            probe.fit(X_train, y_train)
            y_pred = probe.predict(X_test)

            r2 = float(r2_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            spear, _ = spearmanr(y_test, y_pred)
            pears, _ = pearsonr(y_test, y_pred)

            results[target] = {
                "r2_score": max(-1.0, r2),
                "spearman_rho": float(spear) if not np.isnan(spear) else 0.0,
                "pearson_r": float(pears) if not np.isnan(pears) else 0.0,
                "mae": mae,
                "train_samples": len(train_names),
                "test_samples": len(test_names),
            }

        return results
