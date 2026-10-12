"""
knotinfo_bench.py - KnotInfo Ground Truth Benchmark & Hard Negative Mining.

Extracts exact topological invariants from database_knotinfo:
  - signature: int
  - three_genus: int
  - braid_index: int
  - unknotting_number: float (handles unresolved [2,3] intervals gracefully)
  - alexander_polynomial: string & evaluated coefficients
  - jones_polynomial: string

Provides:
  1. Ground-truth invariant targets for all 250 census knots.
  2. Hard negative pair mining:
     - Same crossing number pairs (within split)
     - Same Alexander polynomial pairs
     - Same Jones polynomial pairs
  3. Trivial baseline predictors:
     - Mean predictor
     - Linear / Ridge regression on crossing number c(K) and writhe w(K).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge

import database_knotinfo


# ==============================================================================
# KnotInfo Data Extraction & Parsing
# ==============================================================================

@dataclass
class KnotInfoEntry:
    name: str
    crossing_number: int
    signature: int
    three_genus: int
    braid_index: int
    unknotting_number: float
    unknotting_number_raw: str
    alexander_poly_raw: str
    jones_poly_raw: str
    writhe: int = 0


def parse_unknotting_number(raw_val: Any) -> Tuple[float, str]:
    """Parses unknotting number string, converting intervals like '[2,3]' to 2.5."""
    s = str(raw_val).strip()
    if not s or s == "None":
        return 0.0, "0"
    if s.startswith("[") and s.endswith("]"):
        parts = [float(x.strip()) for x in s[1:-1].split(",") if x.strip()]
        if parts:
            return float(np.mean(parts)), s
    try:
        return float(int(s)), s
    except Exception:
        # Fallback regex search for digits
        m = re.findall(r"\d+", s)
        if m:
            nums = [float(x) for x in m]
            return float(np.mean(nums)), s
    return 1.0, s


class KnotInfoBenchmark:
    """Manages KnotInfo data, hard negatives, and trivial baseline predictors."""

    def __init__(self, census_path: Optional[str] = None):
        if census_path is None:
            census_path = str(Path(__file__).resolve().parent.parent.parent / "Knotgen" / "census_braids.json")
        with open(census_path, "r") as f:
            self.census = json.load(f)

        self.entries: Dict[str, KnotInfoEntry] = {}
        self._load_knotinfo()

    def _load_knotinfo(self) -> None:
        raw_links = {row.get("name"): row for row in database_knotinfo.link_list()}

        # Unknot special case
        self.entries["unknot"] = KnotInfoEntry(
            name="unknot",
            crossing_number=0,
            signature=0,
            three_genus=0,
            braid_index=1,
            unknotting_number=0.0,
            unknotting_number_raw="0",
            alexander_poly_raw="1",
            jones_poly_raw="1",
            writhe=0,
        )

        for name, data in self.census.items():
            if name == "unknot":
                continue
            row = raw_links.get(name, {})
            c = int(row.get("crossing_number", data.get("crossings", 0)))
            sig = int(row.get("signature", 0))
            genus = int(row.get("three_genus", 0))
            braid_idx = int(row.get("braid_index", data.get("nstrands", 2)))
            unknot_num, unknot_raw = parse_unknotting_number(row.get("unknotting_number", "1"))
            alex_raw = str(row.get("alexander_polynomial", "1"))
            jones_raw = str(row.get("jones_polynomial", "1"))

            # Calculate writhe from census braid word if present
            braid_word = data.get("braid", [])
            w = sum(1 if g > 0 else -1 for g in braid_word) if braid_word else 0

            self.entries[name] = KnotInfoEntry(
                name=name,
                crossing_number=c,
                signature=sig,
                three_genus=genus,
                braid_index=braid_idx,
                unknotting_number=unknot_num,
                unknotting_number_raw=unknot_raw,
                alexander_poly_raw=alex_raw,
                jones_poly_raw=jones_raw,
                writhe=w,
            )

    def get_knot(self, name: str) -> KnotInfoEntry:
        return self.entries.get(name, self.entries["unknot"])

    def get_splits(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Splits knots strictly by knot identity (keeping knot and mirror in same split):
          - train_knots: c <= 8 (37 knots)
          - test_9_knots: c == 9 (49 knots)
          - test_10_knots: c == 10 (164-165 knots)
        """
        train_knots, test_9, test_10 = [], [], []
        for name, entry in self.entries.items():
            if entry.crossing_number <= 8:
                train_knots.append(name)
            elif entry.crossing_number == 9:
                test_9.append(name)
            elif entry.crossing_number == 10:
                test_10.append(name)
        return sorted(train_knots), sorted(test_9), sorted(test_10)

    # ==========================================================================
    # Hard Negative Mining
    # ==========================================================================

    def mine_hard_negatives(
        self,
        anchor_name: str,
        candidate_pool: Sequence[str],
        num_negatives: int = 5,
        rng: Optional[np.random.Generator] = None,
    ) -> List[str]:
        """
        Mines hard negative knots from candidate_pool for anchor_name:
          Priority 1: Same Alexander polynomial or same Jones polynomial (different knot).
          Priority 2: Same crossing number c(K).
          Priority 3: Other knots in candidate_pool.
        """
        if rng is None:
            rng = np.random.default_rng()

        anchor = self.get_knot(anchor_name)
        pool = [k for k in candidate_pool if k != anchor_name]
        if not pool:
            return []

        # Priority 1: Identical polynomial
        same_poly = [
            k for k in pool
            if (self.get_knot(k).alexander_poly_raw == anchor.alexander_poly_raw
                or self.get_knot(k).jones_poly_raw == anchor.jones_poly_raw)
        ]

        # Priority 2: Identical crossing number
        same_crossing = [
            k for k in pool
            if k not in same_poly and self.get_knot(k).crossing_number == anchor.crossing_number
        ]

        # Priority 3: Remainder
        other = [k for k in pool if k not in same_poly and k not in same_crossing]

        selected: List[str] = []
        if same_poly:
            perm = rng.permutation(len(same_poly))
            selected.extend([same_poly[i] for i in perm[:num_negatives]])

        remaining = num_negatives - len(selected)
        if remaining > 0 and same_crossing:
            perm = rng.permutation(len(same_crossing))
            selected.extend([same_crossing[i] for i in perm[:remaining]])

        remaining = num_negatives - len(selected)
        if remaining > 0 and other:
            perm = rng.permutation(len(other))
            selected.extend([other[i] for i in perm[:remaining]])

        return selected[:num_negatives]

    # ==========================================================================
    # Trivial Baseline Predictors (Crossing Number & Writhe)
    # ==========================================================================

    def evaluate_trivial_predictors(
        self,
        train_knots: Sequence[str],
        test_knots: Sequence[str],
        targets: Sequence[str] = ("signature", "three_genus", "braid_index", "unknotting_number"),
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates trivial predictors on unseen test knots:
          1. Mean predictor: predicts the train set target mean for all test knots.
          2. Linear fit: Ridge regression on [crossing_number, writhe].
        Returns MAE and accuracy (for integer targets within 0.5) for each target.
        """
        results: Dict[str, Dict[str, float]] = {}

        X_train = np.array([[self.get_knot(k).crossing_number, self.get_knot(k).writhe] for k in train_knots], dtype=np.float32)
        X_test = np.array([[self.get_knot(k).crossing_number, self.get_knot(k).writhe] for k in test_knots], dtype=np.float32)

        for target in targets:
            y_train = np.array([getattr(self.get_knot(k), target) for k in train_knots], dtype=np.float32)
            y_test = np.array([getattr(self.get_knot(k), target) for k in test_knots], dtype=np.float32)

            # 1. Mean predictor
            y_pred_mean = np.full_like(y_test, np.mean(y_train))
            mae_mean = float(np.mean(np.abs(y_test - y_pred_mean)))
            acc_mean = float(np.mean(np.round(y_pred_mean) == np.round(y_test)))

            # 2. Linear fit (crossing number & writhe)
            clf = Ridge(alpha=1.0)
            clf.fit(X_train, y_train)
            y_pred_lin = clf.predict(X_test)
            mae_lin = float(np.mean(np.abs(y_test - y_pred_lin)))
            acc_lin = float(np.mean(np.round(y_pred_lin) == np.round(y_test)))

            results[target] = {
                "mean_mae": mae_mean,
                "mean_acc": acc_mean,
                "linear_mae": mae_lin,
                "linear_acc": acc_lin,
            }

        return results
