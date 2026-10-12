"""
reidemeister.py - Stochastic Reidemeister Move Generator in Braid & Diagram Space.

Provides exact ambient isotopy transformations for knots using Artin braid relations,
Markov stabilization/destabilization, and planar isotopies:
  - R1: Markov stabilization (strand insertion with twist) & destabilization
  - R2: Cancelling pair insertion (adjacent sigma_i * sigma_i^-1) & deletion
  - R3: Artin braid relation / Yang-Baxter slide (sigma_i * sigma_{i+1} * sigma_i <-> sigma_{i+1} * sigma_i * sigma_{i+1})
  - Planar isotopies: Far commutation (sigma_i * sigma_j <-> sigma_j * sigma_i for |i-j| >= 2)
  - Cyclic shift / Braid conjugation (Markov move 1)

All moves provably preserve the ambient isotopy class [K] and exact topological invariants
(Alexander polynomial, Jones polynomial, determinant, SnapPy hyperbolic volume).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .draw import render_knot_diagram
from .knotgen import Knot, braid


# ==============================================================================
# Low-Level Braid Word Reidemeister Operations
# ==============================================================================

def reidemeister_1_stabilize(
    word: Sequence[int],
    nstrands: int,
    positive: bool = True,
) -> Tuple[List[int], int]:
    """
    R1 Move (Markov Stabilization):
    beta in B_n -> beta * sigma_n^{pm 1} in B_{n+1}.
    Preserves the closure link/knot type exactly.
    """
    sign = 1 if positive else -1
    new_gen = nstrands
    new_word = list(word) + [sign * new_gen]
    return new_word, nstrands + 1


def reidemeister_1_destabilize(
    word: Sequence[int],
    nstrands: int,
) -> Optional[Tuple[List[int], int]]:
    """
    Inverse R1 Move (Markov Destabilization):
    If the generator for the highest strand (pm (nstrands - 1)) appears exactly once
    at the boundary (end or start of the braid word), it can be dropped, reducing
    strand count to nstrands - 1.
    """
    if nstrands <= 2 or not word:
        return None
    target_gen = nstrands - 1
    w = list(word)
    counts = sum(1 for g in w if abs(g) == target_gen)
    if counts == 1:
        if abs(w[-1]) == target_gen:
            return w[:-1], nstrands - 1
        if abs(w[0]) == target_gen:
            return w[1:], nstrands - 1
    return None


def reidemeister_2_insert(
    word: Sequence[int],
    nstrands: int,
    strand: Optional[int] = None,
    pos: Optional[int] = None,
    positive_first: bool = True,
) -> List[int]:
    """
    R2 Move (Cancelling Pair Insertion):
    Inserts sigma_i * sigma_i^{-1} (or inverse) at an arbitrary position in the braid word.
    """
    w = list(word)
    if nstrands <= 1:
        nstrands = 2
    max_gen = max(1, nstrands - 1)
    if strand is None or not (1 <= strand <= max_gen):
        strand = random.randint(1, max_gen)
    if pos is None or not (0 <= pos <= len(w)):
        pos = random.randint(0, len(w))

    s1 = strand if positive_first else -strand
    s2 = -s1
    w[pos:pos] = [s1, s2]
    return w


def reidemeister_2_delete(
    word: Sequence[int],
) -> Optional[List[int]]:
    """
    Inverse R2 Move (Cancelling Pair Deletion):
    Finds and cancels an adjacent pair sigma_i * sigma_i^{-1} or sigma_i^{-1} * sigma_i.
    """
    w = list(word)
    for i in range(len(w) - 1):
        if w[i] == -w[i + 1]:
            del w[i : i + 2]
            return w
    return None


def reidemeister_3_slide(
    word: Sequence[int],
) -> Optional[List[int]]:
    """
    R3 Move (Artin Braid Relation / Triangle Slide):
    Replaces:
      sigma_i * sigma_{i+1} * sigma_i <-> sigma_{i+1} * sigma_i * sigma_{i+1}
    or inverse with identical signs:
      -sigma_i * -sigma_{i+1} * -sigma_i <-> -sigma_{i+1} * -sigma_i * -sigma_{i+1}
    """
    w = list(word)
    candidates: List[Tuple[int, List[int]]] = []

    for i in range(len(w) - 2):
        g1, g2, g3 = w[i], w[i + 1], w[i + 2]
        # Same sign check
        if (g1 > 0 and g2 > 0 and g3 > 0) or (g1 < 0 and g2 < 0 and g3 < 0):
            sign = 1 if g1 > 0 else -1
            a1, a2, a3 = abs(g1), abs(g2), abs(g3)
            # Pattern A: a, a+1, a -> a+1, a, a+1
            if a1 == a3 and a2 == a1 + 1:
                replacement = [sign * (a1 + 1), sign * a1, sign * (a1 + 1)]
                candidates.append((i, replacement))
            # Pattern B: a+1, a, a+1 -> a, a+1, a
            elif a1 == a3 and a1 == a2 + 1:
                replacement = [sign * a2, sign * (a2 + 1), sign * a2]
                candidates.append((i, replacement))

    if not candidates:
        return None

    idx, repl = random.choice(candidates)
    w[idx : idx + 3] = repl
    return w


def braid_far_commute(
    word: Sequence[int],
) -> Optional[List[int]]:
    """
    Planar Isotopy Move (Far Commutation):
    Swaps adjacent generators sigma_i * sigma_j <-> sigma_j * sigma_i when |i - j| >= 2.
    """
    w = list(word)
    swaps: List[int] = []
    for i in range(len(w) - 1):
        if abs(abs(w[i]) - abs(w[i + 1])) >= 2:
            swaps.append(i)

    if not swaps:
        return None

    idx = random.choice(swaps)
    w[idx], w[idx + 1] = w[idx + 1], w[idx]
    return w


def braid_cyclic_shift(
    word: Sequence[int],
    shift: int = 1,
) -> List[int]:
    """
    Planar Isotopy Move (Conjugation / Cyclic Rotation):
    Cyclic rotation of the braid word w -> w[k:] + w[:k].
    Preserves the braid closure isotopy class by Markov's First Theorem.
    """
    w = list(word)
    if len(w) <= 1:
        return w
    s = shift % len(w)
    return w[s:] + w[:s]


# ==============================================================================
# High-Level Stochastic Reidemeister Engine
# ==============================================================================

def apply_stochastic_reidemeister_moves(
    word: Sequence[int],
    nstrands: int,
    num_moves: int = 3,
    max_strands: int = 6,
    max_length: int = 32,
    p_r1: float = 0.20,
    p_r2: float = 0.35,
    p_r3: float = 0.20,
    p_commute: float = 0.15,
    p_conjugate: float = 0.10,
) -> Tuple[List[int], int]:
    """
    Applies a stochastic sequence of ambient isotopy moves (R1, R2, R3, commutation, conjugation)
    to a braid word, returning a provably equivalent knot representation.

    Args:
        word: Initial braid word.
        nstrands: Initial strand count.
        num_moves: Number of stochastic moves to execute.
        max_strands: Upper bound on braid strand count to prevent unbounded growth.
        max_length: Upper bound on braid word length.

    Returns:
        Tuple[List[int], int]: Modified braid word and strand count.
    """
    cur_word = list(word)
    cur_n = nstrands

    moves_pool = ["r1", "r2", "r3", "commute", "conjugate"]
    weights = [p_r1, p_r2, p_r3, p_commute, p_conjugate]

    for _ in range(num_moves):
        chosen_move = random.choices(moves_pool, weights=weights, k=1)[0]

        if chosen_move == "r1":
            # Decide whether to stabilize or destabilize
            destab = reidemeister_1_destabilize(cur_word, cur_n)
            if destab is not None and random.random() < 0.5:
                cur_word, cur_n = destab
            elif cur_n < max_strands and len(cur_word) < max_length:
                cur_word, cur_n = reidemeister_1_stabilize(
                    cur_word, cur_n, positive=random.choice([True, False])
                )

        elif chosen_move == "r2":
            # Decide whether to insert or delete
            del_res = reidemeister_2_delete(cur_word)
            if del_res is not None and random.random() < 0.4:
                cur_word = del_res
            elif len(cur_word) + 2 <= max_length:
                cur_word = reidemeister_2_insert(
                    cur_word, cur_n, positive_first=random.choice([True, False])
                )

        elif chosen_move == "r3":
            r3_res = reidemeister_3_slide(cur_word)
            if r3_res is not None:
                cur_word = r3_res
            else:
                # Fallback to cyclic shift if no R3 subword is present
                cur_word = braid_cyclic_shift(cur_word, shift=random.choice([-1, 1]))

        elif chosen_move == "commute":
            comm_res = braid_far_commute(cur_word)
            if comm_res is not None:
                cur_word = comm_res

        elif chosen_move == "conjugate":
            if cur_word:
                cur_word = braid_cyclic_shift(cur_word, shift=random.choice([-1, 1]))

        # Invariant: strand count must always strictly exceed the maximum generator index
        if cur_word:
            cur_n = max(cur_n, max(abs(g) for g in cur_word) + 1)

    return cur_word, cur_n


def generate_reidemeister_variant(
    knot: Union[Knot, Dict[str, Any]],
    num_moves: int = 3,
    name: Optional[str] = None,
) -> Knot:
    """
    Generates a new Knot instance that is related to the input by stochastic Reidemeister moves.
    """
    if isinstance(knot, dict):
        word = list(knot.get("braid", []))
        nstrands = int(knot.get("nstrands", 2))
        base_name = str(knot.get("name", "knot"))
    else:
        word = list(knot.word)
        nstrands = int(knot.n)
        base_name = knot.name

    new_word, new_n = apply_stochastic_reidemeister_moves(word, nstrands, num_moves=num_moves)
    new_name = name or f"{base_name}_R{num_moves}"
    return braid(new_word, new_n, name=new_name)


def generate_reidemeister_diagram_pair(
    knot: Union[Knot, Dict[str, Any]],
    num_moves: int = 3,
    size: Tuple[int, int] = (224, 224),
    line_width: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray, Knot, Knot]:
    """
    Generates a paired (D, R_i(D)) image dataset sample for self-supervised contrastive learning.

    Returns:
        Tuple[img_original, img_moved, knot_original, knot_moved]
    """
    if isinstance(knot, dict):
        word = list(knot.get("braid", []))
        nstrands = int(knot.get("nstrands", 2))
        base_name = str(knot.get("name", "knot"))
        k_orig = braid(word, nstrands, name=base_name)
    else:
        k_orig = knot

    k_moved = generate_reidemeister_variant(k_orig, num_moves=num_moves)

    img_orig = render_knot_diagram(k_orig, size=size, line_width=line_width)
    img_moved = render_knot_diagram(k_moved, size=size, line_width=line_width)

    return img_orig, img_moved, k_orig, k_moved
