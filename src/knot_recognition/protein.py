from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ProteinBackbone:
    chain_id: str
    ca_coords: np.ndarray


@dataclass(frozen=True)
class Crossing:
    i: int
    j: int
    point: np.ndarray
    over: int
    under: int
    ti: float
    tj: float


def extract_ca_polyline(pdb_path: str, chain_id: Optional[str] = None) -> ProteinBackbone:
    chains: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21].strip() or "_"
            try:
                res_seq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                continue
            chains.setdefault(chain, []).append((res_seq, np.array([x, y, z], dtype=float)))

    if not chains:
        raise ValueError("No CA atoms found in PDB file.")

    if chain_id is None:
        chain_id = max(chains.keys(), key=lambda c: len(chains[c]))

    if chain_id not in chains:
        raise ValueError(f"Chain '{chain_id}' not found in PDB file.")

    coords = sorted(chains[chain_id], key=lambda t: t[0])
    ca_coords = np.stack([c for _, c in coords], axis=0)
    return ProteinBackbone(chain_id=chain_id, ca_coords=ca_coords)


def sample_viewpoints(n: int = 32) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    points = []
    phi = (1 + 5 ** 0.5) / 2
    for k in range(n):
        z = 1 - 2 * (k + 0.5) / n
        r = (1 - z * z) ** 0.5
        theta = 2 * np.pi * (k / phi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    return np.array(points, dtype=float)


def project_polyline(coords: np.ndarray, view: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    view = np.asarray(view, dtype=float)
    view = view / np.linalg.norm(view)
    # build orthonormal basis (u, v, view)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(view, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    u = np.cross(up, view)
    u = u / np.linalg.norm(u)
    v = np.cross(view, u)
    # projection
    xy = np.stack([coords @ u, coords @ v], axis=1)
    depth = coords @ view
    return xy, depth


def _seg_intersect(p1, p2, q1, q2):
    # segment intersection in 2D (excluding collinear overlap)
    p = np.array(p1)
    r = np.array(p2) - p
    q = np.array(q1)
    s = np.array(q2) - q
    rxs = r[0] * s[1] - r[1] * s[0]
    q_p = q - p
    q_pxr = q_p[0] * r[1] - q_p[1] * r[0]
    if abs(rxs) < 1e-9:
        return None
    t = (q_p[0] * s[1] - q_p[1] * s[0]) / rxs
    u = q_pxr / rxs
    if 0 < t < 1 and 0 < u < 1:
        pt = p + t * r
        return t, u, pt
    return None


def detect_crossings(xy: np.ndarray, depth: np.ndarray) -> List[Crossing]:
    n = len(xy)
    crossings: List[Crossing] = []
    for i in range(n - 1):
        p1, p2 = xy[i], xy[i + 1]
        for j in range(i + 2, n - 1):
            if j == i + 1:
                continue
            q1, q2 = xy[j], xy[j + 1]
            hit = _seg_intersect(p1, p2, q1, q2)
            if hit is None:
                continue
            t, u, pt = hit
            # compute depth at intersection along each segment
            zi = depth[i] + t * (depth[i + 1] - depth[i])
            zj = depth[j] + u * (depth[j + 1] - depth[j])
            if zi >= zj:
                over, under = i, j
            else:
                over, under = j, i
            crossings.append(Crossing(i=i, j=j, point=pt, over=over, under=under, ti=t, tj=u))
    return crossings


def gauss_code_from_crossings(crossings: List[Crossing], n_points: int) -> List[int]:
    # Assign ids per crossing
    indexed = {idx: c for idx, c in enumerate(crossings, start=1)}
    events = []
    for cid, c in indexed.items():
        events.append((c.i, c.ti, cid, c.over == c.i))
        events.append((c.j, c.tj, cid, c.over == c.j))
    # sort by segment index then along-segment position
    events.sort(key=lambda e: (e[0], e[1]))
    gauss = []
    for _, __, cid, is_over in events:
        gauss.append(cid if is_over else -cid)
    return gauss
