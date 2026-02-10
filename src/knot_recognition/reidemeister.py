"""
Heuristic Reidemeister move detection from a single knot diagram image.

This module provides a conservative detector for R1/R2/R3 candidates using
skeleton geometry and simplified crossing graphs. It is intended for
qualitative inspection and pipeline development, not as a definitive oracle.
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import networkx as nx
from PIL import Image, ImageDraw

from .gauss_pd import cluster_junctions, prune_spurs, simplify_graph, skeleton_to_graph
from .preprocess import PreprocessConfig, Preprocessor
from .utils import imread_any


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ReidemeisterConfig:
    image_width: int = 512
    r1_max_cycle_len: int = 80
    r2_max_edge_len: int = 120
    r3_max_perimeter: int = 220
    max_nodes_for_r1: int = 20000
    score_floor: float = 0.0
    nms_iou: float = 0.3


@dataclass(frozen=True)
class MoveCandidate:
    move: str
    bbox: BBox
    score: float
    details: Dict[str, Any]


class ReidemeisterDetector:
    def __init__(self, config: Optional[ReidemeisterConfig] = None):
        self.config = config or ReidemeisterConfig()
        self.preprocessor = Preprocessor(PreprocessConfig(width=self.config.image_width))

    def detect(self, img_path: str, overlay_path: Optional[str] = None) -> List[Dict[str, Any]]:
        img = Image.fromarray(imread_any(img_path))
        img_np = np.array(img)
        skel, _ = self.preprocessor.run(img_np)

        candidates = []
        candidates.extend(self._detect_r2_r3(skel))
        candidates.extend(self._detect_r1(skel))

        candidates = self._nms(candidates, iou_thresh=self.config.nms_iou)
        candidates = [c for c in candidates if c.score >= self.config.score_floor]
        candidates.sort(key=lambda c: c.score, reverse=True)

        if overlay_path:
            self._save_overlay(img, candidates, overlay_path)

        return [asdict(c) for c in candidates]

    def _detect_r1(self, skel) -> List[MoveCandidate]:
        cfg = self.config
        G = skeleton_to_graph(skel, connectivity=8)
        G = prune_spurs(G, min_length=cfg.r1_max_cycle_len // 4)

        if G.number_of_nodes() > cfg.max_nodes_for_r1:
            return []

        # Use a simple graph for cycle detection
        cycles = nx.cycle_basis(G)
        candidates = []
        for cycle in cycles:
            if len(cycle) > cfg.r1_max_cycle_len:
                continue
            coords = [G.nodes[n]["pos"] for n in cycle]
            bbox = _bbox_from_coords(coords)
            score = _size_score(len(cycle), cfg.r1_max_cycle_len)
            candidates.append(
                MoveCandidate(
                    move="R1",
                    bbox=bbox,
                    score=score,
                    details={"cycle_len": len(cycle)},
                )
            )
        return candidates

    def _detect_r2_r3(self, skel) -> List[MoveCandidate]:
        cfg = self.config
        G = skeleton_to_graph(skel, connectivity=8)
        G = prune_spurs(G, min_length=cfg.r2_max_edge_len // 4)
        junctions, junction_map = cluster_junctions(G, deg_thresh=3)
        if not junctions:
            return []
        SG = simplify_graph(G, junctions, junction_map)

        # Build edge length map for junction-only edges
        edge_lengths = {}
        edge_paths = {}
        for u, v, k, data in SG.edges(keys=True, data=True):
            if SG.nodes[u]["type"] != "junction" or SG.nodes[v]["type"] != "junction":
                continue
            path = data.get("path", [])
            length = max(1, len(path))
            edge_lengths.setdefault((u, v), []).append(length)
            edge_paths.setdefault((u, v), []).append(path)

        candidates = []

        # R2: parallel edges between two junctions
        for (u, v), lengths in edge_lengths.items():
            if len(lengths) < 2:
                continue
            avg_len = float(sum(lengths)) / len(lengths)
            if avg_len > cfg.r2_max_edge_len:
                continue
            paths = edge_paths[(u, v)]
            bbox = _bbox_from_paths(paths)
            score = _size_score(avg_len, cfg.r2_max_edge_len)
            candidates.append(
                MoveCandidate(
                    move="R2",
                    bbox=bbox,
                    score=score,
                    details={"edge_count": len(lengths), "avg_len": avg_len},
                )
            )

        # R3: triangle among three junctions
        junction_nodes = [n for n, d in SG.nodes(data=True) if d.get("type") == "junction"]
        adj = {n: set() for n in junction_nodes}
        for (u, v), lengths in edge_lengths.items():
            adj[u].add(v)
            adj[v].add(u)

        seen = set()
        for i, u in enumerate(junction_nodes):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v]:
                    if w <= v or w not in adj[u]:
                        continue
                    key = tuple(sorted((u, v, w)))
                    if key in seen:
                        continue
                    seen.add(key)

                    per = 0.0
                    paths = []
                    for a, b in [(u, v), (v, w), (u, w)]:
                        lengths = edge_lengths.get((a, b)) or edge_lengths.get((b, a))
                        paths_list = edge_paths.get((a, b)) or edge_paths.get((b, a))
                        if not lengths or not paths_list:
                            per = None
                            break
                        idx = int(np.argmin(lengths))
                        per += lengths[idx]
                        paths.append(paths_list[idx])
                    if per is None or per > cfg.r3_max_perimeter:
                        continue

                    bbox = _bbox_from_paths(paths)
                    score = _size_score(per, cfg.r3_max_perimeter)
                    candidates.append(
                        MoveCandidate(
                            move="R3",
                            bbox=bbox,
                            score=score,
                            details={"perimeter": per},
                        )
                    )

        return candidates

    def _save_overlay(self, img: Image.Image, candidates: Iterable[MoveCandidate], out_path: str) -> None:
        draw = ImageDraw.Draw(img)
        for cand in candidates:
            x, y, w, h = cand.bbox
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            label = f"{cand.move} {cand.score:.2f}"
            draw.text((x + 2, y + 2), label, fill="red")
        img.save(out_path)

    def _nms(self, candidates: List[MoveCandidate], iou_thresh: float) -> List[MoveCandidate]:
        selected: List[MoveCandidate] = []
        for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
            if all(_bbox_iou(cand.bbox, s.bbox) < iou_thresh for s in selected):
                selected.append(cand)
        return selected


def detect_moves(img_path: str, config: Optional[ReidemeisterConfig] = None, overlay_path: Optional[str] = None):
    return ReidemeisterDetector(config).detect(img_path, overlay_path=overlay_path)


def _bbox_from_coords(coords: List[Tuple[int, int]]) -> BBox:
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    return (float(min_c), float(min_r), float(max_c - min_c), float(max_r - min_r))


def _bbox_from_paths(paths: List[List[Tuple[int, int]]]) -> BBox:
    coords = [p for path in paths for p in path]
    if not coords:
        return (0.0, 0.0, 1.0, 1.0)
    return _bbox_from_coords(coords)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def _size_score(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, 1.0 - (value / max_value))


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--overlay", default=None)
    args = parser.parse_args()

    results = detect_moves(args.image, overlay_path=args.overlay)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
