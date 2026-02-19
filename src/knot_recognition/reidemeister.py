from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
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
    scales: Tuple[float, ...] = (0.85, 1.0, 1.15)
    r1_max_cycle_len: int = 80
    r1_min_cycle_len: int = 8
    r1_max_junctions: int = 2
    r2_max_edge_len: int = 120
    r2_length_ratio: float = 1.5
    r3_max_perimeter: int = 220
    r3_edge_ratio: float = 1.5
    max_nodes_for_r1: int = 20000
    spur_prune_ratio: float = 0.25
    junction_degree: int = 3
    use_geometry: bool = True
    geom_weight: float = 0.5
    geom_missing_score: float = 0.3
    use_kpca: bool = False
    kpca_components: int = 6
    kpca_gamma: Optional[float] = None
    score_floor: float = 0.0
    nms_iou: float = 0.3

    @classmethod
    def synthetic_defaults(cls) -> "ReidemeisterConfig":
        return cls(
            r1_min_cycle_len=6,
            r1_max_cycle_len=400,
            r1_max_junctions=12,
            r2_max_edge_len=300,
            r2_length_ratio=2.5,
            r3_max_perimeter=400,
            r3_edge_ratio=2.0,
            spur_prune_ratio=0.0,
            junction_degree=4,
            use_geometry=False,
            score_floor=0.0,
            nms_iou=1.01,
        )


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
        return self.detect_array(img_np, overlay_path=overlay_path)

    def detect_array(self, img_np: np.ndarray, overlay_path: Optional[str] = None) -> List[Dict[str, Any]]:
        skel, gray = self._prepare_skeleton(img_np)

        candidates = self._collect_candidates(skel)

        candidates = self._postprocess(candidates, gray)

        if overlay_path:
            img = Image.fromarray(img_np)
            self._save_overlay(img, candidates, overlay_path)

        return [asdict(c) for c in candidates]

    def detect_skeleton(self, skel: np.ndarray, gray: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        candidates = self._collect_candidates(skel)
        candidates = self._postprocess(candidates, gray)
        return [asdict(c) for c in candidates]

    def _prepare_skeleton(self, img_np):
        cfg = self.config
        base_skel, gray = self.preprocessor.run(img_np)
        base_h, base_w = base_skel.shape[:2]
        union = base_skel.astype(bool)

        for scale in cfg.scales:
            if abs(scale - 1.0) < 1e-6:
                continue
            scaled_width = max(32, int(cfg.image_width * scale))
            scaled_pre = Preprocessor(PreprocessConfig(width=scaled_width))
            skel, _ = scaled_pre.run(img_np)
            resized = _resize_mask(skel, (base_h, base_w))
            union |= resized

        return union, gray

    def _collect_candidates(self, skel) -> List[MoveCandidate]:
        candidates = []
        candidates.extend(self._detect_r2_r3(skel))
        candidates.extend(self._detect_r1(skel))
        return candidates

    def _postprocess(self, candidates: List[MoveCandidate], gray) -> List[MoveCandidate]:
        if self.config.use_geometry:
            candidates = self._apply_geometry_scores(candidates, gray)
        candidates = self._nms(candidates, iou_thresh=self.config.nms_iou)
        candidates = [c for c in candidates if c.score >= self.config.score_floor]
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _detect_r1(self, skel) -> List[MoveCandidate]:
        cfg = self.config
        G = skeleton_to_graph(skel, connectivity=8)
        spur_len = max(2, int(cfg.spur_prune_ratio * cfg.r1_max_cycle_len))
        G = prune_spurs(G, min_length=spur_len)

        if G.number_of_nodes() > cfg.max_nodes_for_r1:
            return []

        cycles = nx.cycle_basis(G)
        candidates = []
        for cycle in cycles:
            if len(cycle) < cfg.r1_min_cycle_len:
                continue
            if len(cycle) > cfg.r1_max_cycle_len:
                continue
            junction_count = sum(1 for n in cycle if G.degree(n) >= cfg.junction_degree)
            if junction_count > cfg.r1_max_junctions:
                continue
            coords = [G.nodes[n]["pos"] for n in cycle]
            bbox = _bbox_from_coords(coords)
            score = _size_score(len(cycle), cfg.r1_max_cycle_len)
            candidates.append(
                MoveCandidate(
                    move="R1",
                    bbox=bbox,
                    score=score,
                    details={"cycle_len": len(cycle), "size_score": score},
                )
            )
        return candidates

    def _detect_r2_r3(self, skel) -> List[MoveCandidate]:
        cfg = self.config
        G = skeleton_to_graph(skel, connectivity=8)
        spur_len = max(2, int(cfg.spur_prune_ratio * cfg.r2_max_edge_len))
        G = prune_spurs(G, min_length=spur_len)
        junctions, junction_map = cluster_junctions(G, deg_thresh=cfg.junction_degree)
        if not junctions:
            return []
        SG = simplify_graph(G, junctions, junction_map)
        edge_lengths, edge_paths = _edge_stats(SG)

        candidates = []
        candidates.extend(self._r2_candidates(edge_lengths, edge_paths))

        # R3: triangle among three junctions
        junction_nodes = _junction_nodes(SG)
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
                    edge_lens = []
                    for a, b in [(u, v), (v, w), (u, w)]:
                        lengths = edge_lengths.get((a, b)) or edge_lengths.get((b, a))
                        paths_list = edge_paths.get((a, b)) or edge_paths.get((b, a))
                        if not lengths or not paths_list:
                            per = None
                            break
                        idx = int(np.argmin(lengths))
                        edge_lens.append(lengths[idx])
                        per += lengths[idx]
                        paths.append(paths_list[idx])
                    if per is None or per > cfg.r3_max_perimeter:
                        continue
                    if max(edge_lens) / max(1.0, min(edge_lens)) > cfg.r3_edge_ratio:
                        continue

                    bbox = _bbox_from_paths(paths)
                    score = _size_score(per, cfg.r3_max_perimeter)
                    candidates.append(
                        MoveCandidate(
                            move="R3",
                            bbox=bbox,
                            score=score,
                            details={"perimeter": per, "size_score": score},
                        )
                    )

        return candidates

    def _r2_candidates(self, edge_lengths, edge_paths) -> List[MoveCandidate]:
        cfg = self.config
        candidates = []
        for (u, v), lengths in edge_lengths.items():
            if len(lengths) < 2:
                continue
            min_len = min(lengths)
            max_len = max(lengths)
            if min_len <= 0 or (max_len / min_len) > cfg.r2_length_ratio:
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
                    details={
                        "edge_count": len(lengths),
                        "avg_len": avg_len,
                        "size_score": score,
                    },
                )
            )
        return candidates

    def _apply_geometry_scores(self, candidates: List[MoveCandidate], gray) -> List[MoveCandidate]:
        cfg = self.config
        if not candidates:
            return candidates

        by_move: Dict[str, List[MoveCandidate]] = {}
        for cand in candidates:
            by_move.setdefault(cand.move, []).append(cand)

        rescored: List[MoveCandidate] = []
        for move, cands in by_move.items():
            descs = []
            idxs = []
            for i, cand in enumerate(cands):
                desc = _extract_cov_descriptor(gray, cand.bbox)
                if desc is None:
                    continue
                descs.append(desc)
                idxs.append(i)

            if not descs:
                for cand in cands:
                    details = dict(cand.details)
                    details.update({"geom_score": cfg.geom_missing_score, "geom_dist": None})
                    rescored.append(
                        MoveCandidate(
                            move=cand.move,
                            bbox=cand.bbox,
                            score=cand.score * cfg.geom_missing_score,
                            details=details,
                        )
                    )
                continue

            X = np.vstack(descs)
            X = _maybe_kpca(X, cfg)
            proto = np.median(X, axis=0)
            dists = np.linalg.norm(X - proto, axis=1)
            scale = np.median(dists) + 1e-6
            pos_map = {idx: j for j, idx in enumerate(idxs)}

            for i, cand in enumerate(cands):
                if i not in pos_map:
                    geom_score = cfg.geom_missing_score
                    geom_dist = None
                else:
                    xi = X[pos_map[i]]
                    geom_dist = float(np.linalg.norm(xi - proto))
                    geom_score = float(np.exp(-geom_dist / scale))
                size_score = float(cand.details.get("size_score", cand.score))
                final_score = (1.0 - cfg.geom_weight) * size_score + cfg.geom_weight * geom_score
                details = dict(cand.details)
                details.update({"geom_score": geom_score, "geom_dist": geom_dist})
                rescored.append(
                    MoveCandidate(
                        move=cand.move,
                        bbox=cand.bbox,
                        score=final_score,
                        details=details,
                    )
                )

        return rescored

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


def _resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = shape
    resized = cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _junction_nodes(SG) -> List[int]:
    return [n for n, d in SG.nodes(data=True) if d.get("type") == "junction"]


def _edge_stats(SG):
    edge_lengths = {}
    edge_paths = {}
    for u, v, k, data in SG.edges(keys=True, data=True):
        if SG.nodes[u]["type"] != "junction" or SG.nodes[v]["type"] != "junction":
            continue
        path = data.get("path", [])
        length = max(1, len(path))
        edge_lengths.setdefault((u, v), []).append(length)
        edge_paths.setdefault((u, v), []).append(path)
    return edge_lengths, edge_paths


def _extract_cov_descriptor(gray: np.ndarray, bbox: BBox, pad: int = 4, eps: float = 1e-6):
    if gray is None:
        return None
    h, w = gray.shape[:2]
    x, y, bw, bh = bbox
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    x1 = min(w, int(x + bw) + pad)
    y1 = min(h, int(y + bh) + pad)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    patch = gray[y0:y1, x0:x1].astype(np.float32) / 255.0

    gy, gx = np.gradient(patch)
    ph, pw = patch.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw]
    xx = (xx / max(1, pw - 1)) * 2.0 - 1.0
    yy = (yy / max(1, ph - 1)) * 2.0 - 1.0

    feats = np.stack([xx, yy, gx, gy, patch], axis=-1).reshape(-1, 5)
    feats = feats - feats.mean(axis=0, keepdims=True)
    cov = (feats.T @ feats) / max(1, feats.shape[0] - 1)
    cov = cov + eps * np.eye(cov.shape[0])

    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    log_cov = vecs @ np.diag(np.log(vals)) @ vecs.T
    idx = np.triu_indices_from(log_cov)
    return log_cov[idx]


def _maybe_kpca(X: np.ndarray, cfg: ReidemeisterConfig) -> np.ndarray:
    if not cfg.use_kpca or X.shape[0] < max(3, cfg.kpca_components + 1):
        return X
    gamma = cfg.kpca_gamma
    if gamma is None:
        dists = _pairwise_sq_dists(X)
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        gamma = 1.0 / max(1e-6, med)
    K = np.exp(-gamma * _pairwise_sq_dists(X))
    one_n = np.ones((K.shape[0], K.shape[0])) / K.shape[0]
    Kc = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    vals, vecs = np.linalg.eigh(Kc)
    order = np.argsort(vals)[::-1]
    vals = vals[order][: cfg.kpca_components]
    vecs = vecs[:, order][:, : cfg.kpca_components]
    vals = np.clip(vals, 1e-9, None)
    return vecs * np.sqrt(vals)


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    G = X @ X.T
    sq = np.sum(X ** 2, axis=1, keepdims=True)
    return np.maximum(0.0, sq - 2 * G + sq.T)


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
