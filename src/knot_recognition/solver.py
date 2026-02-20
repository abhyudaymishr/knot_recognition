import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import networkx as nx
from PIL import Image

from .gauss_pd import (
    GaussPDConfig,
    GaussPDExtractor,
    cluster_junctions,
    prune_spurs,
    simplify_graph,
    skeleton_to_graph,
)
from .infer import InferenceConfig, KnotRecognizer
from .preprocess import PreprocessConfig, Preprocessor
from .utils import imread_any


@dataclass(frozen=True)
class SolverConfig:
    image_width: int = 512
    connectivity: int = 4
    spur_length: int = 5
    junction_degree: int = 3
    max_iters: int = 3
    enable_r1: bool = True
    enable_r2: bool = True
    r1_min_cycle_len: int = 6
    r1_max_cycle_len: int = 200
    r1_max_junctions: int = 4
    r2_max_edge_len: int = 140
    r2_length_ratio: float = 1.6


def solve_image(
    img_path: str,
    checkpoint: Optional[str] = None,
    mapping_csv: Optional[str] = None,
    *,
    config: Optional[SolverConfig] = None,
    device: Optional[str] = None,
    save_reduced: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = config or SolverConfig()
    img = Image.fromarray(imread_any(img_path))
    preprocessor = Preprocessor(PreprocessConfig(width=cfg.image_width))
    skel, _ = preprocessor.run(np.array(img))

    reduced_skel, stats = reduce_skeleton(skel, cfg)
    gauss, pd = GaussPDExtractor(GaussPDConfig()).extract(reduced_skel)

    if save_reduced:
        reduced_img = Image.fromarray((reduced_skel.astype("uint8") * 255))
        reduced_img.save(save_reduced)

    result = {
        "auto_gauss": gauss,
        "auto_pd": pd,
        "reduction": stats,
    }

    if checkpoint:
        recognizer = KnotRecognizer.from_checkpoint(
            checkpoint,
            device=device,
            config=InferenceConfig(),
        )
        pred = recognizer.predict_image(img, mapping_csv=mapping_csv)
        pred["auto_gauss"] = gauss
        pred["auto_pd"] = pd
        pred["reduction"] = stats
        return pred

    return result


def reduce_skeleton(skel: np.ndarray, cfg: SolverConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    G = skeleton_to_graph(skel, connectivity=cfg.connectivity)
    crossings_before = _count_crossings(G, cfg.junction_degree)
    r1_removed_total = 0
    r2_removed_total = 0
    iters = 0

    for i in range(cfg.max_iters):
        iters = i + 1
        changed = False
        if cfg.enable_r1:
            r1_removed = _reduce_r1(G, cfg)
            if r1_removed:
                r1_removed_total += r1_removed
                changed = True
        if cfg.enable_r2:
            r2_removed = _reduce_r2(G, cfg)
            if r2_removed:
                r2_removed_total += r2_removed
                changed = True
        if changed:
            G = prune_spurs(G, min_length=cfg.spur_length)
        else:
            break

    reduced = _graph_to_skeleton(G, skel.shape)
    crossings_after = _count_crossings(G, cfg.junction_degree)

    stats = {
        "iterations": iters,
        "r1_removed": r1_removed_total,
        "r2_removed": r2_removed_total,
        "crossings_before": crossings_before,
        "crossings_after": crossings_after,
        "nodes_before": int(skel.sum()),
        "nodes_after": int(reduced.sum()),
    }
    return reduced, stats


def _reduce_r1(G: nx.Graph, cfg: SolverConfig) -> int:
    cycles = nx.cycle_basis(G)
    removed_cycles = 0
    nodes_to_remove = set()
    for cycle in cycles:
        if len(cycle) < cfg.r1_min_cycle_len or len(cycle) > cfg.r1_max_cycle_len:
            continue
        junction_count = sum(1 for n in cycle if G.degree(n) >= cfg.junction_degree)
        if junction_count > cfg.r1_max_junctions:
            continue
        cycle_set = set(cycle)
        attachments = {
            n
            for n in cycle
            if any(nb not in cycle_set for nb in G.neighbors(n))
        }
        for n in cycle:
            if n not in attachments:
                nodes_to_remove.add(n)
        if nodes_to_remove:
            removed_cycles += 1

    if nodes_to_remove:
        G.remove_nodes_from(nodes_to_remove)
    return removed_cycles


def _reduce_r2(G: nx.Graph, cfg: SolverConfig) -> int:
    junctions, junction_map = cluster_junctions(G, deg_thresh=cfg.junction_degree)
    if not junctions:
        return 0
    SG = simplify_graph(G, junctions, junction_map)
    pos2id = {G.nodes[n]["pos"]: n for n in G.nodes}

    edges_by_pair = {}
    for u, v, k, edata in SG.edges(keys=True, data=True):
        pair = tuple(sorted((u, v)))
        edges_by_pair.setdefault(pair, []).append(edata)

    removed = 0
    nodes_to_remove = set()
    for pair, edges in edges_by_pair.items():
        if len(edges) < 2:
            continue
        lengths = [len(ed["path"]) for ed in edges]
        min_len = min(lengths)
        max_len = max(lengths)
        if min_len > cfg.r2_max_edge_len:
            continue
        if max_len / max(1.0, min_len) > cfg.r2_length_ratio:
            continue
        keep_idx = lengths.index(min_len)
        for idx, ed in enumerate(edges):
            if idx == keep_idx:
                continue
            path = ed["path"]
            for pos in path[1:-1]:
                node_id = pos2id.get(tuple(pos))
                if node_id is not None:
                    nodes_to_remove.add(node_id)
            removed += 1

    if nodes_to_remove:
        G.remove_nodes_from(nodes_to_remove)
    return removed


def _count_crossings(G: nx.Graph, junction_degree: int) -> int:
    return sum(1 for _, d in G.degree() if d >= junction_degree)


def _graph_to_skeleton(G: nx.Graph, shape: Tuple[int, int]) -> np.ndarray:
    skel = np.zeros(shape, dtype=bool)
    for _, data in G.nodes(data=True):
        r, c = data["pos"]
        skel[r, c] = True
    return skel


def _json_safe(obj):
    if isinstance(obj, dict):
        safe = {}
        for k, v in obj.items():
            if not isinstance(k, (str, int, float, bool)) and k is not None:
                k = str(k)
            safe[k] = _json_safe(v)
        return safe
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reduced-skeleton", default=None)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--no-r1", action="store_true")
    parser.add_argument("--no-r2", action="store_true")
    parser.add_argument("--r1-max", type=int, default=200)
    parser.add_argument("--r2-max", type=int, default=140)
    args = parser.parse_args()

    cfg = SolverConfig(
        max_iters=args.max_iters,
        enable_r1=not args.no_r1,
        enable_r2=not args.no_r2,
        r1_max_cycle_len=args.r1_max,
        r2_max_edge_len=args.r2_max,
    )

    res = solve_image(
        args.image,
        checkpoint=args.checkpoint,
        mapping_csv=args.mapping,
        config=cfg,
        device=args.device,
        save_reduced=args.reduced_skeleton,
    )

    import json

    print(json.dumps(_json_safe(res), indent=2, default=str))


if __name__ == "__main__":
    main()
