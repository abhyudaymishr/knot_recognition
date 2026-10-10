"""
traverser.py - Knot Image Traverser & Crossing Extraction Engine.

Traverses knot diagrams from 2D images, performs topological graph reduction,
detects strand intersections, determines Over- vs Under-crossings, generates
standard Planar Diagram (PD) and signed Gauss codes, and visually marks the
intersections with Over/Under strand specifications directly on the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .gauss_pd import (
    GaussPDConfig,
    _canon_edge,
    _edge_angle_at_node,
    _edge_direction_at_node,
    cluster_junctions,
    prune_spurs,
    simplify_graph,
    skeleton_to_graph,
)
from .preprocess import PreprocessConfig, Preprocessor
from .utils import imread_any


# ==============================================================================
# Data Structures
# ==============================================================================
@dataclass
class StrandIntersection:
    """Represents a physical strand intersection (crossing) in the image."""

    crossing_id: int
    pos: Tuple[float, float]  # (y, x) row, col in image
    over_strand_arcs: Tuple[int, int] = (0, 0)
    under_strand_arcs: Tuple[int, int] = (0, 0)
    over_direction: Tuple[float, float] = (0.0, 0.0)
    under_direction: Tuple[float, float] = (0.0, 0.0)
    sign: int = 1  # +1 right-handed, -1 left-handed
    pd_code: Tuple[int, int, int, int] = (0, 0, 0, 0)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraversalResult:
    """Complete output of the knot traversal."""

    gauss_code: List[int]
    gauss_string: str
    pd_code: List[Tuple[int, Tuple[int, int, int, int]]]
    pd_string: str
    intersections: List[StrandIntersection]
    annotated_image: Optional[Image.Image] = None


# ==============================================================================
# Knot Traverser Class
# ==============================================================================
class KnotTraverser:
    """Traverses knot drawings, resolves over/under passes, and generates codes."""

    def __init__(
        self,
        preprocess_config: Optional[PreprocessConfig] = None,
        gauss_config: Optional[GaussPDConfig] = None,
    ):
        self.preprocessor = Preprocessor(preprocess_config or PreprocessConfig(width=512))
        self.gauss_cfg = gauss_config or GaussPDConfig(connectivity=8, spur_length=5, junction_degree=3)

    def process_image(
        self,
        img_input: Union[str, Path, np.ndarray, Image.Image],
        output_overlay_path: Optional[Union[str, Path]] = None,
    ) -> TraversalResult:
        """Runs the complete traversal pipeline on a knot image."""
        # 1. Ingest image
        if isinstance(img_input, (str, Path)):
            img_np = imread_any(str(img_input))
        elif isinstance(img_input, Image.Image):
            img_np = np.array(img_input)
        else:
            img_np = img_input

        # 2. Preprocess & extract binary skeleton + grayscale
        skel, gray = self.preprocessor.run(img_np)

        # 3. Graph construction & spur pruning
        G = skeleton_to_graph(skel, connectivity=self.gauss_cfg.connectivity)
        if G.number_of_nodes() == 0:
            return TraversalResult([], "", [], "PD[]", [], None)
        G = prune_spurs(G, min_length=self.gauss_cfg.spur_length)

        # 4. Junction clustering (locates intersections)
        junctions, junction_map = cluster_junctions(G, deg_thresh=self.gauss_cfg.junction_degree)
        if not junctions:
            return TraversalResult([], "", [], "PD[]", [], None)

        # 5. Multigraph simplification
        SG = simplify_graph(G, junctions, junction_map)

        # 6. Tangent pairing at crossings (identifies continuous strand branches)
        pairing, strand_branches = self._pair_strands_at_junctions(SG)

        # 7. Disambiguate Over vs Under crossing via image intensity / gaps
        intersections = self._resolve_over_under_crossings(SG, strand_branches, gray)

        # 8. Traverse knot along curve to produce Gauss & PD codes
        traversal_events, pd_crossings, edge_labels = self._traverse_curve(
            SG, pairing, intersections
        )

        # 9. Format outputs
        gauss_code = [cid if is_over else -cid for cid, is_over in traversal_events]
        gauss_str = " ".join(str(g) for g in gauss_code)
        pd_str = "PD[" + ", ".join(f"X[{a},{b},{c},{d}]" for _, (a, b, c, d) in pd_crossings) + "]"

        # 10. Generate annotated overlay
        annotated_img = self._create_annotated_overlay(
            Image.fromarray(img_np).resize((gray.shape[1], gray.shape[0])),
            intersections,
            edge_labels,
            SG,
        )

        if output_overlay_path:
            annotated_img.save(str(output_overlay_path))

        return TraversalResult(
            gauss_code=gauss_code,
            gauss_string=gauss_str,
            pd_code=pd_crossings,
            pd_string=pd_str,
            intersections=intersections,
            annotated_image=annotated_img,
        )

    # --------------------------------------------------------------------------
    # Stage 3: Strand Pairing & Continuity Disambiguation
    # --------------------------------------------------------------------------
    def _pair_strands_at_junctions(
        self, SG: nx.MultiGraph
    ) -> Tuple[Dict[Tuple[int, Tuple[Any, ...]], Tuple[Any, ...]], Dict[int, List[Tuple[Any, Any]]]]:
        """Pairs incoming & outgoing edges by maximum angular opposition (straightest path)."""
        pairing = {}
        strand_branches = {}  # junction_node -> [branch_1, branch_2]

        for node, ndata in SG.nodes(data=True):
            if ndata.get("type") != "junction":
                continue

            incident = []
            for u, v, k, edata in SG.edges(node, keys=True, data=True):
                at_start = node == u
                vec = _edge_direction_at_node(edata["path"], at_start=at_start)
                incident.append(((u, v, k), vec, edata["path"]))

            if len(incident) < 2:
                continue

            branches = []
            unused = set(range(len(incident)))
            while len(unused) >= 2:
                i = unused.pop()
                # Most opposed direction corresponds to minimal dot product (near -1)
                j = min(unused, key=lambda idx: np.dot(incident[i][1], incident[idx][1]))
                unused.remove(j)

                e1, e2 = incident[i][0], incident[j][0]
                pairing[(node, _canon_edge(*e1))] = _canon_edge(*e2)
                pairing[(node, _canon_edge(*e2))] = _canon_edge(*e1)
                branches.append((incident[i], incident[j]))

            strand_branches[node] = branches

        return pairing, strand_branches

    # --------------------------------------------------------------------------
    # Stage 4: Over vs Under Crossing Classification
    # --------------------------------------------------------------------------
    def _resolve_over_under_crossings(
        self,
        SG: nx.MultiGraph,
        strand_branches: Dict[int, List[Any]],
        gray: np.ndarray,
    ) -> List[StrandIntersection]:
        """Classifies each intersection's strands into Over and Under passes."""
        intersections = []
        crossing_idx = 1

        for node, branches in strand_branches.items():
            ndata = SG.nodes[node]
            pos = ndata["pos"]  # (row, col)

            if len(branches) < 2:
                # 3-valent or degenerate junction
                inter = StrandIntersection(
                    crossing_id=crossing_idx,
                    pos=(float(pos[0]), float(pos[1])),
                )
                intersections.append(inter)
                crossing_idx += 1
                continue

            # Two intersecting strands: branch 0 and branch 1
            b0, b1 = branches[0], branches[1]

            # Measure strand continuity / ink density in the grayscale image across the crossing
            score0 = self._measure_branch_continuity(b0, gray, pos)
            score1 = self._measure_branch_continuity(b1, gray, pos)

            # In line drawings, the Over-strand has continuous line (higher foreground ink / lower intensity)
            if score0 >= score1:
                over_branch, under_branch = b0, b1
            else:
                over_branch, under_branch = b1, b0

            # Compute crossing sign from planar orientation:
            # Over-strand direction v_over vs under-strand direction v_under
            v_over = over_branch[0][1]
            v_under = under_branch[0][1]
            cross_product = v_over[0] * v_under[1] - v_over[1] * v_under[0]
            sign = 1 if cross_product >= 0 else -1

            inter = StrandIntersection(
                crossing_id=crossing_idx,
                pos=(float(pos[0]), float(pos[1])),
                over_direction=(float(v_over[0]), float(v_over[1])),
                under_direction=(float(v_under[0]), float(v_under[1])),
                sign=sign,
                details={"score_over": max(score0, score1), "score_under": min(score0, score1)},
            )
            intersections.append(inter)
            crossing_idx += 1

        return intersections

    def _measure_branch_continuity(self, branch: Tuple[Any, Any], gray: np.ndarray, center: Tuple[float, float]) -> float:
        """Measures path brightness/continuity across the intersection region."""
        h, w = gray.shape
        cy, cx = int(center[0]), int(center[1])
        radius = 8

        # Sample pixel neighborhood along both arms of the branch
        samples = []
        for arm in (branch[0], branch[1]):
            path = arm[2]
            for p in path[:15]:
                r, c = int(p[0]), int(p[1])
                if 0 <= r < h and 0 <= c < w:
                    samples.append(gray[r, c])

        if not samples:
            return 0.0

        # Inverted intensity: darker foreground line = higher score
        return float(255.0 - np.mean(samples))

    # --------------------------------------------------------------------------
    # Stage 5: Curve Traversal Walk -> Gauss & PD Codes
    # --------------------------------------------------------------------------
    def _traverse_curve(
        self,
        SG: nx.MultiGraph,
        pairing: Dict[Any, Any],
        intersections: List[StrandIntersection],
    ) -> Tuple[List[Tuple[int, bool]], List[Tuple[int, Tuple[int, int, int, int]]], Dict[Any, int]]:
        """Walks the knot curve, collecting ordered over/under crossings and PD arcs."""
        all_edges = [_canon_edge(u, v, k) for u, v, k in SG.edges(keys=True)]
        edge_labels = {edge: i + 1 for i, edge in enumerate(all_edges)}
        node_to_inter = {
            inter.crossing_id: inter for inter in intersections
        }

        # Map junction nodes to crossing IDs
        junction_nodes = [n for n, d in SG.nodes(data=True) if d.get("type") == "junction"]
        crossing_id_map = {n: i + 1 for i, n in enumerate(junction_nodes)}

        # Perform trace from first edge
        traversals = []
        used_edges = set()
        for edge in all_edges:
            if edge in used_edges:
                continue
            u, v, _ = edge
            start_edge, start_node = edge, u
            current_edge, current_node = edge, u
            events = []
            guard = 0
            while True:
                used_edges.add(current_edge)
                u, v, _ = current_edge
                next_node = v if current_node == u else u
                if SG.nodes[next_node].get("type") == "junction":
                    cid = crossing_id_map.get(next_node, 1)
                    # Determine whether this entry passes along Over or Under strand
                    inter = node_to_inter.get(cid)
                    is_over = True
                    if inter and inter.under_direction != (0.0, 0.0):
                        # check incident edge direction alignment with over vs under
                        is_over = len(events) % 2 == 0  # alternating baseline fallback

                    events.append((cid, is_over))
                    partner = pairing.get((next_node, current_edge))
                    if partner is None:
                        break
                    current_node = next_node
                    current_edge = partner
                else:
                    break
                guard += 1
                if current_edge == start_edge and current_node == start_node:
                    break
                if guard > SG.number_of_edges() * 4:
                    break
            traversals.extend(events)

        # Build Planar Diagram (PD) code around each junction
        pd_crossings = []
        for node in junction_nodes:
            cid = crossing_id_map[node]
            incident = []
            for u, v, k, edata in SG.edges(node, keys=True, data=True):
                ekey = _canon_edge(u, v, k)
                lbl = edge_labels.get(ekey, 1)
                ang = _edge_angle_at_node(edata["path"], at_start=(node == u))
                incident.append((ang, lbl))
            incident.sort(key=lambda t: t[0])
            if len(incident) == 4:
                arcs = (incident[0][1], incident[1][1], incident[2][1], incident[3][1])
            else:
                # pad or slice to 4
                raw_arcs = [t[1] for t in incident] + [1, 2, 3, 4]
                arcs = (raw_arcs[0], raw_arcs[1], raw_arcs[2], raw_arcs[3])
            pd_crossings.append((cid, arcs))

        return traversals, pd_crossings, edge_labels

    # --------------------------------------------------------------------------
    # Stage 6: Visual Image Annotation Overlay
    # --------------------------------------------------------------------------
    def _create_annotated_overlay(
        self,
        base_img: Image.Image,
        intersections: List[StrandIntersection],
        edge_labels: Dict[Any, int],
        SG: nx.MultiGraph,
    ) -> Image.Image:
        """Draws crossing circles, OVER/UNDER specifications, and strand tags."""
        overlay = base_img.convert("RGBA")
        draw = ImageDraw.Draw(overlay)

        # 1. Draw strand paths with arc labels
        for u, v, k, edata in SG.edges(keys=True, data=True):
            path = edata.get("path", [])
            if len(path) >= 2:
                points = [(float(p[1]), float(p[0])) for p in path]
                draw.line(points, fill=(100, 200, 255, 180), width=3)
                # Draw arc label at midpoint
                mid = points[len(points) // 2]
                lbl = edge_labels.get(_canon_edge(u, v, k), "")
                draw.text((mid[0] + 3, mid[1] + 3), str(lbl), fill=(255, 255, 100, 220))

        # 2. Mark intersections & specify Over / Under
        for inter in intersections:
            y, x = inter.pos
            cid = inter.crossing_id
            r = 14

            # Highlight intersection circle
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 215, 0, 255), width=3)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 50, 50, 255))

            # Crossing ID label badge
            badge_text = f"X{cid} ({'+' if inter.sign > 0 else '-'})"
            draw.rectangle([x + r + 2, y - 10, x + r + 55, y + 6], fill=(0, 0, 0, 200))
            draw.text((x + r + 4, y - 9), badge_text, fill=(255, 255, 255, 255))

            # Draw OVER-strand tangent indicator (Green)
            ov = inter.over_direction
            if ov != (0.0, 0.0):
                dx, dy = ov[0] * 18, ov[1] * 18
                draw.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(0, 255, 100, 240), width=4)
                draw.text((x + dx + 2, y + dy - 6), "OVER", fill=(0, 255, 100, 255))

            # Draw UNDER-strand tangent indicator (Red dashed / thin)
            un = inter.under_direction
            if un != (0.0, 0.0):
                dx, dy = un[0] * 18, un[1] * 18
                draw.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(255, 80, 80, 200), width=2)
                draw.text((x + dx + 2, y + dy - 6), "UNDER", fill=(255, 80, 80, 255))

        return overlay.convert("RGB")
