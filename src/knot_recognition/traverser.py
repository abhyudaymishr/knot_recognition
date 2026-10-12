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
class AnnotatedImage(np.ndarray):
    """Numpy ndarray subclass representing an annotated image overlay.

    Seamlessly compatible with both OpenCV (e.g. `cv2.imwrite("out.png", img)`)
    and PIL-style workflows (e.g. `img.save("out.png")`, `img.to_pil()`).
    """

    def __new__(cls, input_array: np.ndarray):
        obj = np.asarray(input_array).view(cls)
        return obj

    def save(self, fp: Union[str, Path], *args: Any, **kwargs: Any) -> None:
        """Saves the annotated overlay using PIL, converting BGR to RGB automatically."""
        if self.ndim == 3 and self.shape[2] == 3:
            rgb = cv2.cvtColor(self, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
        else:
            img_pil = Image.fromarray(self)
        img_pil.save(str(fp), *args, **kwargs)

    def to_pil(self) -> Image.Image:
        """Converts to a PIL Image (RGB format)."""
        if self.ndim == 3 and self.shape[2] == 3:
            rgb = cv2.cvtColor(self, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        return Image.fromarray(self)


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
    # Classification is tri-state: uncertain candidates must never be silently
    # promoted to over/under crossings or included in a trusted Gauss code.
    status: str = "resolved"  # resolved | ambiguous | rejected
    confidence: float = 1.0
    junction_node: Optional[int] = None
    over_edges: Tuple[Tuple[Any, ...], Tuple[Any, ...]] = ()
    under_edges: Tuple[Tuple[Any, ...], Tuple[Any, ...]] = ()
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraversalResult:
    """Complete output of the knot traversal."""

    gauss_code: List[int]
    gauss_string: str
    pd_code: List[Tuple[int, Tuple[int, int, int, int]]]
    pd_string: str
    intersections: List[StrandIntersection]
    annotated_image: Optional[AnnotatedImage] = None
    warnings: List[str] = field(default_factory=list)
    classification: Optional[Dict[str, Any]] = None
    similar_knots: List[Dict[str, Any]] = field(default_factory=list)
    consensus: Optional[Dict[str, Any]] = None


# ==============================================================================
# Knot Traverser Class
# ==============================================================================
class KnotTraverser:
    """Traverses knot drawings, resolves over/under passes, and generates codes."""

    def __init__(
        self,
        preprocess_config: Optional[PreprocessConfig] = None,
        gauss_config: Optional[GaussPDConfig] = None,
        cluster_radius: float = 12.0,
        sample_radius: int = 18,
        nanotransformer: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.cluster_radius = float(cluster_radius)
        self.sample_radius = int(sample_radius)
        self.nanotransformer = nanotransformer
        self.preprocessor = Preprocessor(preprocess_config or PreprocessConfig(width=512))
        self.gauss_cfg = gauss_config or GaussPDConfig(
            connectivity=kwargs.get("connectivity", 8),
            spur_length=kwargs.get("spur_length", 5),
            junction_degree=kwargs.get("junction_degree", 3),
        )

    def traverse_and_classify(
        self,
        img_input: Union[str, Path, np.ndarray, Image.Image],
        nanotransformer: Optional[Any] = None,
        output_overlay_path: Optional[Union[str, Path]] = None,
        top_k: int = 5,
    ) -> TraversalResult:
        """Runs diagram traversal empowered by NanoTransformer classification & similar knot retrieval."""
        result = self.process_image(img_input, output_overlay_path=output_overlay_path, _skip_model=True)
        model = nanotransformer or self.nanotransformer
        if model is None:
            return result

        try:
            pred_res = model.predict_image(img_input, top_k=top_k)
            similar = model.fetch_similar_knots(pred_res["embedding"], top_k=top_k)
            result.classification = pred_res
            result.similar_knots = similar

            resolved_crossings = [i for i in result.intersections if i.status == "resolved"]
            n_crossings_traverser = len(resolved_crossings)
            top_pred = pred_res.get("predictions", [{}])[0]
            top_label = top_pred.get("label", "unknown")
            top_crossings = top_pred.get("crossings", 0)

            consensus_match = (n_crossings_traverser == top_crossings)
            crossing_diff = n_crossings_traverser - top_crossings
            notes = []
            if consensus_match:
                notes.append(
                    f"Crossing count agreement: traverser detected {n_crossings_traverser} crossings "
                    f"matching predicted knot {top_label}."
                )
            else:
                notes.append(
                    f"Crossing divergence: traverser detected {n_crossings_traverser} crossings, "
                    f"whereas top model prediction '{top_label}' expects {top_crossings} crossings (delta: {crossing_diff:+d})."
                )

            result.consensus = {
                "traverser_crossings": n_crossings_traverser,
                "predicted_label": top_label,
                "predicted_crossings": top_crossings,
                "agreement": consensus_match,
                "confidence": float(top_pred.get("probability", 0.0)),
                "notes": notes,
            }
        except Exception as exc:
            result.warnings.append(f"NanoTransformer classification failed: {exc}")

        return result

    def process_image(
        self,
        img_input: Union[str, Path, np.ndarray, Image.Image],
        output_overlay_path: Optional[Union[str, Path]] = None,
        _skip_model: bool = False,
    ) -> TraversalResult:
        """Runs the complete traversal pipeline on a knot image."""
        if not _skip_model and self.nanotransformer is not None:
            return self.traverse_and_classify(
                img_input,
                nanotransformer=self.nanotransformer,
                output_overlay_path=output_overlay_path,
            )

        # 1. Ingest image
        if img_input is None:
            raise ValueError(
                "Input image is None or could not be loaded. "
                "Ensure cv2.imread(...) loaded a valid image file."
            )
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
            blank = AnnotatedImage(img_np)
            return TraversalResult([], "", [], "PD[]", [], blank, ["No skeleton pixels were extracted; crossing absence is not verified."])
        G = prune_spurs(G, min_length=self.gauss_cfg.spur_length)

        # 4. Junction clustering (locates intersections)
        junctions, junction_map = cluster_junctions(
            G,
            deg_thresh=self.gauss_cfg.junction_degree,
            cluster_radius=self.cluster_radius,
        )
        if not junctions:
            annotated_img = self._create_annotated_overlay(
                Image.fromarray(img_np).resize((gray.shape[1], gray.shape[0])),
                [],
                {},
                nx.MultiGraph(),
            )
            return TraversalResult(
                [], "", [], "PD[]", [], annotated_img,
                ["No junction candidates were found. This does not establish that the image contains no crossing."],
            )

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
            warnings=[
                f"{sum(i.status == 'ambiguous' for i in intersections)} ambiguous and "
                f"{sum(i.status == 'rejected' for i in intersections)} rejected junction candidate(s) "
                "were excluded from Gauss/PD codes."
            ] if any(i.status != "resolved" for i in intersections) else [],
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
        """Classifies crossings only when local image evidence is decisive.

        A raster crossing with a weak or contradictory gap signal is returned as
        ``ambiguous``. Callers can inspect/confirm it; we do not guess a label.
        """
        intersections = []
        crossing_idx = 1

        for node, branches in strand_branches.items():
            ndata = SG.nodes[node]
            pos = ndata["pos"]  # (row, col)

            if len(branches) != 2 or any(len(branch) != 2 for branch in branches):
                # A standard crossing has exactly four arms, paired into two
                # continuous strands. Higher-valence skeleton artifacts are not
                # evidence of a crossing and are explicitly rejected.
                inter = StrandIntersection(
                    crossing_id=crossing_idx,
                    pos=(float(pos[0]), float(pos[1])),
                    status="rejected",
                    confidence=0.0,
                    junction_node=node,
                    details={"reason": "expected_two_strand_pairs", "pair_count": len(branches)},
                )
                intersections.append(inter)
                crossing_idx += 1
                continue

            # Two intersecting strands: branch 0 and branch 1
            b0, b1 = branches[0], branches[1]

            # Require two locally straight continuations which cross at a
            # meaningful angle. Degree-four skeleton blobs are common around
            # thick strokes and must not automatically become knot crossings.
            v0 = np.asarray(b0[0][1], dtype=float)
            v1 = np.asarray(b0[1][1], dtype=float)
            v2 = np.asarray(b1[0][1], dtype=float)
            v3 = np.asarray(b1[1][1], dtype=float)
            straightness = [float(np.dot(v0, v1)), float(np.dot(v2, v3))]
            crossing_angle = abs(float(v0[0] * v2[1] - v0[1] * v2[0]))
            if max(straightness) > -0.70 or crossing_angle < 0.25:
                intersections.append(StrandIntersection(
                    crossing_id=crossing_idx,
                    pos=(float(pos[0]), float(pos[1])),
                    status="rejected",
                    confidence=0.0,
                    junction_node=node,
                    details={
                        "reason": "junction_geometry_not_crossing_like",
                        "strand_straightness": straightness,
                        "crossing_angle_sine": crossing_angle,
                    },
                ))
                crossing_idx += 1
                continue

            # Measure strand continuity / ink density in the grayscale image across the crossing
            score0 = self._measure_branch_continuity(b0, gray, pos)
            score1 = self._measure_branch_continuity(b1, gray, pos)
            separation = abs(score0 - score1)
            # Ink continuity is only useful if one candidate is visibly more
            # continuous. Normalize the margin by available score range to make
            # the threshold robust to grayscale/contrast changes.
            # This is a normalized evidence margin, not a calibrated probability.
            confidence = float(np.clip(separation / max(32.0, max(score0, score1)), 0.0, 1.0))
            resolved = min(score0, score1) >= 24.0 and separation >= 28.0 and confidence >= 0.28
            over_branch, under_branch = (b0, b1) if score0 >= score1 else (b1, b0)

            # Compute crossing sign from planar orientation:
            # Over-strand direction v_over vs under-strand direction v_under
            v_over = over_branch[0][1]
            v_under = under_branch[0][1]
            cross_product = v_over[0] * v_under[1] - v_over[1] * v_under[0]
            sign = 1 if cross_product >= 0 else -1

            inter = StrandIntersection(
                crossing_id=crossing_idx,
                pos=(float(pos[0]), float(pos[1])),
                over_direction=(float(v_over[0]), float(v_over[1])) if resolved else (0.0, 0.0),
                under_direction=(float(v_under[0]), float(v_under[1])) if resolved else (0.0, 0.0),
                sign=sign,
                status="resolved" if resolved else "ambiguous",
                confidence=confidence if resolved else min(confidence, 0.49),
                junction_node=node,
                over_edges=tuple(_canon_edge(*arm[0]) for arm in over_branch),
                under_edges=tuple(_canon_edge(*arm[0]) for arm in under_branch),
                details={
                    "continuity_scores": [float(score0), float(score1)],
                    "score_margin": float(separation),
                    "reason": "continuity_margin" if resolved else "insufficient_continuity_margin",
                },
            )
            intersections.append(inter)
            crossing_idx += 1

        return intersections

    def _measure_branch_continuity(self, branch: Tuple[Any, Any], gray: np.ndarray, center: Tuple[float, float]) -> float:
        """Estimate ink continuity through the crossing along one paired strand.

        Sample the expected tangent line on both sides of the center. Sampling
        the existing graph path alone is insufficient: both broken strands have
        dark arms, while the over-strand is distinguished by continuity across
        the small under-strand gap.
        """
        h, w = gray.shape
        cy, cx = float(center[0]), float(center[1])
        vec = np.asarray(branch[0][1], dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return 0.0
        vec /= norm
        normal = np.array([-vec[1], vec[0]])
        values = []
        # Exclude the center itself: an over-strand can make the center dark
        # even when the candidate under-strand has a gap there.
        for distance in np.linspace(3.0, max(4.0, float(self.sample_radius)), max(8, self.sample_radius - 2)):
            for direction in (-1.0, 1.0):
                for offset in (-1.0, 0.0, 1.0):
                    x = int(round(cx + direction * distance * vec[0] + offset * normal[0]))
                    y = int(round(cy + direction * distance * vec[1] + offset * normal[1]))
                    if 0 <= y < h and 0 <= x < w:
                        values.append(255.0 - float(gray[y, x]))
        return float(np.mean(values)) if values else 0.0

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
        # Map only confirmed crossings into topological codes. Candidate and
        # rejected junctions remain visible in the review output, but cannot
        # contaminate the Gauss/PD result.
        node_to_inter = {
            inter.junction_node: inter for inter in intersections
            if inter.junction_node is not None and inter.status == "resolved"
        }
        crossing_id_map = {node: inter.crossing_id for node, inter in node_to_inter.items()}
        junction_nodes = [n for n in crossing_id_map if n in SG]

        # Trace each strand from a free end where possible. Starting each
        # canonical edge at its arbitrary stored endpoint can walk away from a
        # crossing and lose its event entirely.
        traversals = []
        used_edges = set()
        junction_set = set(junction_nodes)
        ordered_edges = sorted(
            all_edges,
            key=lambda e: (int((e[0] in junction_set) + (e[1] in junction_set)), e),
        )
        for edge in ordered_edges:
            if edge in used_edges:
                continue
            u, v, _ = edge
            if (u in junction_set) != (v in junction_set):
                start_node = v if u in junction_set else u
            else:
                start_node = u
            start_state = (edge, start_node)
            current_edge, current_node = edge, start_node
            events = []
            guard = 0
            while True:
                state = (current_edge, current_node)
                if state != start_state and current_edge in used_edges:
                    break
                used_edges.add(current_edge)
                u, v, _ = current_edge
                next_node = v if current_node == u else u
                if SG.nodes[next_node].get("type") == "junction":
                    inter = node_to_inter.get(next_node)
                    if inter is not None:
                        cid = inter.crossing_id
                        # The classification is tied to the actual paired edges
                        # at this junction, never to event order or alternation.
                        is_over = current_edge in inter.over_edges
                        is_under = current_edge in inter.under_edges
                        if is_over or is_under:
                            events.append((cid, is_over))
                    partner = pairing.get((next_node, current_edge))
                    if partner is None:
                        break
                    if partner in used_edges and (partner, next_node) != start_state:
                        break
                    current_node = next_node
                    current_edge = partner
                else:
                    break
                guard += 1
                if (current_edge, current_node) == start_state:
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
    ) -> AnnotatedImage:
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
            r = max(10, int(self.sample_radius * 0.75))

            # Highlight intersection circle
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 215, 0, 255), width=3)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 50, 50, 255))

            # Do not visually present an uncertain candidate as a resolved
            # signed crossing.
            if inter.status == "resolved":
                badge_text = f"X{cid} ({'+' if inter.sign > 0 else '-'}) {inter.confidence:.2f}"
                badge_color = (0, 80, 0, 230)
            else:
                badge_text = f"X{cid} {inter.status.upper()}"
                badge_color = (120, 75, 0, 230) if inter.status == "ambiguous" else (90, 0, 0, 220)
            draw.rectangle([x + r + 2, y - 10, x + r + 110, y + 6], fill=badge_color)
            draw.text((x + r + 4, y - 9), badge_text, fill=(255, 255, 255, 255))

            # Draw OVER-strand tangent indicator (Green)
            ov = inter.over_direction
            if ov != (0.0, 0.0):
                dx, dy = ov[0] * self.sample_radius, ov[1] * self.sample_radius
                draw.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(0, 255, 100, 240), width=4)
                draw.text((x + dx + 2, y + dy - 6), "OVER", fill=(0, 255, 100, 255))

            # Draw UNDER-strand tangent indicator (Red dashed / thin)
            un = inter.under_direction
            if un != (0.0, 0.0):
                dx, dy = un[0] * self.sample_radius, un[1] * self.sample_radius
                draw.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(255, 80, 80, 200), width=2)
                draw.text((x + dx + 2, y + dy - 6), "UNDER", fill=(255, 80, 80, 255))

        rgb_np = np.array(overlay.convert("RGB"))
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        return AnnotatedImage(bgr_np)
