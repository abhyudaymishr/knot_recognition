"""
Real-time knot detection and Reidemeister move prediction.

Usage:
    python -m knot_recognition.realtime --source 0 --checkpoint ./checkpoints/best.pth --moves
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .infer import InferenceConfig, KnotRecognizer
from .reidemeister import ReidemeisterConfig, ReidemeisterDetector


@dataclass
class RealTimeConfig:
    source: str = "0"
    frame_width: int = 640
    show: bool = True
    moves: bool = False
    move_interval: int = 5
    classify_interval: int = 5
    checkpoint: Optional[str] = None
    mapping: Optional[str] = None
    window_name: str = "knot-rt"
    roi_enabled: bool = True
    roi_padding: int = 12
    roi_min_area_ratio: float = 0.02
    show_roi: bool = False
    min_persist: int = 3
    track_max_age: int = 5
    match_iou: float = 0.3
    move_score_floor: float = 0.35
    move_geom_weight: float = 0.7


class RealTimeKnotApp:
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.cap = self._open_capture(config.source)
        if config.frame_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)

        self.recognizer = None
        if config.checkpoint:
            self.recognizer = KnotRecognizer.from_checkpoint(
                config.checkpoint, config=InferenceConfig()
            )
        self.detector = self._build_detector() if config.moves else None

        self.last_pred = None
        self.last_moves = []
        self.tracks = []
        self.last_roi = None

    def _open_capture(self, source: str):
        if source.isdigit():
            return cv2.VideoCapture(int(source))
        return cv2.VideoCapture(source)

    def _build_detector(self):
        cfg = self.config
        move_cfg = ReidemeisterConfig(
            use_geometry=True,
            use_kpca=False,
            geom_weight=cfg.move_geom_weight,
            score_floor=cfg.move_score_floor,
            spur_prune_ratio=0.4,
            r1_max_junctions=1,
            r2_length_ratio=1.2,
            r3_edge_ratio=1.2,
        )
        return ReidemeisterDetector(move_cfg)

    def run(self):
        frame_idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            if self.recognizer and frame_idx % max(1, self.config.classify_interval) == 0:
                self.last_pred = self.recognizer.predict_image(pil_img, mapping_csv=self.config.mapping)

            if self.detector and frame_idx % max(1, self.config.move_interval) == 0:
                candidates = self._detect_moves(rgb)
                self._update_tracks(candidates)

            self._draw_overlay(frame)

            if self.config.show:
                cv2.imshow(self.config.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_idx += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def _detect_moves(self, rgb: np.ndarray):
        roi = self._find_roi(rgb) if self.config.roi_enabled else None
        self.last_roi = roi
        if roi is None:
            return []
        x, y, w, h = roi
        crop = rgb[y : y + h, x : x + w]
        moves = self.detector.detect_array(crop)
        for m in moves:
            bx, by, bw, bh = m["bbox"]
            m["bbox"] = (bx + x, by + y, bw, bh)
        return moves

    def _find_roi(self, rgb: np.ndarray):
        cfg = self.config
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        h, w = gray.shape[:2]
        frame_area = float(h * w)
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area / frame_area < cfg.roi_min_area_ratio:
            return None
        x, y, bw, bh = cv2.boundingRect(best)
        x = max(0, x - cfg.roi_padding)
        y = max(0, y - cfg.roi_padding)
        bw = min(w - x, bw + 2 * cfg.roi_padding)
        bh = min(h - y, bh + 2 * cfg.roi_padding)
        return (x, y, bw, bh)

    def _update_tracks(self, candidates):
        cfg = self.config
        # age existing tracks
        for t in self.tracks:
            t["age"] += 1

        for cand in candidates:
            matched = None
            best_iou = 0.0
            for t in self.tracks:
                if t["move"] != cand.get("move"):
                    continue
                iou = _bbox_iou(t["bbox"], cand["bbox"])
                if iou > cfg.match_iou and iou > best_iou:
                    best_iou = iou
                    matched = t
            if matched:
                matched["bbox"] = cand["bbox"]
                matched["score"] = cand.get("score", 0.0)
                matched["age"] = 0
                matched["hits"] += 1
            else:
                self.tracks.append(
                    {
                        "bbox": cand["bbox"],
                        "move": cand.get("move"),
                        "score": cand.get("score", 0.0),
                        "hits": 1,
                        "age": 0,
                    }
                )

        self.tracks = [t for t in self.tracks if t["age"] <= cfg.track_max_age]
        self.last_moves = [t for t in self.tracks if t["hits"] >= cfg.min_persist]

    def _draw_overlay(self, frame):
        if self.last_pred:
            label = self.last_pred.get("predicted_label", "?")
            prob = self.last_pred.get("pred_prob", 0.0)
            text = f"{label} ({prob:.2f})"
            cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        if self.config.show_roi and self.last_roi:
            x, y, w, h = self.last_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 0), 2)

        for cand in self.last_moves:
            x, y, w, h = cand["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            move = cand.get("move", "?")
            score = cand.get("score", 0.0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{move} {score:.2f}",
                (x1 + 2, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--moves", action="store_true")
    parser.add_argument("--move-interval", type=int, default=5)
    parser.add_argument("--classify-interval", type=int, default=5)
    parser.add_argument("--roi-padding", type=int, default=12)
    parser.add_argument("--roi-min-area", type=float, default=0.02)
    parser.add_argument("--min-persist", type=int, default=3)
    parser.add_argument("--move-score-floor", type=float, default=0.35)
    parser.add_argument("--move-geom-weight", type=float, default=0.7)
    parser.add_argument("--show-roi", action="store_true")
    args = parser.parse_args()

    config = RealTimeConfig(
        source=str(args.source),
        frame_width=args.frame_width,
        checkpoint=args.checkpoint,
        mapping=args.mapping,
        moves=args.moves,
        move_interval=args.move_interval,
        classify_interval=args.classify_interval,
        roi_padding=args.roi_padding,
        roi_min_area_ratio=args.roi_min_area,
        min_persist=args.min_persist,
        move_score_floor=args.move_score_floor,
        move_geom_weight=args.move_geom_weight,
        show_roi=args.show_roi,
    )
    RealTimeKnotApp(config).run()


if __name__ == "__main__":
    main()


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
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
