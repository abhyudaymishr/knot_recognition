"""
Real-time knot detection and Reidemeister move prediction.

Usage:
    python -m knot_recognition.realtime --source 0 --checkpoint ./checkpoints/best.pth --moves
"""
from dataclasses import dataclass
from typing import Optional

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
        self.detector = ReidemeisterDetector(ReidemeisterConfig()) if config.moves else None

        self.last_pred = None
        self.last_moves = []

    def _open_capture(self, source: str):
        if source.isdigit():
            return cv2.VideoCapture(int(source))
        return cv2.VideoCapture(source)

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
                self.last_moves = self.detector.detect_array(rgb)

            self._draw_overlay(frame)

            if self.config.show:
                cv2.imshow(self.config.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_idx += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def _draw_overlay(self, frame):
        if self.last_pred:
            label = self.last_pred.get("predicted_label", "?")
            prob = self.last_pred.get("pred_prob", 0.0)
            text = f"{label} ({prob:.2f})"
            cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

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
    args = parser.parse_args()

    config = RealTimeConfig(
        source=str(args.source),
        frame_width=args.frame_width,
        checkpoint=args.checkpoint,
        mapping=args.mapping,
        moves=args.moves,
        move_interval=args.move_interval,
        classify_interval=args.classify_interval,
    )
    RealTimeKnotApp(config).run()


if __name__ == "__main__":
    main()
