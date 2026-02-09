from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class PreprocessConfig:
    width: int = 512
    blur_kernel: Tuple[int, int] = (5, 5)
    adaptive_block_size: int = 31
    adaptive_c: int = 10
    morph_kernel: Tuple[int, int] = (3, 3)
    morph_iterations: int = 1


class Preprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def run(self, img_rgb):
        cfg = self.config
        h, w = img_rgb.shape[:2]
        scale = cfg.width / float(w)
        img = cv2.resize(img_rgb, (cfg.width, int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gray = cv2.equalizeHist(gray)

        blur = cv2.GaussianBlur(gray, cfg.blur_kernel, 0)

        th = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            cfg.adaptive_block_size,
            cfg.adaptive_c,
        )
        if th.mean() < 10:
            t = threshold_otsu(blur)
            th = (blur < t).astype("uint8") * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg.morph_kernel)
        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_iterations)

        sk = skeletonize(opened.astype(bool))
        return sk, gray


def preprocess_for_skeleton(img_rgb, width=512):
    return Preprocessor(PreprocessConfig(width=width)).run(img_rgb)
