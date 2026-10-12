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
        gray = _to_gray(img_rgb, cfg.width)
        blur = cv2.GaussianBlur(gray, cfg.blur_kernel, 0)
        th = _binarize(blur, cfg.adaptive_block_size, cfg.adaptive_c)
        opened = _morph_open(th, cfg.morph_kernel, cfg.morph_iterations)
        sk = skeletonize(opened.astype(bool))
        return sk, gray


def preprocess_for_skeleton(img_rgb, width=512):
    return Preprocessor(PreprocessConfig(width=width)).run(img_rgb)


def _to_gray(img_rgb, width: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    scale = width / float(w)
    img = cv2.resize(img_rgb, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(gray)


def _binarize(blur: np.ndarray, block_size: int, adaptive_c: int) -> np.ndarray:
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        adaptive_c,
    )
    if th.mean() < 10:
        t = threshold_otsu(blur)
        th = (blur < t).astype("uint8") * 255
    return th


def _morph_open(th: np.ndarray, kernel: Tuple[int, int], iterations: int) -> np.ndarray:
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, morph_kernel, iterations=iterations)
