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
    denoise: bool = True
    noise_threshold: float = 600.0
    min_component_area: int = 40


class Preprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def run(self, img_rgb):
        cfg = self.config
        gray, is_noisy = _to_gray(img_rgb, cfg.width, cfg.denoise, cfg.noise_threshold)
        blur = cv2.GaussianBlur(gray, cfg.blur_kernel, 0)
        th = _binarize(blur, cfg.adaptive_block_size, cfg.adaptive_c, is_noisy)
        if cfg.denoise and is_noisy:
            th = _filter_components(th, cfg.min_component_area)
        opened = _morph_open(th, cfg.morph_kernel, cfg.morph_iterations)
        sk = skeletonize(opened.astype(bool))
        return sk, gray


def preprocess_for_skeleton(img_rgb, width=512, denoise=True):
    return Preprocessor(PreprocessConfig(width=width, denoise=denoise)).run(img_rgb)


def _to_gray(img_rgb, width: int, denoise: bool = True, noise_thresh: float = 600.0) -> Tuple[np.ndarray, bool]:
    h, w = img_rgb.shape[:2]
    scale = width / float(w)
    img = cv2.resize(img_rgb, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_noisy = lap_var > noise_thresh

    if denoise and is_noisy:
        # Edge-preserving bilateral filter suppresses rock grain and surface roughness
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    else:
        gray = cv2.equalizeHist(gray)

    return gray, is_noisy


def _binarize(blur: np.ndarray, block_size: int, adaptive_c: int, is_noisy: bool = False) -> np.ndarray:
    if is_noisy:
        # For noisy/photographic inputs, Otsu on bilateral image produces cleaner strand masks
        t = threshold_otsu(blur)
        th = (blur < t).astype("uint8") * 255
    else:
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


def _filter_components(th: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    if num_labels <= 1:
        return th
    out = np.zeros_like(th)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def _morph_open(th: np.ndarray, kernel: Tuple[int, int], iterations: int) -> np.ndarray:
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, morph_kernel, iterations=iterations)
