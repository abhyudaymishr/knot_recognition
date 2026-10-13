"""
denoise.py - Fourier-domain denoising and measure-theoretic segmentation of cable photos.

Fourier part (each step is a multiplicative filter on the 2D spectrum; images are
mirror-padded first so the FFT's periodic wrap-around adds no artificial edge):
  1. Homomorphic filter: illumination is a slowly varying multiplicative factor,
     so it is low-frequency in log I. A high-emphasis filter
        H(D) = (gH - gL) (1 - exp(-D^2 / (2 D0^2))) + gL
     on FFT(log I) flattens lighting without touching edges.
  2. Spectral notch: periodic background texture (wood grain, fabric, tiles)
     concentrates in isolated spectral peaks; a cable's energy is spread over
     all directions. Coefficients far above the radial median of |F| (outside
     the lowest frequencies) are shrunk to that median, keeping their phase.
  3. Wiener shrinkage: white sensor noise has a flat power spectrum. Its level
     is estimated from the outermost frequency ring and removed with the gain
     G = max(0, 1 - sigma^2 / |F|^2).
  4. Scale band-pass: a difference of Gaussians tuned to the cable width keeps
     tube-sized structure and suppresses specks and slow residual shading.

Measure-theoretic part:
  The response R defines superlevel sets E_t = {R > t}. Their Lebesgue measure
  m(t) = |E_t| and their topology (b0 pieces, b1 holes, counting only parts of
  measure at least (0.6 w)^2 -- smaller sets are treated as null) are tracked over
  t. The threshold is taken at the middle of the longest t-interval on which the
  topology is constant and m(t) is a plausible cable area: the most persistent,
  hence least noise-dependent, segmentation. Its (b0, b1) feed the Euler check.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _freq_radius(h: int, w: int) -> np.ndarray:
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt(fx ** 2 + fy ** 2)  # cycles / pixel


def _reflect_pad(x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Mirror-pad by half the size so the periodic FFT sees no wrap-around jump."""
    ph, pw = x.shape[0] // 2, x.shape[1] // 2
    return np.pad(x, ((ph, ph), (pw, pw)), mode="reflect"), (ph, pw)


def _crop(x: np.ndarray, pad: Tuple[int, int], shape) -> np.ndarray:
    return x[pad[0]:pad[0] + shape[0], pad[1]:pad[1] + shape[1]]


def to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114])
    return a


def homomorphic(gray: np.ndarray, d0: float = 1 / 64, g_low: float = 0.25, g_high: float = 1.0) -> np.ndarray:
    L = np.log1p(gray)
    P, pad = _reflect_pad(L - L.mean())
    F = np.fft.fft2(P)
    D = _freq_radius(*P.shape)
    H = (g_high - g_low) * (1 - np.exp(-(D ** 2) / (2 * d0 ** 2))) + g_low
    return _crop(np.real(np.fft.ifft2(F * H)), pad, gray.shape)


def spectral_notch(x: np.ndarray, k: float = 6.0, min_freq: float = 1 / 48, nbins: int = 96) -> Tuple[np.ndarray, int]:
    F = np.fft.fft2(x)
    mag = np.abs(F)
    D = _freq_radius(*x.shape)
    bins = np.minimum((D / D.max() * nbins).astype(int), nbins - 1)
    med = np.zeros(nbins)
    for b in range(nbins):
        sel = mag[bins == b]
        med[b] = np.median(sel) if sel.size else 0
    ref = med[bins]
    peaks = (mag > k * ref) & (D > min_freq)
    scale = np.ones_like(mag)
    scale[peaks] = ref[peaks] / np.maximum(mag[peaks], 1e-12)
    return np.real(np.fft.ifft2(F * scale)), int(peaks.sum())


def wiener(x: np.ndarray, ring: float = 0.42) -> Tuple[np.ndarray, float]:
    Xp, pad = _reflect_pad(x)
    F = np.fft.fft2(Xp)
    P = np.abs(F) ** 2
    D = _freq_radius(*Xp.shape)
    noise_power = float(np.median(P[D > ring]))
    G = np.clip(1 - noise_power / np.maximum(P, 1e-12), 0, 1)
    return _crop(np.real(np.fft.ifft2(F * G)), pad, x.shape), noise_power


def band_pass(x: np.ndarray, width_px: float) -> np.ndarray:
    """Difference of Gaussians in the Fourier domain, tuned to a tube of the given width."""
    Xp, pad = _reflect_pad(x)
    F = np.fft.fft2(Xp)
    D = _freq_radius(*Xp.shape)
    s_small = max(width_px / 6.0, 0.7)        # keeps the tube profile
    s_large = 2.5 * width_px                   # removes background shading at larger scales
    g = lambda s: np.exp(-2 * (np.pi * D * s) ** 2)  # FT of a Gaussian with std s
    return _crop(np.real(np.fft.ifft2(F * (g(s_small) - g(s_large)))), pad, x.shape)


def estimate_width(gray: np.ndarray) -> float:
    """Cable width from the spectrum: the radial power peak of the band-limited image ~ 1/(2 w)."""
    x = gray - gray.mean()
    P = np.abs(np.fft.fft2(x)) ** 2
    D = _freq_radius(*gray.shape)
    nb = 128
    bins = np.minimum((D / 0.5 * nb).astype(int), nb - 1)
    prof = np.bincount(bins.ravel(), weights=(P * D).ravel(), minlength=nb) / np.maximum(np.bincount(bins.ravel(), minlength=nb), 1)
    lo = max(2, int(nb * (1 / 60) / 0.5))
    hi = int(nb * (1 / 4) / 0.5)
    k = lo + int(np.argmax(prof[lo:hi]))
    f = (k + 0.5) * 0.5 / nb
    return float(np.clip(1 / (2 * f), 4, 40))


def enhance(img: np.ndarray, width_px: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cable-positive response map (cable bright) and a report of what each filter did."""
    gray = to_gray(img)
    w = width_px or estimate_width(gray)
    h = homomorphic(gray)
    n, n_peaks = spectral_notch(h)
    d, noise = wiener(n)
    r = band_pass(d, w)
    # polarity by measure: the cable is the minority set, so its tail is the heavier one
    q_lo, q_med, q_hi = np.percentile(r, [2, 50, 98])
    if (q_med - q_lo) > (q_hi - q_med):
        r = -r
    r = (r - r.mean()) / (r.std() + 1e-9)
    return r, {"width_px": w, "notched_peaks": n_peaks, "noise_power": noise}


def _topology(mask: np.ndarray, min_area: float) -> Tuple[int, int]:
    import cv2

    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    b0 = int(sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 4 * min_area))
    nb, lab, stb, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    b1 = int(sum(1 for i in range(1, nb) if i not in border and stb[i, cv2.CC_STAT_AREA] >= min_area))
    return b0, b1


def stable_segmentation(r: np.ndarray, width_px: float, n_levels: int = 48,
                        area_range: Tuple[float, float] = (0.01, 0.45),
                        levels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Most persistent superlevel set of ``r`` within a plausible area band.

    ``levels`` gives explicit thresholds (e.g. probabilities 0.15..0.85 for a
    network's mask output); by default quantiles of ``r`` are used.
    """
    import cv2

    min_area = max(9.0, (0.6 * width_px) ** 2)
    ts = np.asarray(levels) if levels is not None else np.quantile(r, np.linspace(0.55, 0.995, n_levels))
    rows = []
    for t in ts:
        m = r > t
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        meas = float(m.mean())
        b0, b1 = _topology(m, min_area)
        rows.append((t, meas, b0, b1))
    best, cur = None, None
    for i, (t, meas, b0, b1) in enumerate(rows):
        ok = area_range[0] <= meas <= area_range[1] and b0 >= 1
        key = (b0, b1) if ok else None
        if key is not None and cur is not None and cur["key"] == key:
            cur["end"] = i
        else:
            if cur is not None and cur["key"] is not None:
                if best is None or (cur["end"] - cur["start"]) > (best["end"] - best["start"]):
                    best = cur
            cur = {"key": key, "start": i, "end": i}
    if cur is not None and cur["key"] is not None and (best is None or (cur["end"] - cur["start"]) > (best["end"] - best["start"])):
        best = cur
    if best is None:
        t = float(np.quantile(r, 0.85))
        info = {"stable": False}
    else:
        t = float(rows[(best["start"] + best["end"]) // 2][0])
        info = {"stable": True, "persistence_levels": best["end"] - best["start"] + 1,
                "pieces": best["key"][0], "holes": best["key"][1]}
    m = r > t
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    info.update(threshold=t, area_fraction=float(m.mean()), min_component_measure=min_area)
    return m, info


def segment(img: np.ndarray, width_px: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    r, rep = enhance(img, width_px)
    m, info = stable_segmentation(r, rep["width_px"])
    rep.update(info)
    return m, rep
