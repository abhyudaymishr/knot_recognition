"""
draw.py - Procedural 2D/3D Knot & Braid Image Rendering Engine for Knotgen.

Provides vector-to-raster diagram generation with over/under strand crossing resolution,
smooth cubic Bezier spline interpolation, and halo-based gap rendering.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .knotgen import Knot


def _cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 40) -> np.ndarray:
    """Evaluates a cubic Bezier curve at n points."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - t) ** 3 * p0 + 3.0 * (1.0 - t) ** 2 * t * p1 + 3.0 * (1.0 - t) * t ** 2 * p2 + t ** 3 * p3


def render_knot_diagram(
    knot: "Knot",
    filename: Optional[Union[str, Path]] = None,
    size: Tuple[int, int] = (512, 512),
    line_width: float = 4.0,
    gap_ratio: float = 2.4,
    bg_color: str = "white",
    line_color: str = "black",
    dpi: int = 100,
) -> np.ndarray:
    """
    Renders a planar knot diagram with over/under crossings for a Knot instance.

    Args:
        knot: The Knot instance to render.
        filename: Optional path to save the generated image (e.g. .png, .jpg, .svg).
        size: Target image dimensions (width, height) in pixels.
        line_width: Width of the knot strand in points.
        gap_ratio: Ratio of the white under-strand crossing break halo relative to line_width.
        bg_color: Background color.
        line_color: Knot strand stroke color.
        dpi: Dots per inch for rasterization.

    Returns:
        np.ndarray: RGB image array of shape (height, width, 3) with dtype uint8.
    """
    figsize = (size[0] / dpi, size[1] / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")
    ax.axis("off")

    n = knot.n
    word = knot.word
    m = len(word)

    if m == 0 or n <= 1:
        # Unknot: render a clean circular loop
        theta = np.linspace(0, 2 * np.pi, 250)
        ax.plot(np.cos(theta), np.sin(theta), color=line_color, lw=line_width, solid_capstyle="round")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
    else:
        dy = 1.6
        y_top = 0.0
        y_bot = -(m * dy)

        under_segments: List[np.ndarray] = []
        over_segments: List[np.ndarray] = []
        regular_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        cur_x = {i: float(i) for i in range(n)}

        for step, g in enumerate(word):
            y0 = -step * dy
            y1 = -(step + 1) * dy
            ymid = (y0 + y1) / 2.0

            i = abs(g) - 1
            sign = 1 if g > 0 else -1

            # Straight strands that do not cross at this step
            for j in range(n):
                if j != i and j != i + 1:
                    regular_segments.append(((cur_x[j], y0), (cur_x[j], y1)))

            xa = cur_x[i]
            xb = cur_x[i + 1]

            # Cubic Bezier interpolation across the swap
            p0_a, p3_a = np.array([xa, y0]), np.array([xb, y1])
            p1_a, p2_a = np.array([xa, ymid]), np.array([xb, ymid])
            curve_a = _cubic_bezier(p0_a, p1_a, p2_a, p3_a, n=35)

            p0_b, p3_b = np.array([xb, y0]), np.array([xa, y1])
            p1_b, p2_b = np.array([xb, ymid]), np.array([xa, ymid])
            curve_b = _cubic_bezier(p0_b, p1_b, p2_b, p3_b, n=35)

            if sign > 0:
                # Positive crossing: strand i (curve_a) passes OVER strand i+1 (curve_b)
                under_segments.append(curve_b)
                over_segments.append(curve_a)
            else:
                # Negative crossing: strand i (curve_a) passes UNDER strand i+1 (curve_b)
                under_segments.append(curve_a)
                over_segments.append(curve_b)

            cur_x[i], cur_x[i + 1] = xb, xa

        # Closure routing
        closure_segments: List[np.ndarray] = []
        if knot.closure == "braid":
            # Standard braid closure: loop bottom ends back to top around the right
            for j in range(n):
                x_start = cur_x[j]
                x_end = float(j)
                x_outer = n - 0.5 + (j + 1) * 0.75

                arc_bot_x = np.linspace(x_start, x_outer, 25)
                arc_bot_y = y_bot - 0.75 * (1.0 - np.sin(np.linspace(0.0, np.pi / 2.0, 25)))

                arc_top_x = np.linspace(x_outer, x_end, 25)
                arc_top_y = y_top + 0.75 * (1.0 - np.cos(np.linspace(0.0, np.pi / 2.0, 25)))

                pts_closure = np.vstack([
                    np.column_stack([arc_bot_x, arc_bot_y]),
                    np.array([[x_outer, y_bot - 0.75], [x_outer, y_top + 0.75]]),
                    np.column_stack([arc_top_x, arc_top_y]),
                ])
                closure_segments.append(pts_closure)

        elif isinstance(knot.closure, tuple):
            tp, bp = knot.closure
            for j1, j2 in tp:
                x1, x2 = float(j1), float(j2)
                xm = (x1 + x2) / 2.0
                r = abs(x2 - x1) / 2.0
                theta = np.linspace(np.pi, 0.0, 35)
                cap_x = xm + r * np.cos(theta)
                cap_y = y_top + r * np.sin(theta)
                closure_segments.append(np.column_stack([cap_x, cap_y]))

            for j1, j2 in bp:
                x1, x2 = cur_x[j1], cur_x[j2]
                xm = (x1 + x2) / 2.0
                r = abs(x2 - x1) / 2.0
                theta = np.linspace(0.0, -np.pi, 35)
                cup_x = xm + r * np.cos(theta)
                cup_y = y_bot + r * np.sin(theta)
                closure_segments.append(np.column_stack([cup_x, cup_y]))

        # Render pass 1: Background straight segments and closures
        for seg in regular_segments:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=line_color, lw=line_width, solid_capstyle="round", zorder=1)
        for seg in closure_segments:
            ax.plot(seg[:, 0], seg[:, 1], color=line_color, lw=line_width, solid_capstyle="round", zorder=1)

        # Render pass 2: Under crossing strands
        for curve in under_segments:
            ax.plot(curve[:, 0], curve[:, 1], color=line_color, lw=line_width, solid_capstyle="round", zorder=2)

        # Render pass 3: Over crossing strands with protective background gap halo
        gap_lw = line_width * gap_ratio
        for curve in over_segments:
            ax.plot(curve[:, 0], curve[:, 1], color=bg_color, lw=gap_lw, solid_capstyle="round", zorder=3)
            ax.plot(curve[:, 0], curve[:, 1], color=line_color, lw=line_width, solid_capstyle="round", zorder=4)

    # Export to memory buffer
    fig.canvas.draw()
    rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
    img_array = rgba_buffer[:, :, :3].copy()

    if filename is not None:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, facecolor=bg_color, bbox_inches="tight", pad_inches=0.1)

    plt.close(fig)
    return img_array


def render_torus_smooth(
    p: int,
    q: int,
    filename: Optional[Union[str, Path]] = None,
    size: Tuple[int, int] = (512, 512),
    line_width: float = 4.0,
    r_major: float = 2.0,
    r_minor: float = 0.8,
    num_points: int = 1200,
    bg_color: str = "white",
    line_color: str = "black",
    dpi: int = 100,
) -> np.ndarray:
    """
    Renders a 3D smooth space embedding of torus knot T(p, q) projected to 2D.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x = (r_major + r_minor * np.cos(q * t)) * np.cos(p * t)
    y = (r_major + r_minor * np.cos(q * t)) * np.sin(p * t)
    z = r_minor * np.sin(q * t)

    figsize = (size[0] / dpi, size[1] / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")
    ax.axis("off")

    # Render segments ordered by depth z for natural over/under appearance
    # Discretize into small segments and sort by average z
    segments = []
    for k in range(len(t) - 1):
        x_seg = [x[k], x[k + 1]]
        y_seg = [y[k], y[k + 1]]
        z_avg = (z[k] + z[k + 1]) / 2.0
        segments.append((z_avg, x_seg, y_seg))

    segments.sort(key=lambda s: s[0])

    for z_avg, x_seg, y_seg in segments:
        ax.plot(x_seg, y_seg, color=line_color, lw=line_width, solid_capstyle="round")

    fig.canvas.draw()
    rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
    img_array = rgba_buffer[:, :, :3].copy()

    if filename is not None:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, facecolor=bg_color, bbox_inches="tight", pad_inches=0.1)

    plt.close(fig)
    return img_array
