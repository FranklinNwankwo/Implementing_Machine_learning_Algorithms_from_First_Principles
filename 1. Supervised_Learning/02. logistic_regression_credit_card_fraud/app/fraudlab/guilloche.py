"""
Guilloché line-work for the app header, the fine engraved patterns printed on
banknotes and payment cards as an anti-counterfeiting feature.

Both patterns are drawn from real curves rather than images:

* the rosette layers hypotrochoids  x = (R − r)·cos t + d·cos((R − r)/r · t)
* the band stacks phase-shifted sine waves, like the border of a banknote.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _path(xs: np.ndarray, ys: np.ndarray) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    return f"M{pts.split(' ', 1)[0]} L{pts.split(' ', 1)[1]}"


@lru_cache(maxsize=4)
def rosette_svg(colour: str = "#2D6A5F", size: int = 220) -> str:
    """A layered hypotrochoid rosette, centred in a size×size box."""
    c = size / 2
    layers = [
        # (R, r, d, scale, opacity, rotation)
        (1.0, 7 / 24, 0.62, 0.47, 0.55, 0.0),
        (1.0, 5 / 18, 0.55, 0.39, 0.45, np.pi / 18),
        (1.0, 3 / 11, 0.42, 0.29, 0.40, np.pi / 11),
    ]
    paths = []
    for R, r, d, scale, opacity, rot in layers:
        t = np.linspace(0, 2 * np.pi * 24, 9000)
        x = (R - r) * np.cos(t) + d * np.cos((R - r) / r * t)
        y = (R - r) * np.sin(t) - d * np.sin((R - r) / r * t)
        xr = x * np.cos(rot) - y * np.sin(rot)
        yr = x * np.sin(rot) + y * np.cos(rot)
        k = scale * size / (np.max(np.hypot(x, y)))
        paths.append(
            f'<path d="{_path(c + k * xr, c + k * yr)}" fill="none" stroke="{colour}" '
            f'stroke-width="0.55" stroke-opacity="{opacity}"/>'
        )
    paths.append(f'<circle cx="{c}" cy="{c}" r="{0.485 * size:.1f}" fill="none" stroke="{colour}" '
                 f'stroke-width="0.8" stroke-opacity="0.5"/>')
    return (f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" '
            f'aria-label="Guilloché rosette" xmlns="http://www.w3.org/2000/svg">{"".join(paths)}</svg>')


@lru_cache(maxsize=4)
def band_svg(colour: str = "#2D6A5F", width: int = 1200, height: int = 26, n_waves: int = 7) -> str:
    """A banknote-border band of interleaved sine waves, stretched to the container width."""
    x = np.linspace(0, width, 900)
    paths = []
    for i in range(n_waves):
        phase = i * np.pi / n_waves
        y = height / 2 + (height / 2 - 2) * np.sin(2 * np.pi * x / 46 + phase) * np.cos(2 * np.pi * x / 610 + phase / 3)
        paths.append(f'<path d="{_path(x, y)}" fill="none" stroke="{colour}" stroke-width="0.6" '
                     f'stroke-opacity="{0.35 + 0.05 * (i % 3)}"/>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none" '
            f'aria-hidden="true" xmlns="http://www.w3.org/2000/svg">{"".join(paths)}</svg>')
