"""Section 2: sensory encoder (image -> connectome input).

This is an INVENTED PROXY, not a validated visual model. There is no
retinotopic mapping here: each photoreceptor body ID (sorted, for a
reproducible deterministic order) is assigned one cell of a grid the image
is resized into. R1-R6 channels receive luminance; R8 channels receive a
chrominance summary. Document any change to this mapping in docs/model.md --
reproducibility is the point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

LETTERBOX_FILL = (128, 128, 128)


@dataclass(frozen=True)
class EncodedInput:
    """Injected current per body ID, ready to add into the simulator's external-input vector."""

    r1_r6_current: dict[int, float]
    r8_current: dict[int, float]


def _grid_dims(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize preserving aspect ratio, padding with neutral gray (no cropping/distortion)."""
    target_w, target_h = size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), LETTERBOX_FILL)
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def _sample_grid(image: Image.Image, rows: int, cols: int, channel: str) -> np.ndarray:
    """Resize to an rows x cols grid and return a flat array of per-cell values in [0, 1]."""
    small = image.resize((cols, rows), Image.LANCZOS)
    if channel == "luminance":
        arr = np.asarray(small.convert("L"), dtype=np.float64) / 255.0
    elif channel == "chrominance":
        hsv = np.asarray(small.convert("HSV"), dtype=np.float64)
        arr = hsv[:, :, 0] / 255.0  # hue channel, normalized
    else:
        raise ValueError(f"unknown channel: {channel}")
    return arr.flatten()


def encode_image(
    image: Image.Image,
    r1_r6_body_ids: list[int],
    r8_body_ids: list[int],
    grid_size: tuple[int, int] = (256, 256),
    gain: float = 1.0,
) -> EncodedInput:
    """Map an RGB image onto injected current for each photoreceptor body ID.

    `image` should already be precheck-passed and alpha-flattened (RGB, no alpha).
    """
    letterboxed = letterbox(image, grid_size)

    r1_r6_ids_sorted = sorted(r1_r6_body_ids)
    r8_ids_sorted = sorted(r8_body_ids)

    lum_rows, lum_cols = _grid_dims(len(r1_r6_ids_sorted))
    chrom_rows, chrom_cols = _grid_dims(len(r8_ids_sorted))

    lum_values = _sample_grid(letterboxed, lum_rows, lum_cols, "luminance")
    chrom_values = _sample_grid(letterboxed, chrom_rows, chrom_cols, "chrominance")

    r1_r6_current = {
        body_id: float(lum_values[i]) * gain
        for i, body_id in enumerate(r1_r6_ids_sorted)
        if i < len(lum_values)
    }
    r8_current = {
        body_id: float(chrom_values[i]) * gain
        for i, body_id in enumerate(r8_ids_sorted)
        if i < len(chrom_values)
    }

    return EncodedInput(r1_r6_current=r1_r6_current, r8_current=r8_current)
