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


def _sample_luminance_grid(image: Image.Image, rows: int, cols: int) -> np.ndarray:
    small = image.resize((cols, rows), Image.LANCZOS)
    arr = np.asarray(small.convert("L"), dtype=np.float64) / 255.0
    return arr.flatten()


def _sample_chrominance_grid(image: Image.Image, rows: int, cols: int) -> np.ndarray:
    """Per-cell chrominance signal: saturation-weighted hue, confidence-gated.

    Two compounding problems with naively resizing an image and reading the
    HSV hue channel of the result: (1) hue is circular (0 and 1 are the same
    color), so a plain linear mean of hue values that straddle the
    wraparound point is meaningless; (2) hue is undefined/noisy for
    near-gray (low-saturation) pixels. (1) is fixed by averaging unit
    vectors (cos(hue), sin(hue)) rather than hue itself (a saturation-
    weighted circular mean). But fixing the circular-mean math alone does
    NOT fix (2)'s real-world impact: confirmed directly on a synthetic
    apples-vs-oranges test image where the fruit occupies ~20-30% of the
    frame, most grid cells fall entirely on background, and if the
    background has any consistent (even faint) color cast -- as most real
    photos and the TractSeg anatomical-slice background both do -- that
    cast is itself a *stable, class-irrelevant* hue that swamps the small
    number of genuinely informative (fruit- or overlay-) cells once
    aggregated downstream, even with per-cell hue computed correctly.

    Fixed by gating each cell's injected signal by its own confidence
    (saturation): `signal = hue * saturation`. A background cell (low
    saturation) contributes near-zero regardless of what hue it nominally
    reads; a strongly-colored cell contributes at close to full strength.
    This suppresses the class-irrelevant majority instead of just computing
    it more precisely.
    """
    hsv = np.asarray(image.convert("HSV"), dtype=np.float64)
    hue_rad = hsv[:, :, 0] / 255.0 * 2 * np.pi
    sat = hsv[:, :, 1] / 255.0

    def resize_float(arr: np.ndarray) -> np.ndarray:
        img_f = Image.fromarray(arr.astype(np.float32), mode="F")
        return np.asarray(img_f.resize((cols, rows), Image.BILINEAR), dtype=np.float64)

    sin_small = resize_float(np.sin(hue_rad) * sat)
    cos_small = resize_float(np.cos(hue_rad) * sat)
    sat_small = resize_float(sat)

    hue_small = (np.arctan2(sin_small, cos_small) / (2 * np.pi)) % 1.0
    confidence = np.clip(sat_small, 0.0, 1.0)
    return (hue_small * confidence).flatten()


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

    lum_values = _sample_luminance_grid(letterboxed, lum_rows, lum_cols)
    chrom_values = _sample_chrominance_grid(letterboxed, chrom_rows, chrom_cols)

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
