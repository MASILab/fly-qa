#!/usr/bin/env python3
"""Generate a synthetic apples-vs-oranges toy dataset.

A simple, strongly-separable demo task for exercising the connectome
pipeline independent of TractSeg-specific complexity. Apple = "good"
(should pass), orange = "bad" (should fail) -- an arbitrary but convenient
mapping onto the QA pass/fail framing, not a claim that oranges are
defective apples.

Usage:
    python scripts/generate_fruit_demo.py --output-dir .devtest/fruit_demo --n-per-class 60
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 400


def _make_background(rng: random.Random) -> Image.Image:
    base = rng.randint(200, 245)
    return Image.new("RGB", (SIZE, SIZE), (base, base - rng.randint(0, 10), base - rng.randint(5, 15)))


def make_apple(rng: random.Random) -> Image.Image:
    img = _make_background(rng)
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE // 2 + rng.randint(-30, 30), SIZE // 2 + rng.randint(-20, 30)
    r = rng.randint(90, 130)

    is_green = rng.random() < 0.25
    color = (
        (rng.randint(90, 140), rng.randint(150, 190), rng.randint(40, 70))
        if is_green
        else (rng.randint(170, 220), rng.randint(20, 55), rng.randint(20, 45))
    )

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    dimple_w = r * 0.35
    draw.ellipse([cx - dimple_w / 2, cy - r - 5, cx + dimple_w / 2, cy - r + 20], fill=img.getpixel((2, 2)))

    stem_color = (90, 60, 30)
    draw.line([cx, cy - r + 10, cx + rng.randint(-5, 15), cy - r - 25], fill=stem_color, width=5)
    if rng.random() < 0.6:
        leaf_color = (60, 140, 50)
        lx, ly = cx + 15, cy - r - 15
        draw.ellipse([lx, ly, lx + 25, ly + 12], fill=leaf_color)

    hl_r = r * 0.25
    draw.ellipse(
        [cx - r * 0.4, cy - r * 0.4, cx - r * 0.4 + hl_r, cy - r * 0.4 + hl_r],
        fill=tuple(min(255, c + 60) for c in color),
    )
    return img.filter(ImageFilter.GaussianBlur(1))


def make_orange(rng: random.Random) -> Image.Image:
    img = _make_background(rng)
    draw = ImageDraw.Draw(img)
    cx, cy = SIZE // 2 + rng.randint(-30, 30), SIZE // 2 + rng.randint(-20, 30)
    r = rng.randint(90, 130)
    color = (rng.randint(225, 255), rng.randint(110, 160), rng.randint(0, 35))

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    calyx_color = (rng.randint(140, 170), rng.randint(100, 120), rng.randint(20, 40))
    draw.ellipse([cx - 8, cy - r - 2, cx + 8, cy - r + 10], fill=calyx_color)

    for _ in range(140):
        angle = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0, r * 0.92)
        px = cx + dist * math.cos(angle)
        py = cy + dist * math.sin(angle)
        shade = rng.randint(-25, -5)
        dot_color = tuple(max(0, c + shade) for c in color)
        draw.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=dot_color)

    hl_r = r * 0.22
    draw.ellipse(
        [cx - r * 0.4, cy - r * 0.4, cx - r * 0.4 + hl_r, cy - r * 0.4 + hl_r],
        fill=tuple(min(255, c + 50) for c in color),
    )
    return img.filter(ImageFilter.GaussianBlur(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".devtest/fruit_demo"))
    parser.add_argument("--n-per-class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    (args.output_dir / "apples").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oranges").mkdir(parents=True, exist_ok=True)
    for i in range(args.n_per_class):
        make_apple(rng).save(args.output_dir / "apples" / f"apple_{i:03d}.png")
        make_orange(rng).save(args.output_dir / "oranges" / f"orange_{i:03d}.png")

    print(f"wrote {args.n_per_class} apples + {args.n_per_class} oranges to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
