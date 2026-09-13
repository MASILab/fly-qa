"""Section 5: validation protocol.

Build a labeled calibration set (known-good + synthetically defective images at known
severities), hold out a separate labeled test set, and report sensitivity/specificity
for the FROZEN-WEIGHT decoder against that held-out set alongside a frozen baseline.
No accuracy claim is meaningful before this has actually run -- see the ground rules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


class DefectType(str, Enum):
    COLOR_SHIFT = "color_shift"
    BANDING = "banding"
    CORRUPTION = "corruption"
    ALPHA_LOSS = "alpha_loss"
    RESOLUTION_MISMATCH = "resolution_mismatch"


@dataclass(frozen=True)
class LabeledImage:
    path: Path
    is_defective: bool
    defect_type: DefectType | None
    severity: float | None  # 0..1, None for known-good


def inject_color_shift(image: Image.Image, severity: float, rng: random.Random) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    channel = rng.randrange(3)
    shift = severity * 255.0 * rng.choice([-1, 1])
    arr[:, :, channel] = np.clip(arr[:, :, channel] + shift, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def inject_banding(image: Image.Image, severity: float, rng: random.Random) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    levels = max(2, int(256 * (1.0 - severity)))
    step = 256.0 / levels
    arr = np.floor(arr / step) * step
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def inject_corruption(image: Image.Image, severity: float, rng: random.Random) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    num_blocks = max(1, int(severity * 20))
    for _ in range(num_blocks):
        bw, bh = rng.randint(5, max(6, w // 10)), rng.randint(5, max(6, h // 10))
        x0, y0 = rng.randint(0, max(0, w - bw)), rng.randint(0, max(0, h - bh))
        arr[y0 : y0 + bh, x0 : x0 + bw] = rng.randint(0, 255)
    return Image.fromarray(arr, mode="RGB")


def inject_alpha_loss(image: Image.Image, severity: float, rng: random.Random) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float64)
    faded = alpha * (1.0 - severity)
    out = rgba.copy()
    out.putalpha(Image.fromarray(faded.astype(np.uint8)))
    return out


def inject_resolution_mismatch(image: Image.Image, severity: float, rng: random.Random) -> Image.Image:
    w, h = image.size
    scale = 1.0 - severity * 0.9
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


_INJECTORS: dict[DefectType, Callable[[Image.Image, float, random.Random], Image.Image]] = {
    DefectType.COLOR_SHIFT: inject_color_shift,
    DefectType.BANDING: inject_banding,
    DefectType.CORRUPTION: inject_corruption,
    DefectType.ALPHA_LOSS: inject_alpha_loss,
    DefectType.RESOLUTION_MISMATCH: inject_resolution_mismatch,
}


def build_calibration_set(
    good_image_paths: list[Path],
    output_dir: Path,
    defects_per_image: int = 3,
    seed: int = 0,
) -> list[LabeledImage]:
    """For each known-good image, write the original plus `defects_per_image` synthetically
    defective variants (random type + severity) into output_dir. Returns ground truth labels.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled: list[LabeledImage] = []

    for src in good_image_paths:
        image = Image.open(src).convert("RGB")

        good_path = output_dir / f"{src.stem}__good.png"
        image.save(good_path)
        labeled.append(LabeledImage(path=good_path, is_defective=False, defect_type=None, severity=None))

        for i in range(defects_per_image):
            defect_type = rng.choice(list(DefectType))
            severity = rng.uniform(0.2, 0.9)
            defective = _INJECTORS[defect_type](image, severity, rng)
            defective_path = output_dir / f"{src.stem}__{defect_type.value}_{i}.png"
            defective.convert("RGB" if defect_type != DefectType.ALPHA_LOSS else "RGBA").save(defective_path)
            labeled.append(
                LabeledImage(
                    path=defective_path, is_defective=True, defect_type=defect_type, severity=severity
                )
            )

    return labeled


@dataclass(frozen=True)
class ValidationReport:
    n_total: int
    n_defective: int
    n_good: int
    true_positives: int
    false_negatives: int
    false_positives: int
    true_negatives: int
    n_flagged: int

    @property
    def sensitivity(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else float("nan")

    @property
    def specificity(self) -> float:
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom else float("nan")

    def summary(self, label: str) -> str:
        return (
            f"[{label}] n={self.n_total} (defective={self.n_defective}, good={self.n_good}) "
            f"sensitivity={self.sensitivity:.1%} specificity={self.specificity:.1%} "
            f"flagged={self.n_flagged} -- matches this rule on this calibration set, "
            f"NOT a validated defect classifier."
        )


def evaluate(
    labeled_images: list[LabeledImage],
    predict: Callable[[Path], str],  # returns "pass" | "fail" | "flag"
) -> ValidationReport:
    tp = fn = fp = tn = flagged = 0
    for item in labeled_images:
        verdict = predict(item.path)
        if verdict == "flag":
            flagged += 1
        predicted_defective = verdict == "fail"
        if item.is_defective and predicted_defective:
            tp += 1
        elif item.is_defective and not predicted_defective:
            fn += 1
        elif not item.is_defective and predicted_defective:
            fp += 1
        else:
            tn += 1

    n_defective = sum(1 for i in labeled_images if i.is_defective)
    return ValidationReport(
        n_total=len(labeled_images),
        n_defective=n_defective,
        n_good=len(labeled_images) - n_defective,
        true_positives=tp,
        false_negatives=fn,
        false_positives=fp,
        true_negatives=tn,
        n_flagged=flagged,
    )
