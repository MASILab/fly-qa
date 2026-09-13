"""Section 4: non-neural pre-check layer.

Most real PNG QA problems are cheap to catch without any connectome
simulation. This runs first; only images that pass are worth feeding to the
"fly brain" if you actually want its verdict too. Findings from this layer
are tagged `layer="precheck"` so they are never confused with (or credited
to) the connectome-layer decoder's output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

EXPECTED_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

ALL_BLACK_WHITE_STD_THRESHOLD = 1.0  # near-zero std across all pixels
NEAR_DUPLICATE_HAMMING_THRESHOLD = 4  # dHash bits differing, out of 64


@dataclass(frozen=True)
class PrecheckFinding:
    code: str
    severity: str  # "fail" | "warn"
    message: str
    layer: str = "precheck"


@dataclass
class PrecheckResult:
    path: Path
    decoded: bool
    findings: list[PrecheckFinding] = field(default_factory=list)
    image_rgba: Image.Image | None = None  # kept for the encoder if precheck passes
    exact_hash: str | None = None
    perceptual_hash: int | None = None

    @property
    def passed(self) -> bool:
        return self.decoded and not any(f.severity == "fail" for f in self.findings)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash for near-duplicate detection (not cryptographic)."""
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = 0
    for i, bit in enumerate(diff.flatten()):
        if bit:
            bits |= 1 << i
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def check_file_integrity(path: Path) -> tuple[bool, list[PrecheckFinding], Image.Image | None]:
    findings: list[PrecheckFinding] = []

    try:
        with path.open("rb") as f:
            sig = f.read(8)
        if sig != EXPECTED_PNG_SIGNATURE:
            findings.append(
                PrecheckFinding("bad_signature", "fail", f"Not a valid PNG signature: {sig!r}")
            )
            return False, findings, None
    except OSError as exc:
        findings.append(PrecheckFinding("unreadable_file", "fail", f"Could not read file: {exc}"))
        return False, findings, None

    try:
        with Image.open(path) as img:
            img.load()  # force full decode -- catches truncated IDAT etc.
            image = img.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        findings.append(PrecheckFinding("decode_failed", "fail", f"Failed to decode PNG: {exc}"))
        return False, findings, None

    return True, findings, image


def check_dimensions(
    image: Image.Image,
    expected_size: tuple[int, int] | None = None,
    min_size: tuple[int, int] = (1, 1),
) -> list[PrecheckFinding]:
    findings = []
    w, h = image.size
    if w < min_size[0] or h < min_size[1]:
        findings.append(
            PrecheckFinding("dimensions_too_small", "fail", f"Image is {w}x{h}, below minimum {min_size}")
        )
    if expected_size is not None and (w, h) != expected_size:
        findings.append(
            PrecheckFinding(
                "dimensions_mismatch", "warn",
                f"Image is {w}x{h}, expected {expected_size[0]}x{expected_size[1]}",
            )
        )
    return findings


def check_alpha(image: Image.Image, expect_alpha: bool = False) -> list[PrecheckFinding]:
    findings = []
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)

    if expect_alpha and not has_alpha:
        findings.append(PrecheckFinding("missing_alpha", "fail", "Expected an alpha channel but none was found"))
        return findings

    if has_alpha and image.mode == "RGBA":
        alpha = np.asarray(image.getchannel("A"))
        if alpha.std() < 0.5:
            findings.append(
                PrecheckFinding(
                    "degenerate_alpha", "warn",
                    f"Alpha channel is uniform (value={int(alpha.flat[0])}) -- likely unintentional",
                )
            )
    return findings


def check_statistical_anomalies(image: Image.Image, path: Path) -> list[PrecheckFinding]:
    findings = []
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    std = gray.std()
    mean = gray.mean()

    if std < ALL_BLACK_WHITE_STD_THRESHOLD:
        if mean < 10:
            findings.append(PrecheckFinding("all_black_frame", "fail", "Image is uniformly (near-)black"))
        elif mean > 245:
            findings.append(PrecheckFinding("all_white_frame", "fail", "Image is uniformly (near-)white"))
        else:
            findings.append(
                PrecheckFinding("flat_frame", "warn", f"Image has almost no variance (std={std:.2f})")
            )

    w, h = image.size
    file_size = path.stat().st_size
    uncompressed_estimate = w * h * len(image.getbands())
    if uncompressed_estimate > 0:
        ratio = file_size / uncompressed_estimate
        if ratio < 0.0005:
            findings.append(
                PrecheckFinding(
                    "suspiciously_small_file", "warn",
                    f"File size ({file_size}B) is unusually small for {w}x{h} -- possible corruption or near-empty content",
                )
            )
    return findings


def check_duplicate(
    image: Image.Image,
    exact_hash: str,
    seen_exact: dict[str, Path],
    seen_perceptual: dict[int, Path],
    path: Path,
) -> tuple[list[PrecheckFinding], int]:
    findings = []
    phash = _dhash(image)

    if exact_hash in seen_exact:
        findings.append(
            PrecheckFinding(
                "exact_duplicate", "warn", f"Byte-identical to {seen_exact[exact_hash]}"
            )
        )
    else:
        for other_hash, other_path in seen_perceptual.items():
            if _hamming(phash, other_hash) <= NEAR_DUPLICATE_HAMMING_THRESHOLD:
                findings.append(
                    PrecheckFinding(
                        "near_duplicate", "warn", f"Visually near-identical to {other_path}"
                    )
                )
                break

    return findings, phash


def run_precheck(
    path: Path,
    seen_exact: dict[str, Path] | None = None,
    seen_perceptual: dict[int, Path] | None = None,
    expected_size: tuple[int, int] | None = None,
    expect_alpha: bool = False,
) -> PrecheckResult:
    decoded, findings, image = check_file_integrity(path)
    if not decoded or image is None:
        return PrecheckResult(path=path, decoded=False, findings=findings)

    findings += check_dimensions(image, expected_size=expected_size)
    findings += check_alpha(image, expect_alpha=expect_alpha)
    findings += check_statistical_anomalies(image, path)

    exact_hash = _sha256_file(path)
    phash = None
    if seen_exact is not None and seen_perceptual is not None:
        dup_findings, phash = check_duplicate(image, exact_hash, seen_exact, seen_perceptual, path)
        findings += dup_findings
        seen_exact.setdefault(exact_hash, path)
        seen_perceptual.setdefault(phash, path)

    # Flatten alpha onto neutral gray for the encoder; raw alpha stays on `image` for callers
    # that need it (e.g. a future alpha-specific check), never forced through the sensory path.
    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (128, 128, 128))
    background.paste(rgba, mask=rgba.getchannel("A"))

    return PrecheckResult(
        path=path,
        decoded=True,
        findings=findings,
        image_rgba=rgba,
        exact_hash=exact_hash,
        perceptual_hash=phash,
    )
