import random
from pathlib import Path

from PIL import Image

from fly_qa.validation import (
    DefectType,
    build_calibration_set,
    evaluate,
    inject_alpha_loss,
    inject_banding,
    inject_color_shift,
    inject_corruption,
    inject_resolution_mismatch,
)


def _sample_image() -> Image.Image:
    img = Image.new("RGB", (100, 100))
    for x in range(100):
        for y in range(100):
            img.putpixel((x, y), ((x * 2) % 256, (y * 2) % 256, (x + y) % 256))
    return img


def test_color_shift_changes_a_channel():
    img = _sample_image()
    shifted = inject_color_shift(img, 0.5, random.Random(0))
    assert list(shifted.getdata()) != list(img.getdata())


def test_banding_reduces_color_levels():
    img = _sample_image()
    banded = inject_banding(img, 0.8, random.Random(0))
    import numpy as np

    original_unique = len(np.unique(np.asarray(img)))
    banded_unique = len(np.unique(np.asarray(banded)))
    assert banded_unique < original_unique


def test_corruption_modifies_pixels():
    img = _sample_image()
    corrupted = inject_corruption(img, 0.5, random.Random(0))
    assert list(corrupted.getdata()) != list(img.getdata())


def test_alpha_loss_reduces_alpha_values():
    img = _sample_image().convert("RGBA")
    faded = inject_alpha_loss(img, 0.5, random.Random(0))
    import numpy as np

    orig_alpha = np.asarray(img.getchannel("A"), dtype=np.float64).mean()
    new_alpha = np.asarray(faded.getchannel("A"), dtype=np.float64).mean()
    assert new_alpha < orig_alpha


def test_resolution_mismatch_shrinks_image():
    img = _sample_image()
    shrunk = inject_resolution_mismatch(img, 0.5, random.Random(0))
    assert shrunk.size[0] < img.size[0]
    assert shrunk.size[1] < img.size[1]


def test_build_calibration_set_creates_labeled_images(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    good_paths = []
    for i in range(3):
        p = src_dir / f"img{i}.png"
        _sample_image().save(p)
        good_paths.append(p)

    out_dir = tmp_path / "calibration"
    labeled = build_calibration_set(good_paths, out_dir, defects_per_image=2, seed=42)

    assert len(labeled) == 3 * (1 + 2)
    n_good = sum(1 for l in labeled if not l.is_defective)
    n_bad = sum(1 for l in labeled if l.is_defective)
    assert n_good == 3
    assert n_bad == 6
    for item in labeled:
        assert item.path.exists()


def test_build_calibration_set_is_reproducible_with_seed(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    p = src_dir / "img.png"
    _sample_image().save(p)

    labeled1 = build_calibration_set([p], tmp_path / "cal1", defects_per_image=3, seed=7)
    labeled2 = build_calibration_set([p], tmp_path / "cal2", defects_per_image=3, seed=7)

    types1 = [l.defect_type for l in labeled1]
    types2 = [l.defect_type for l in labeled2]
    assert types1 == types2


def test_evaluate_computes_sensitivity_and_specificity():
    from fly_qa.validation import LabeledImage

    labeled = [
        LabeledImage(Path("a"), is_defective=True, defect_type=DefectType.BANDING, severity=0.5),
        LabeledImage(Path("b"), is_defective=True, defect_type=DefectType.BANDING, severity=0.5),
        LabeledImage(Path("c"), is_defective=False, defect_type=None, severity=None),
        LabeledImage(Path("d"), is_defective=False, defect_type=None, severity=None),
    ]

    def predict(path: Path) -> str:
        return {"a": "fail", "b": "pass", "c": "pass", "d": "fail"}[path.name]

    report = evaluate(labeled, predict)
    assert report.true_positives == 1
    assert report.false_negatives == 1
    assert report.false_positives == 1
    assert report.true_negatives == 1
    assert abs(report.sensitivity - 0.5) < 1e-9
    assert abs(report.specificity - 0.5) < 1e-9


def test_summary_includes_honesty_disclaimer():
    from fly_qa.validation import ValidationReport

    report = ValidationReport(4, 2, 2, 1, 1, 1, 1, 0)
    text = report.summary("frozen-baseline")
    assert "NOT a validated defect classifier" in text
