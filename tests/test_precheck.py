from pathlib import Path

from PIL import Image

from fly_qa.precheck import run_precheck


def _save(path: Path, image: Image.Image):
    image.save(path)
    return path


def test_valid_png_passes(tmp_path: Path):
    img = Image.new("RGB", (100, 100), (120, 130, 140))
    for x in range(100):
        for y in range(100):
            img.putpixel((x, y), ((x * 3) % 256, (y * 3) % 256, (x + y) % 256))
    path = _save(tmp_path / "good.png", img)

    result = run_precheck(path)
    assert result.decoded
    assert result.passed
    assert result.image_rgba is not None


def test_truncated_file_fails_decode(tmp_path: Path):
    img = Image.new("RGB", (50, 50), (10, 10, 10))
    path = tmp_path / "truncated.png"
    img.save(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])  # truncate mid-file

    result = run_precheck(path)
    assert not result.decoded
    assert any(f.code == "decode_failed" for f in result.findings)
    assert not result.passed


def test_not_a_png_fails_signature(tmp_path: Path):
    path = tmp_path / "fake.png"
    path.write_bytes(b"not a real png file")

    result = run_precheck(path)
    assert not result.decoded
    assert any(f.code == "bad_signature" for f in result.findings)


def test_all_black_frame_flagged(tmp_path: Path):
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    path = _save(tmp_path / "black.png", img)

    result = run_precheck(path)
    assert result.decoded
    assert any(f.code == "all_black_frame" for f in result.findings)
    assert not result.passed  # "fail" severity


def test_all_white_frame_flagged(tmp_path: Path):
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    path = _save(tmp_path / "white.png", img)

    result = run_precheck(path)
    assert any(f.code == "all_white_frame" for f in result.findings)


def test_dimension_mismatch_warns(tmp_path: Path):
    img = Image.new("RGB", (50, 50), (100, 100, 100))
    for x in range(50):
        img.putpixel((x, 0), (x * 5 % 256, 0, 0))
    path = _save(tmp_path / "wrong_size.png", img)

    result = run_precheck(path, expected_size=(100, 100))
    assert any(f.code == "dimensions_mismatch" and f.severity == "warn" for f in result.findings)
    # a warn-only finding shouldn't fail the precheck by itself
    assert result.passed


def test_missing_alpha_flagged_when_expected(tmp_path: Path):
    img = Image.new("RGB", (50, 50), (50, 60, 70))
    for x in range(50):
        img.putpixel((x, 0), (x * 5 % 256, 10, 20))
    path = _save(tmp_path / "no_alpha.png", img)

    result = run_precheck(path, expect_alpha=True)
    assert any(f.code == "missing_alpha" and f.severity == "fail" for f in result.findings)
    assert not result.passed


def test_exact_duplicate_detected(tmp_path: Path):
    img = Image.new("RGB", (40, 40), (30, 40, 50))
    for x in range(40):
        img.putpixel((x, 0), (x * 6 % 256, 5, 5))
    path1 = _save(tmp_path / "a.png", img)
    path2 = _save(tmp_path / "b.png", img)

    seen_exact: dict = {}
    seen_perceptual: dict = {}
    r1 = run_precheck(path1, seen_exact=seen_exact, seen_perceptual=seen_perceptual)
    r2 = run_precheck(path2, seen_exact=seen_exact, seen_perceptual=seen_perceptual)

    assert not any(f.code == "exact_duplicate" for f in r1.findings)
    assert any(f.code == "exact_duplicate" for f in r2.findings)


def test_findings_tagged_precheck_layer(tmp_path: Path):
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    path = _save(tmp_path / "black2.png", img)
    result = run_precheck(path)
    assert all(f.layer == "precheck" for f in result.findings)
