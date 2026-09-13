from pathlib import Path

from fly_qa.decoder import DecoderThresholds
from fly_qa.thresholds_store import load_thresholds, save_thresholds


def test_load_missing_file_returns_uncalibrated_placeholder(tmp_path: Path):
    thresholds = load_thresholds(tmp_path / "nonexistent.json")
    assert thresholds.calibrated is False


def test_save_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "decoder_thresholds.json"
    original = DecoderThresholds(confidence_fail_max=1.5e-6, confidence_flag_max=3.0e-6, calibrated=True)
    save_thresholds(path, original, calibration_notes={"n_calibration_images": 12})

    loaded = load_thresholds(path)
    assert loaded.confidence_fail_max == original.confidence_fail_max
    assert loaded.confidence_flag_max == original.confidence_flag_max
    assert loaded.calibrated is True


def test_save_includes_notes(tmp_path: Path):
    import json

    path = tmp_path / "decoder_thresholds.json"
    thresholds = DecoderThresholds(confidence_fail_max=1.0, confidence_flag_max=2.0, calibrated=True)
    save_thresholds(path, thresholds, calibration_notes={"foo": "bar"})

    data = json.loads(path.read_text())
    assert data["notes"] == {"foo": "bar"}
    assert "generated_at" in data
