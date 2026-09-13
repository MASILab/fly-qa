from fly_qa.cli import resolve_thresholds
from fly_qa.decoder import DecoderThresholds

LOADED = DecoderThresholds(confidence_fail_max=2.0, confidence_flag_max=3.0, calibrated=True)


def test_no_overrides_returns_loaded_unchanged():
    result = resolve_thresholds(LOADED, None, None)
    assert result is LOADED


def test_override_both_values():
    result = resolve_thresholds(LOADED, confidence_fail_max=5.0, confidence_flag_max=8.0)
    assert result.confidence_fail_max == 5.0
    assert result.confidence_flag_max == 8.0
    assert result.calibrated is True


def test_override_fail_only_keeps_loaded_flag_when_still_consistent():
    result = resolve_thresholds(LOADED, confidence_fail_max=1.0, confidence_flag_max=None)
    assert result.confidence_fail_max == 1.0
    assert result.confidence_flag_max == 3.0  # loaded flag_max, still >= new fail_max


def test_override_fail_above_loaded_flag_raises_flag_to_match():
    # Regression test: overriding fail_max higher than the loaded flag_max used to leave
    # flag_max unreachable (decode() checks fail_max first, so a stale lower flag_max was
    # dead code). flag_max must never end up below fail_max.
    result = resolve_thresholds(LOADED, confidence_fail_max=10.0, confidence_flag_max=None)
    assert result.confidence_fail_max == 10.0
    assert result.confidence_flag_max == 10.0


def test_override_flag_only():
    result = resolve_thresholds(LOADED, confidence_fail_max=None, confidence_flag_max=6.0)
    assert result.confidence_fail_max == 2.0  # loaded fail_max unchanged
    assert result.confidence_flag_max == 6.0


def test_override_marks_calibrated_true_even_if_loaded_was_uncalibrated():
    uncalibrated = DecoderThresholds(confidence_fail_max=0.0, confidence_flag_max=0.0, calibrated=False)
    result = resolve_thresholds(uncalibrated, confidence_fail_max=1.0, confidence_flag_max=None)
    assert result.calibrated is True
