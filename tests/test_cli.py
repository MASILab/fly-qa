from datetime import datetime
from pathlib import Path

from fly_qa.cli import RESULT_FIELDNAMES, build_result_row, resolve_thresholds
from fly_qa.decoder import DecoderThresholds
from fly_qa.pipeline import PipelineResult

LOADED = DecoderThresholds(confidence_fail_max=2.0, confidence_flag_max=3.0, calibrated=True)

FIXED_NOW = datetime(2026, 4, 17, 10, 17, 24)


def make_result(**overrides) -> PipelineResult:
    defaults = dict(
        path=Path("some/dir/image.png"),
        precheck_passed=True,
        precheck_findings=[],
        fed_to_connectome=True,
        defect_score=0.42,
        confidence_signal=1.5,
        verdict="pass",
        calibrated=True,
        duration_sec=3.586,
    )
    defaults.update(overrides)
    return PipelineResult(**defaults)


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


def test_result_fieldnames_lead_with_masi_qa_core_columns():
    assert RESULT_FIELDNAMES[:6] == [
        "filename",
        "QA_status",
        "reason",
        "user",
        "date",
        "duration",
    ]


def test_build_result_row_maps_pass_verdict_to_yes():
    row = build_result_row(make_result(verdict="pass"), now=FIXED_NOW)
    assert row["QA_status"] == "yes"
    assert row["verdict"] == "pass"


def test_build_result_row_maps_fail_verdict_to_no():
    row = build_result_row(make_result(verdict="fail"), now=FIXED_NOW)
    assert row["QA_status"] == "no"


def test_build_result_row_maps_flag_verdict_to_maybe():
    row = build_result_row(make_result(verdict="flag"), now=FIXED_NOW)
    assert row["QA_status"] == "maybe"


def test_build_result_row_filename_is_path_string():
    row = build_result_row(make_result(path=Path("a/b/image.png")), now=FIXED_NOW)
    assert row["filename"] == "a/b/image.png"


def test_build_result_row_user_is_fly():
    row = build_result_row(make_result(), now=FIXED_NOW)
    assert row["user"] == "fly"


def test_build_result_row_date_is_formatted_from_now():
    row = build_result_row(make_result(), now=FIXED_NOW)
    assert row["date"] == "2026-04-17 10:17:24"


def test_build_result_row_duration_matches_duration_sec():
    row = build_result_row(make_result(duration_sec=3.586), now=FIXED_NOW)
    assert row["duration"] == 3.586


def test_build_result_row_reason_empty_when_precheck_passed():
    row = build_result_row(make_result(precheck_passed=True, precheck_findings=[]), now=FIXED_NOW)
    assert row["reason"] == ""


def test_build_result_row_reason_carries_precheck_findings_on_failure():
    row = build_result_row(
        make_result(
            precheck_passed=False,
            precheck_findings=["blank(error)", "corrupt(warning)"],
            fed_to_connectome=False,
            defect_score=None,
            confidence_signal=None,
            verdict="fail",
        ),
        now=FIXED_NOW,
    )
    assert row["reason"] == "blank(error); corrupt(warning)"


def test_build_result_row_none_scores_become_empty_strings():
    row = build_result_row(
        make_result(defect_score=None, confidence_signal=None),
        now=FIXED_NOW,
    )
    assert row["defect_score"] == ""
    assert row["confidence_signal"] == ""
