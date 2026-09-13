import numpy as np

from fly_qa.decoder import DecoderThresholds, Verdict, decode
from fly_qa.simulator import SimulationResult

# confidence_fail_max=0.2: verdict is "fail" when confidence_signal <= 0.2 (LOW
# confidence is the "something's wrong" direction -- see decoder.py docstring
# for the evidence behind using confidence_signal, not defect_score, here).
THRESHOLDS = DecoderThresholds(confidence_fail_max=0.2, confidence_flag_max=0.5, calibrated=True)


def _result(rates: dict[int, float]) -> SimulationResult:
    ids = sorted(rates)
    id_to_index = {bid: i for i, bid in enumerate(ids)}
    arr = np.array([rates[bid] for bid in ids])
    return SimulationResult(final_rates=arr, id_to_index=id_to_index)


def test_low_confidence_is_fail():
    result = _result({1: 0.1, 2: 0.8, 3: 0.1})  # confidence (sum of dnpe017) = 0.1
    out = decode(result, dnp20_left_id=1, dnp20_right_id=2, dnpe017_ids=[3], thresholds=THRESHOLDS)
    assert out.verdict == Verdict.FAIL
    assert abs(out.confidence_signal - 0.1) < 1e-9


def test_borderline_confidence_is_flag():
    result = _result({1: 0.1, 2: 0.8, 3: 0.35})  # confidence = 0.35, between fail and flag thresholds
    out = decode(result, dnp20_left_id=1, dnp20_right_id=2, dnpe017_ids=[3], thresholds=THRESHOLDS)
    assert out.verdict == Verdict.FLAG


def test_high_confidence_is_pass():
    result = _result({1: 0.5, 2: 0.5, 3: 0.9})  # confidence = 0.9, above both thresholds
    out = decode(result, dnp20_left_id=1, dnp20_right_id=2, dnpe017_ids=[3], thresholds=THRESHOLDS)
    assert out.verdict == Verdict.PASS


def test_defect_score_still_computed_but_does_not_drive_verdict():
    # Large DNp20 L-R difference, but confidence is high -> still a pass.
    # defect_score is reported for transparency/future work only (see decoder.py
    # docstring for why it was dropped from the verdict decision).
    result = _result({1: 0.0, 2: 0.9, 3: 0.9})  # diff = 0.9 (would have failed the old logic)
    out = decode(result, dnp20_left_id=1, dnp20_right_id=2, dnpe017_ids=[3], thresholds=THRESHOLDS)
    assert abs(out.defect_score - 0.9) < 1e-9
    assert out.verdict == Verdict.PASS


def test_confidence_signal_sums_dnpe017():
    result = _result({1: 0.0, 2: 0.0, 3: 0.4, 4: 0.6})
    out = decode(result, dnp20_left_id=1, dnp20_right_id=2, dnpe017_ids=[3, 4], thresholds=THRESHOLDS)
    assert abs(out.confidence_signal - 1.0) < 1e-9


def test_uncalibrated_thresholds_propagate_flag():
    uncalibrated = DecoderThresholds(confidence_fail_max=0.2, confidence_flag_max=0.5, calibrated=False)
    result = _result({1: 0.0, 2: 0.0, 3: 0.0})
    out = decode(result, dnp20_left_id=1, dnp20_right_id=2, dnpe017_ids=[3], thresholds=uncalibrated)
    assert out.calibrated is False
