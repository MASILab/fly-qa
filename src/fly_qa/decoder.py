"""Section 3: decoder (connectome output -> QA verdict).

Reuses the DoomFly/Stonkfly descending-neuron pattern -- relabeled, not reinvented:
  - DNp20 right-minus-left rate  -> a lateral/differential readout
  - Summed DNpe017 rate          -> a magnitude/summed readout

**Evidence-based deviation from the original spec mapping, recorded here rather
than silently changed:** the spec's own DoomFly precedent uses the DNp20 L-R
difference for "turning" (an inherently lateral action) and the DNpe017 sum for
"moving" (a magnitude action). This project initially carried that mapping over
literally, using DNp20 diff as the primary defect score. Diagnostic evidence
against real labeled TractSeg images (see git history / session notes around
the "failures pass" bug report) showed that's the wrong readout for this
encoder: because the encoder maps whole-image content onto R1-R6/R8 body IDs in
an order with no relationship to real left/right hemisphere anatomy, a genuine,
symmetric change in image content (e.g. a sparser tract overlay) drives both
DNp20 neurons roughly equally -- so the *difference* cancels out the very
signal we want, and measured worse (66.7% best-threshold separation on n=54
real labeled images) than even the raw, un-simulated encoder input (70.4%).
The DNpe017 *sum* preserves that signal instead of cancelling it (72.2%
separation) because it doesn't difference two neurons expected to move
together. The decoder below therefore thresholds on `confidence_signal` (LOW
sum -> more likely defective), not `defect_score`. `defect_score` is still
computed and returned for transparency/future work but does not drive the
verdict. This is still an invented proxy relationship, not a validated one --
rerun scripts/run_validation.py after any encoder/simulator change and recheck
which readout actually separates real labels before trusting either one.

Thresholds are NOT hardcoded here: they must come from calibration against a
labeled set (Section 5). Until calibration has run, `DecoderThresholds` should be
treated as a placeholder and the tool must say so wherever the verdict is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fly_qa.simulator import SimulationResult


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    FLAG = "flag"


@dataclass(frozen=True)
class DecoderThresholds:
    # Verdict is "fail" when confidence_signal <= confidence_fail_max, "flag" when
    # confidence_signal <= confidence_flag_max (and above the fail threshold).
    # LOW confidence is the "something's wrong" direction -- see module docstring.
    confidence_fail_max: float
    confidence_flag_max: float
    calibrated: bool = False  # False until Section 5 calibration has actually run


@dataclass(frozen=True)
class DecoderOutput:
    defect_score: float
    confidence_signal: float
    verdict: Verdict
    calibrated: bool


def decode(
    result: SimulationResult,
    dnp20_left_id: int,
    dnp20_right_id: int,
    dnpe017_ids: list[int],
    thresholds: DecoderThresholds,
) -> DecoderOutput:
    defect_score = result.rate(dnp20_right_id) - result.rate(dnp20_left_id)
    confidence_signal = sum(result.rate(bid) for bid in dnpe017_ids)

    if confidence_signal <= thresholds.confidence_fail_max:
        verdict = Verdict.FAIL
    elif confidence_signal <= thresholds.confidence_flag_max:
        verdict = Verdict.FLAG
    else:
        verdict = Verdict.PASS

    return DecoderOutput(
        defect_score=defect_score,
        confidence_signal=confidence_signal,
        verdict=verdict,
        calibrated=thresholds.calibrated,
    )
