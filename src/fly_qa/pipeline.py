"""Shared precheck -> encode -> simulate -> decode pipeline, used by both the CLI
and the validation script so calibration measures exactly what production runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fly_qa.decoder import DecoderThresholds, decode
from fly_qa.encoder import encode_image
from fly_qa.precheck import run_precheck
from fly_qa.simulator import ConnectomeSimulator

R8_TYPES = ["R8_unclear", "R8d", "R8p", "R8y", "R7R8_unclear"]


@dataclass(frozen=True)
class PipelineConfig:
    r1_r6_ids: list[int]
    r8_ids: list[int]
    dnp20_left_id: int
    dnp20_right_id: int
    dnpe017_ids: list[int]
    grid_size: tuple[int, int] = (256, 256)


@dataclass(frozen=True)
class PipelineResult:
    path: Path
    precheck_passed: bool
    precheck_findings: list[str]
    fed_to_connectome: bool
    defect_score: float | None
    confidence_signal: float | None
    verdict: str
    calibrated: bool
    duration_sec: float
    # Real per-step activity for whatever body IDs were requested via
    # `capture_body_ids` (dashboard visualization only) -- shape (steps, len(capture_body_ids)),
    # in the same order as capture_body_ids. None if not requested or precheck failed.
    step_activity: np.ndarray | None = None


def run_pipeline(
    path: Path,
    simulator: ConnectomeSimulator,
    config: PipelineConfig,
    thresholds: DecoderThresholds,
    capture_body_ids: list[int] | None = None,
) -> PipelineResult:
    t0 = time.perf_counter()
    precheck_result = run_precheck(path)
    finding_strs = [f"{f.code}({f.severity})" for f in precheck_result.findings]

    if not precheck_result.passed or precheck_result.image_rgba is None:
        return PipelineResult(
            path=path,
            precheck_passed=precheck_result.passed,
            precheck_findings=finding_strs,
            fed_to_connectome=False,
            defect_score=None,
            confidence_signal=None,
            verdict="fail",
            calibrated=thresholds.calibrated,
            duration_sec=round(time.perf_counter() - t0, 3),
            step_activity=None,
        )

    encoded = encode_image(
        precheck_result.image_rgba.convert("RGB"), config.r1_r6_ids, config.r8_ids, grid_size=config.grid_size
    )
    external_input = {**encoded.r1_r6_current, **encoded.r8_current}
    sim_result = simulator.run(external_input, capture_body_ids=capture_body_ids)
    decoded = decode(sim_result, config.dnp20_left_id, config.dnp20_right_id, config.dnpe017_ids, thresholds)

    return PipelineResult(
        path=path,
        precheck_passed=True,
        precheck_findings=finding_strs,
        fed_to_connectome=True,
        defect_score=decoded.defect_score,
        confidence_signal=decoded.confidence_signal,
        verdict=decoded.verdict.value,
        calibrated=thresholds.calibrated,
        duration_sec=round(time.perf_counter() - t0, 3),
        step_activity=sim_result.step_snapshots,
    )
