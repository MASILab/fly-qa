"""Load/save calibrated decoder thresholds (Section 5 output).

Deliberately separate from `decoder.py`'s `DecoderThresholds` dataclass: this
module only handles the JSON file on disk, so `decoder.py` stays free of I/O.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from fly_qa.decoder import DecoderThresholds

DEFAULT_THRESHOLDS_PATH = resources.files("fly_qa.data") / "decoder_thresholds.json"

# Placeholder only -- these numbers have NOT been calibrated against any labeled
# set. `calibrated=False` propagates through decode() so every downstream log
# line and report says so explicitly. Run scripts/run_validation.py to replace
# this file with real, calibrated values.
_UNCALIBRATED_PLACEHOLDER = DecoderThresholds(
    confidence_fail_max=0.0, confidence_flag_max=0.0, calibrated=False
)


def load_thresholds(path: Path | None = None) -> DecoderThresholds:
    source = path if path is not None else DEFAULT_THRESHOLDS_PATH
    try:
        with open(source, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return _UNCALIBRATED_PLACEHOLDER

    return DecoderThresholds(
        confidence_fail_max=data["confidence_fail_max"],
        confidence_flag_max=data["confidence_flag_max"],
        calibrated=data.get("calibrated", False),
    )


def save_thresholds(
    path: Path,
    thresholds: DecoderThresholds,
    calibration_notes: dict | None = None,
) -> None:
    data = {
        "confidence_fail_max": thresholds.confidence_fail_max,
        "confidence_flag_max": thresholds.confidence_flag_max,
        "calibrated": thresholds.calibrated,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": calibration_notes or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
