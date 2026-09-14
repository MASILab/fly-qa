#!/usr/bin/env python3
"""Honest held-out check of the calibrated decoder against real labeled TractSeg
QA data from a study folder. Point this at whatever study(s) you passed to
scripts/calibrate_on_real_labels.py --exclude-study -- it must never have been
part of calibration, or this isn't a real held-out check.

Usage:
    python scripts/validate_on_real_labels.py examples/study_226 \\
        --connectome-export ~/.cache/fly_qa/connectome_export
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

from fly_qa.cli import build_pipeline_config, build_simulator
from fly_qa.pipeline import run_pipeline
from fly_qa.thresholds_store import load_thresholds

BANNER = "*** EXPERIMENTAL / UNVALIDATED -- research/art exercise, not a validated defect classifier ***"


def iter_labeled_examples_in(study_dir: Path, process_glob: str = "Tractseg_*"):
    """Supports both nested ({study}/{process}/QA.csv) and flat ({study}/QA.csv) layouts --
    see calibrate_on_real_labels.iter_labeled_examples for the full explanation, including
    why a flat QA.csv's path-prefixed filename rows (aggregate rollups) are skipped."""
    for qa_path in sorted(study_dir.glob(f"{process_glob}/QA.csv")):
        with qa_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            status = row.get("QA_status", "")
            if status not in ("yes", "no", "maybe"):
                continue
            png_path = qa_path.parent / row["filename"]
            if png_path.exists():
                yield png_path, status

    flat_qa = study_dir / "QA.csv"
    if flat_qa.is_file():
        with flat_qa.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            status = row.get("QA_status", "")
            filename = row.get("filename", "")
            if status not in ("yes", "no", "maybe") or "/" in filename or "\\" in filename:
                continue
            png_path = flat_qa.parent / filename
            if png_path.exists():
                yield png_path, status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path, help="e.g. examples/study_226")
    parser.add_argument(
        "--connectome-export", type=Path, default=Path.home() / ".cache" / "fly_qa" / "connectome_export",
    )
    parser.add_argument("--leak", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--process-glob", type=str, default="Tractseg_*", metavar="PATTERN",
        help="Must match what you passed to calibrate_on_real_labels.py --process-glob.",
    )
    parser.add_argument(
        "--thresholds", type=Path, default=None,
        help="Path to the calibrated thresholds file (default: the shipped one -- "
             "must match --output-thresholds from calibrate_on_real_labels.py if you used it).",
    )
    args = parser.parse_args(argv)

    print(BANNER)
    thresholds = load_thresholds(args.thresholds)
    if not thresholds.calibrated:
        print("error: no calibrated thresholds found -- run scripts/calibrate_on_real_labels.py first", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    good_paths, bad_paths = [], []
    for path, status in iter_labeled_examples_in(args.study_dir, args.process_glob):
        (good_paths if status == "yes" else bad_paths).append(path)

    rng.shuffle(good_paths)
    good_paths = good_paths[: args.max_per_class]
    bad_paths = bad_paths[: args.max_per_class]
    print(f"Held-out check on {args.study_dir}: {len(good_paths)} real 'yes', {len(bad_paths)} real 'no'/'maybe' "
          f"(fail<={thresholds.confidence_fail_max:.6g}, flag<={thresholds.confidence_flag_max:.6g})")

    print(f"Loading connectome from {args.connectome_export}...")
    t0 = time.perf_counter()
    simulator, neurons = build_simulator(args.connectome_export, args.leak, args.steps)
    config = build_pipeline_config(neurons, grid_size=(256, 256))
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    tp = fp = tn = fn = 0
    t0 = time.perf_counter()
    all_paths = [(p, True) for p in bad_paths] + [(p, False) for p in good_paths]
    for i, (path, is_bad) in enumerate(all_paths, 1):
        result = run_pipeline(path, simulator, config, thresholds)
        predicted_bad = result.verdict != "pass"
        if is_bad and predicted_bad:
            tp += 1
        elif is_bad and not predicted_bad:
            fn += 1
        elif not is_bad and predicted_bad:
            fp += 1
        else:
            tn += 1
        if i % 25 == 0:
            print(f"  processed {i}/{len(all_paths)} ({time.perf_counter() - t0:.0f}s elapsed)")

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    print(f"\nHeld-out ({args.study_dir}, never used for calibration):")
    print(f"  sensitivity (recall on real no/maybe) = {sensitivity:.1%}  ({tp}/{tp+fn})")
    print(f"  specificity (real yes correctly passed) = {specificity:.1%}  ({tn}/{tn+fp})")
    print("matches this rule on this held-out set -- NOT a validated defect classifier.")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
