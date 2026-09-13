#!/usr/bin/env python3
"""Section 5: validation protocol.

Builds a labeled calibration set (known-good real PNGs + synthetic defects at
known severities), picks decoder thresholds from ONLY the calibration set,
then evaluates the frozen-weight pipeline against a SEPARATE held-out labeled
set it never saw during threshold-picking. Reports sensitivity/specificity
honestly -- this is "matches this rule on this calibration set," not a claim
that the decoder learned anything.

Thresholds on `confidence_signal` (DNpe017 sum), not `defect_score` (DNp20
L-R diff) -- see decoder.py's module docstring for the diagnostic evidence
behind that choice (defect_score measured *worse* than the raw un-simulated
encoder input on real labeled images; confidence_signal measured better).

Usage:
    python scripts/run_validation.py --good-images-dir examples/study_256/Tractseg_CA \\
        --connectome-export ~/.cache/fly_qa/connectome_export
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from fly_qa.cli import build_pipeline_config, build_simulator
from fly_qa.decoder import DecoderThresholds
from fly_qa.pipeline import run_pipeline
from fly_qa.thresholds_store import DEFAULT_THRESHOLDS_PATH, save_thresholds
from fly_qa.validation import DefectType, LabeledImage, build_calibration_set, evaluate

BANNER = "*** EXPERIMENTAL / UNVALIDATED -- research/art exercise, not a validated defect classifier ***"


def _largest_threshold_within_budget(
    good_values: list[float], candidates: list[float], max_false_flags: int
) -> float:
    """Largest T such that flagging value<=T keeps count(good_values<=T) <= max_false_flags.

    Candidate-based (not rank-index-based) to stay correct under heavy ties -- see
    the old heuristic classifier's calibration history for why rank-index breaks
    when many values coincide. Candidates must be sorted ascending.
    """
    best = candidates[0] - 1.0 if candidates else 0.0  # default: nothing flagged
    for candidate in candidates:
        false_flags = sum(1 for v in good_values if v <= candidate)
        if false_flags <= max_false_flags:
            best = candidate
        else:
            break
    return best


def pick_thresholds_from_calibration(
    calibration_labeled: list[LabeledImage],
    scores_by_path: dict[Path, float],
    target_false_flag_rate: float = 0.05,
) -> DecoderThresholds:
    """Pick confidence thresholds: verdict is "fail" when confidence<=confidence_fail_max,
    keeping the false-positive rate on known-good calibration images at or below target.
    """
    good_scores = sorted(scores_by_path[i.path] for i in calibration_labeled if not i.is_defective)
    bad_scores = sorted(scores_by_path[i.path] for i in calibration_labeled if i.is_defective)
    candidates = sorted(set(good_scores) | set(bad_scores))

    n_good = len(good_scores)
    max_false_flags = int(n_good * target_false_flag_rate)
    fail_max = _largest_threshold_within_budget(good_scores, candidates, max_false_flags)

    max_flag_flags = int(n_good * target_false_flag_rate * 2)
    flag_max = _largest_threshold_within_budget(good_scores, candidates, max_flag_flags)
    flag_max = max(flag_max, fail_max)

    return DecoderThresholds(confidence_fail_max=fail_max, confidence_flag_max=flag_max, calibrated=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--good-images-dir", type=Path, required=True,
        help="Directory of known-good real PNGs to build calibration/held-out sets from",
    )
    parser.add_argument(
        "--connectome-export", type=Path, default=Path.home() / ".cache" / "fly_qa" / "connectome_export",
    )
    parser.add_argument("--n-calibration-images", type=int, default=12)
    parser.add_argument("--n-heldout-images", type=int, default=12)
    parser.add_argument("--defects-per-image", type=int, default=4)
    parser.add_argument("--leak", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-thresholds", type=Path, default=Path(str(DEFAULT_THRESHOLDS_PATH)))
    parser.add_argument("--work-dir", type=Path, default=Path(".devtest/validation"))
    args = parser.parse_args(argv)

    print(BANNER)

    all_good = sorted(args.good_images_dir.glob("*.png"))
    needed = args.n_calibration_images + args.n_heldout_images
    if len(all_good) < needed:
        print(f"error: found only {len(all_good)} PNGs in {args.good_images_dir}, need {needed}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    shuffled = all_good.copy()
    rng.shuffle(shuffled)
    calibration_sources = shuffled[: args.n_calibration_images]
    heldout_sources = shuffled[args.n_calibration_images : needed]

    print(f"Building calibration set ({len(calibration_sources)} good images x "
          f"{args.defects_per_image} defects each)...")
    calibration_labeled = build_calibration_set(
        calibration_sources, args.work_dir / "calibration", defects_per_image=args.defects_per_image, seed=args.seed
    )
    print(f"Building held-out set ({len(heldout_sources)} good images, NEVER used for threshold-picking)...")
    heldout_labeled = build_calibration_set(
        heldout_sources, args.work_dir / "heldout", defects_per_image=args.defects_per_image, seed=args.seed + 1
    )

    print(f"Loading connectome from {args.connectome_export}...")
    t0 = time.perf_counter()
    simulator, neurons = build_simulator(args.connectome_export, args.leak, args.steps)
    config = build_pipeline_config(neurons, grid_size=(256, 256))
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    uncalibrated = DecoderThresholds(confidence_fail_max=float("-inf"), confidence_flag_max=float("-inf"), calibrated=False)

    print(f"Running frozen-weight simulation over {len(calibration_labeled)} calibration images...")
    cal_scores: dict[Path, float] = {}
    for item in calibration_labeled:
        result = run_pipeline(item.path, simulator, config, uncalibrated)
        cal_scores[item.path] = result.confidence_signal if result.confidence_signal is not None else 0.0

    thresholds = pick_thresholds_from_calibration(calibration_labeled, cal_scores)
    print(f"Picked thresholds from calibration set: fail<={thresholds.confidence_fail_max:.6g}, "
          f"flag<={thresholds.confidence_flag_max:.6g}")

    print(f"Running frozen-weight simulation over {len(heldout_labeled)} held-out images "
          f"(never used for threshold-picking)...")
    heldout_scores: dict[Path, float] = {}
    for item in heldout_labeled:
        result = run_pipeline(item.path, simulator, config, uncalibrated)
        heldout_scores[item.path] = result.confidence_signal if result.confidence_signal is not None else 0.0

    def verdict_for(score: float) -> str:
        if score <= thresholds.confidence_fail_max:
            return "fail"
        if score <= thresholds.confidence_flag_max:
            return "flag"
        return "pass"

    cal_report = evaluate(calibration_labeled, lambda p: verdict_for(cal_scores[p]))
    heldout_report = evaluate(heldout_labeled, lambda p: verdict_for(heldout_scores[p]))

    print()
    print(cal_report.summary("calibration set (in-sample, thresholds picked here)"))
    print(heldout_report.summary("held-out set (frozen weights, never used for tuning)"))
    print()
    print("Per-defect-type breakdown on held-out set:")
    for defect_type in DefectType:
        subset = [i for i in heldout_labeled if i.defect_type == defect_type]
        if not subset:
            continue
        caught = sum(1 for i in subset if heldout_scores[i.path] <= thresholds.confidence_fail_max)
        print(f"  {defect_type.value:20s}: {caught}/{len(subset)} flagged as fail")

    save_thresholds(
        args.output_thresholds,
        thresholds,
        calibration_notes={
            "n_calibration_images": len(calibration_labeled),
            "n_heldout_images": len(heldout_labeled),
            "heldout_sensitivity": heldout_report.sensitivity,
            "heldout_specificity": heldout_report.specificity,
            "leak": args.leak,
            "steps": args.steps,
            "source_dir": str(args.good_images_dir),
        },
    )
    print(f"\nWrote {args.output_thresholds}")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
