#!/usr/bin/env python3
"""Calibrate and validate the connectome pipeline against a simple two-folder
labeled dataset (all PNGs in --good-dir = pass, all PNGs in --bad-dir = fail),
splitting each into calibration/held-out halves. Generic -- not TractSeg-
specific -- useful for toy sanity checks like apples-vs-oranges.

Usage:
    python scripts/calibrate_and_validate_folders.py \\
        --good-dir .devtest/fruit_demo/apples --bad-dir .devtest/fruit_demo/oranges \\
        --connectome-export ~/.cache/fly_qa/connectome_export \\
        --output-thresholds .devtest/fruit_demo/decoder_thresholds.json
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from fly_qa.cli import build_pipeline_config, build_graph_and_simulator
from fly_qa.decoder import DecoderThresholds
from fly_qa.pipeline import run_pipeline
from fly_qa.thresholds_store import save_thresholds

BANNER = "*** EXPERIMENTAL / UNVALIDATED -- research/art exercise, not a validated defect classifier ***"


def _largest_threshold_within_budget(good_values: list[float], candidates: list[float], max_false_flags: int) -> float:
    best = candidates[0] - 1.0 if candidates else 0.0
    for c in candidates:
        if sum(1 for v in good_values if v <= c) <= max_false_flags:
            best = c
        else:
            break
    return best


def _optimal_threshold(good_scores: list[float], bad_scores: list[float]) -> tuple[float, float, float]:
    """Threshold maximizing Youden's J (sensitivity + specificity - 1) on the given
    scores -- the standard "optimal cutpoint" for a binary classifier, balancing both
    error types equally. Returns (threshold, sensitivity, specificity) at that point.

    This is a diagnostic/reference number, not necessarily what should be deployed --
    a real QA tool usually wants a false-flag-rate BUDGET (see --target-false-flag-rate),
    not the accuracy-maximizing point, since under-flagging real defects is normally
    worse than over-flagging good images by the same amount.
    """
    candidates = sorted(set(good_scores) | set(bad_scores))
    if not candidates:
        return 0.0, float("nan"), float("nan")

    best_j, best = -1.0, (candidates[0], 0.0, 0.0)
    for c in candidates:
        tp = sum(1 for s in bad_scores if s <= c)
        fp = sum(1 for s in good_scores if s <= c)
        sensitivity = tp / len(bad_scores) if bad_scores else 0.0
        specificity = 1 - fp / len(good_scores) if good_scores else 0.0
        j = sensitivity + specificity - 1
        if j > best_j:
            best_j, best = j, (c, sensitivity, specificity)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good-dir", type=Path, required=True)
    parser.add_argument("--bad-dir", type=Path, required=True)
    parser.add_argument("--connectome-export", type=Path, default=Path.home() / ".cache" / "fly_qa" / "connectome_export")
    parser.add_argument("--leak", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--target-false-flag-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-thresholds", type=Path, default=None, help="If set, write calibrated thresholds here")
    args = parser.parse_args(argv)

    print(BANNER)
    good_paths = sorted(args.good_dir.glob("*.png"))
    bad_paths = sorted(args.bad_dir.glob("*.png"))
    if not good_paths or not bad_paths:
        print(f"error: found {len(good_paths)} good / {len(bad_paths)} bad PNGs -- need both", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(good_paths)
    rng.shuffle(bad_paths)
    n_good, n_bad = len(good_paths) // 2, len(bad_paths) // 2
    cal_good, held_good = good_paths[:n_good], good_paths[n_good:]
    cal_bad, held_bad = bad_paths[:n_bad], bad_paths[n_bad:]

    print(f"Calibration: {len(cal_good)} good + {len(cal_bad)} bad")
    print(f"Held-out:    {len(held_good)} good + {len(held_bad)} bad")

    print(f"Loading connectome from {args.connectome_export}...")
    t0 = time.perf_counter()
    graph, simulator = build_graph_and_simulator(args.connectome_export, args.leak, args.steps)
    config = build_pipeline_config(graph.neurons, grid_size=(256, 256))
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    uncal = DecoderThresholds(confidence_fail_max=float("-inf"), confidence_flag_max=float("-inf"), calibrated=False)

    def run_all(paths: list[Path], label: str) -> list[float]:
        t0 = time.perf_counter()
        scores = [run_pipeline(p, simulator, config, uncal).confidence_signal for p in paths]
        print(f"  {label}: {len(paths)} images in {time.perf_counter() - t0:.0f}s")
        return scores

    cal_good_scores = run_all(cal_good, "calibration good")
    cal_bad_scores = run_all(cal_bad, "calibration bad")
    held_good_scores = run_all(held_good, "held-out good")
    held_bad_scores = run_all(held_bad, "held-out bad")

    candidates = sorted(set(cal_good_scores) | set(cal_bad_scores))
    max_fail_flags = int(len(cal_good_scores) * args.target_false_flag_rate)
    fail_max = _largest_threshold_within_budget(cal_good_scores, candidates, max_fail_flags)
    max_flag_flags = int(len(cal_good_scores) * args.target_false_flag_rate * 2)
    flag_max = max(_largest_threshold_within_budget(cal_good_scores, candidates, max_flag_flags), fail_max)

    thresholds = DecoderThresholds(confidence_fail_max=fail_max, confidence_flag_max=flag_max, calibrated=True)
    print(f"\nPicked thresholds (fail-rate budget={args.target_false_flag_rate:.0%}): "
          f"fail<={fail_max:.6g}, flag<={flag_max:.6g}")

    opt_threshold, opt_sensitivity, opt_specificity = _optimal_threshold(cal_good_scores, cal_bad_scores)
    print(f"Optimal threshold (max sensitivity+specificity on calibration set): "
          f"confidence<={opt_threshold:.6g}  ->  sensitivity={opt_sensitivity:.1%}, specificity={opt_specificity:.1%}")
    print("  (reference only -- the budgeted threshold above is what's saved/deployed; "
          "pass --confidence-fail-max on `fly-qa` if you want to use this one instead)")

    def report(name: str, good_scores: list[float], bad_scores: list[float]) -> None:
        tp = sum(1 for s in bad_scores if s <= fail_max)
        fp = sum(1 for s in good_scores if s <= fail_max)
        sensitivity = tp / len(bad_scores) if bad_scores else float("nan")
        specificity = 1 - fp / len(good_scores) if good_scores else float("nan")
        print(f"[{name}] n={len(good_scores) + len(bad_scores)} sensitivity={sensitivity:.1%} "
              f"specificity={specificity:.1%} -- matches this rule on this set, NOT a validated classifier.")

    print()
    report("calibration (in-sample)", cal_good_scores, cal_bad_scores)
    report("held-out (frozen weights, never used for tuning)", held_good_scores, held_bad_scores)

    if args.output_thresholds:
        save_thresholds(
            args.output_thresholds, thresholds,
            calibration_notes={
                "good_dir": str(args.good_dir), "bad_dir": str(args.bad_dir),
                "n_calibration": len(cal_good) + len(cal_bad), "n_heldout": len(held_good) + len(held_bad),
                "leak": args.leak, "steps": args.steps,
            },
        )
        print(f"\nWrote {args.output_thresholds}")

    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
