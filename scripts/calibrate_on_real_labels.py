#!/usr/bin/env python3
"""Calibrate decoder thresholds against REAL labeled TractSeg QA data, not
synthetic defects. scripts/run_validation.py (Section 5 of the build spec)
injects generic PNG defects (banding, color shift, corruption, alpha loss,
resolution mismatch) -- useful for the spec's generic-PNG validation protocol,
but a different defect distribution than what this deployment actually needs
to catch (sparse/degraded tractogram renders, per the real examples/ QA.csv
labels). This script calibrates against the real thing.

Pass --exclude-study for every study you want reserved as a genuine
held-out check (run separately by scripts/validate_on_real_labels.py) --
required, not defaulted, so a held-out study can never be silently included
in calibration by forgetting the flag.

Usage:
    python scripts/calibrate_on_real_labels.py examples/ \\
        --exclude-study study_226 \\
        --connectome-export ~/.cache/fly_qa/connectome_export

    # multiple held-out studies:
    python scripts/calibrate_on_real_labels.py my_pipelines/ \\
        --exclude-study "study #3" --exclude-study "study #7" \\
        --connectome-export ~/.cache/fly_qa/connectome_export
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from fly_qa.cli import build_pipeline_config, build_simulator
from fly_qa.decoder import DecoderThresholds
from fly_qa.pipeline import run_pipeline
from fly_qa.thresholds_store import DEFAULT_THRESHOLDS_PATH, save_thresholds

BANNER = "*** EXPERIMENTAL / UNVALIDATED -- research/art exercise, not a validated defect classifier ***"
TARGET_FALSE_FLAG_RATE = 0.05


def iter_labeled_examples(root: Path, exclude: set[str], process_glob: str = "Tractseg_*"):
    """Supports two QA.csv layouts, both seen in real datasets:
      - nested:  {study}/{process}/QA.csv + PNGs (e.g. Tractseg_AFleft/, matched via process_glob)
      - flat:    {study}/QA.csv + PNGs directly in the study folder (no process level)
    Both are checked unconditionally for every study -- harmless no-op if a layout isn't
    present. A study-level QA.csv whose filenames contain a path separator (e.g.
    "BRAID/sub-....png") is an aggregate rollup referencing a process subfolder, not a
    flat layout -- those rows are skipped here since the same images are already covered
    by the nested pass above; including them too would silently double-count them.
    """
    for study_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if study_dir.name in exclude:
            continue
        nested_qa_paths = sorted(study_dir.glob(f"{process_glob}/QA.csv"))
        flat_qa = study_dir / "QA.csv"

        for qa_path in nested_qa_paths:
            with qa_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                status = row.get("QA_status", "")
                if status not in ("yes", "no", "maybe"):
                    continue
                png_path = qa_path.parent / row["filename"]
                if png_path.exists():
                    yield png_path, status

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


def _largest_threshold_within_budget(good_values, candidates, max_false_flags) -> float:
    best = candidates[0] - 1.0 if candidates else 0.0
    for candidate in candidates:
        false_flags = sum(1 for v in good_values if v <= candidate)
        if false_flags <= max_false_flags:
            best = candidate
        else:
            break
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root directory of labeled example studies (e.g. examples/)")
    parser.add_argument(
        "--connectome-export", type=Path, default=Path.home() / ".cache" / "fly_qa" / "connectome_export",
    )
    parser.add_argument("--leak", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-per-class", type=int, default=150, help="Cap on yes/non-yes examples each, for runtime")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-thresholds", type=Path, default=Path(str(DEFAULT_THRESHOLDS_PATH)))
    parser.add_argument(
        "--exclude-study", action="append", required=True, metavar="NAME",
        help="Study folder name to hold out from calibration entirely (repeatable). "
             "Required -- there is no default -- so a held-out study can never be "
             "silently included in calibration by forgetting this flag.",
    )
    parser.add_argument(
        "--process-glob", type=str, default="Tractseg_*", metavar="PATTERN",
        help="Glob pattern (relative to each study folder) for process folders "
             "containing a QA.csv, e.g. 'Tractseg_*' (default, matches the original "
             "TractSeg examples/ dataset) or '*' to match every process folder "
             "regardless of naming, for a differently-structured dataset.",
    )
    args = parser.parse_args(argv)

    print(BANNER)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 1

    excluded = set(args.exclude_study)
    import random

    rng = random.Random(args.seed)
    good_paths, bad_paths = [], []
    for path, status in iter_labeled_examples(args.root, excluded, args.process_glob):
        (good_paths if status == "yes" else bad_paths).append(path)

    rng.shuffle(good_paths)
    rng.shuffle(bad_paths)
    good_paths = good_paths[: args.max_per_class]
    # use ALL real non-yes examples (there are few) up to the cap
    bad_paths = bad_paths[: args.max_per_class]

    print(f"Calibrating on {len(good_paths)} real 'yes' + {len(bad_paths)} real 'no'/'maybe' images "
          f"from {args.root} (excluding {excluded})")

    print(f"Loading connectome from {args.connectome_export}...")
    t0 = time.perf_counter()
    simulator, neurons = build_simulator(args.connectome_export, args.leak, args.steps)
    config = build_pipeline_config(neurons, grid_size=(256, 256))
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    uncalibrated = DecoderThresholds(confidence_fail_max=float("-inf"), confidence_flag_max=float("-inf"), calibrated=False)

    good_scores, bad_scores = [], []
    t0 = time.perf_counter()
    for i, path in enumerate(good_paths + bad_paths, 1):
        result = run_pipeline(path, simulator, config, uncalibrated)
        score = result.confidence_signal if result.confidence_signal is not None else 0.0
        (good_scores if path in good_paths else bad_scores).append(score)
        if i % 25 == 0:
            print(f"  processed {i}/{len(good_paths) + len(bad_paths)} ({time.perf_counter() - t0:.0f}s elapsed)")

    good_scores.sort()
    bad_scores.sort()
    candidates = sorted(set(good_scores) | set(bad_scores))
    n_good = len(good_scores)
    max_false_flags = int(n_good * TARGET_FALSE_FLAG_RATE)
    fail_max = _largest_threshold_within_budget(good_scores, candidates, max_false_flags)
    max_flag_flags = int(n_good * TARGET_FALSE_FLAG_RATE * 2)
    flag_max = max(_largest_threshold_within_budget(good_scores, candidates, max_flag_flags), fail_max)

    thresholds = DecoderThresholds(confidence_fail_max=fail_max, confidence_flag_max=flag_max, calibrated=True)

    caught = sum(1 for s in bad_scores if s <= fail_max)
    false_flags = sum(1 for s in good_scores if s <= fail_max)
    print(f"\nPicked: fail<={fail_max:.6g}, flag<={flag_max:.6g}")
    print(f"In-sample: caught {caught}/{len(bad_scores)} real no/maybe ({caught/len(bad_scores)*100 if bad_scores else 0:.1f}%), "
          f"false-flagged {false_flags}/{n_good} real yes ({false_flags/n_good*100 if n_good else 0:.1f}%)")
    print(f"Run scripts/validate_on_real_labels.py against {', '.join(sorted(excluded))} for an honest held-out check.")

    save_thresholds(
        args.output_thresholds,
        thresholds,
        calibration_notes={
            "calibration_method": "real_labels",
            "n_good": n_good,
            "n_bad": len(bad_scores),
            "in_sample_recall": caught / len(bad_scores) if bad_scores else None,
            "in_sample_false_flag_rate": false_flags / n_good if n_good else None,
            "leak": args.leak,
            "steps": args.steps,
            "source_root": str(args.root),
            "excluded_studies": sorted(excluded),
            "process_glob": args.process_glob,
        },
    )
    print(f"Wrote {args.output_thresholds}")
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
