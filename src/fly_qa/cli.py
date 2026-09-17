"""fly-qa: PNGQAFly -- a connectome-driven image QA scanner.

*** EXPERIMENTAL / UNVALIDATED ***
This reuses the retained MaleCNS v1.0 connectome (~165K neurons, ~25M synapses,
UNMODIFIED) with a task-specific sensory encoder and descending-neuron decoder.
It is a research/art exercise, following the same architecture as nftechie/doomfly
and nftechie/stonkfly -- it does NOT constitute a validated defect classifier.
No accuracy claim is meaningful until scripts/run_validation.py has actually run
against a held-out labeled set; even then, report it as "matches this rule on
this calibration set," not "learned to detect defects."
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

from fly_qa.connectome_data import (
    ConnectomeGraph,
    build_connectome_graph,
    body_ids_by_type,
    load_connectome_export,
)
from fly_qa.decoder import DecoderThresholds
from fly_qa.events import EventBus, ResultEvent
from fly_qa.pipeline import R8_TYPES, PipelineConfig, PipelineResult, run_pipeline
from fly_qa.simulator import ConnectomeSimulator
from fly_qa.thresholds_store import load_thresholds
from fly_qa.viz import pick_visualization_subset

BANNER = "*** EXPERIMENTAL / UNVALIDATED -- research/art exercise, not a validated defect classifier ***"

# MASI-QA core columns (https://github.com/MASILab/masi-qa QA.csv), followed by
# fly_qa-specific detail columns not present in that format.
RESULT_FIELDNAMES = [
    "filename",
    "QA_status",
    "reason",
    "user",
    "date",
    "duration",
    "verdict",
    "defect_score",
    "confidence_signal",
    "precheck_passed",
    "precheck_findings",
    "fed_to_connectome",
    "calibrated",
]

# fly_qa verdicts (pass/fail/flag, see CLAUDE.md) map onto MASI-QA's QA_status vocabulary.
VERDICT_TO_QA_STATUS = {"pass": "yes", "fail": "no", "flag": "maybe"}


def build_result_row(result: PipelineResult, now: datetime | None = None) -> dict:
    if now is None:
        now = datetime.now()
    return {
        "filename": str(result.path),
        "QA_status": VERDICT_TO_QA_STATUS[result.verdict],
        "reason": "; ".join(result.precheck_findings) if not result.precheck_passed else "",
        "user": "fly",
        "date": now.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": result.duration_sec,
        "verdict": result.verdict,
        "defect_score": result.defect_score if result.defect_score is not None else "",
        "confidence_signal": result.confidence_signal
        if result.confidence_signal is not None
        else "",
        "precheck_passed": result.precheck_passed,
        "precheck_findings": "; ".join(result.precheck_findings),
        "fed_to_connectome": result.fed_to_connectome,
        "calibrated": result.calibrated,
    }


def resolve_thresholds(
    loaded: DecoderThresholds,
    confidence_fail_max: float | None,
    confidence_flag_max: float | None,
) -> DecoderThresholds:
    """Apply CLI --confidence-fail-max/--confidence-flag-max overrides onto thresholds
    loaded from a file, keeping flag_max >= fail_max so both tiers stay reachable.

    decode() checks fail_max first: if flag_max carried over from the loaded file were
    left below a newly-overridden (higher) fail_max, the flag tier would be dead code --
    any confidence low enough to hit it would already have hit fail_max first.
    """
    if confidence_fail_max is None and confidence_flag_max is None:
        return loaded

    new_fail_max = (
        confidence_fail_max
        if confidence_fail_max is not None
        else loaded.confidence_fail_max
    )
    new_flag_max = (
        confidence_flag_max
        if confidence_flag_max is not None
        else loaded.confidence_flag_max
    )
    new_flag_max = max(new_flag_max, new_fail_max)

    return DecoderThresholds(
        confidence_fail_max=new_fail_max,
        confidence_flag_max=new_flag_max,
        calibrated=True,  # explicitly set by the user -- not an uncalibrated placeholder
    )


def find_pngs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.png") if p.is_file())


def build_pipeline_config(neurons, grid_size: tuple[int, int]) -> PipelineConfig:
    r1_r6_ids = body_ids_by_type(neurons, "R1-R6")
    r8_ids = [bid for t in R8_TYPES for bid in body_ids_by_type(neurons, t)]
    dnp20_ids = body_ids_by_type(neurons, "DNp20")
    dnpe017_ids = body_ids_by_type(neurons, "DNpe017")

    if len(dnp20_ids) != 2:
        raise RuntimeError(
            f"expected exactly 2 DNp20 neurons (L/R), found {len(dnp20_ids)}"
        )

    neurons_indexed = (
        neurons if neurons.index.name == "bodyId" else neurons.set_index("bodyId")
    )
    instances = (
        neurons_indexed.loc[dnp20_ids, "instance"]
        if "instance" in neurons_indexed.columns
        else None
    )
    if instances is not None:
        left_id = next(
            bid for bid in dnp20_ids if str(instances.get(bid, "")).endswith("_L")
        )
        right_id = next(
            bid for bid in dnp20_ids if str(instances.get(bid, "")).endswith("_R")
        )
    else:
        left_id, right_id = sorted(dnp20_ids)

    return PipelineConfig(
        r1_r6_ids=r1_r6_ids,
        r8_ids=r8_ids,
        dnp20_left_id=left_id,
        dnp20_right_id=right_id,
        dnpe017_ids=dnpe017_ids,
        grid_size=grid_size,
    )


def build_graph_and_simulator(
    export_dir: Path, leak: float, steps: int
) -> tuple[ConnectomeGraph, ConnectomeSimulator]:
    neurons, roi_conn = load_connectome_export(export_dir)
    graph = build_connectome_graph(neurons, roi_conn)
    simulator = ConnectomeSimulator(
        graph.weight_matrix, graph.id_to_index, leak=leak, steps=steps
    )
    return graph, simulator


def build_simulator(export_dir: Path, leak: float, steps: int):
    """Back-compat wrapper for callers (e.g. scripts/run_validation.py) that only need
    the simulator + neuron table, not the full graph (needed for the dashboard's viz subset)."""
    graph, simulator = build_graph_and_simulator(export_dir, leak, steps)
    return simulator, graph.neurons


def run(
    root: Path,
    export_dir: Path,
    output_csv: Path,
    leak: float,
    steps: int,
    grid_size: tuple[int, int],
    launch_dashboard: bool,
    port: int | None,
    thresholds_path: Path | None = None,
    confidence_fail_max: float | None = None,
    confidence_flag_max: float | None = None,
) -> int:
    print(BANNER)

    pngs = find_pngs(root)
    if not pngs:
        print(f"No PNG files found under {root}")
        return 0

    print(f"Loading connectome from {export_dir} (leak={leak}, steps={steps})...")
    t0 = time.perf_counter()
    graph, simulator = build_graph_and_simulator(export_dir, leak, steps)
    config = build_pipeline_config(graph.neurons, grid_size)
    print(
        f"  loaded in {time.perf_counter() - t0:.1f}s "
        f"({len(config.r1_r6_ids)} R1-R6, {len(config.r8_ids)} R8-family, "
        f"DNp20 L={config.dnp20_left_id}/R={config.dnp20_right_id}, {len(config.dnpe017_ids)} DNpe017)"
    )

    loaded_thresholds = load_thresholds(thresholds_path)
    thresholds = resolve_thresholds(
        loaded_thresholds, confidence_fail_max, confidence_flag_max
    )
    if confidence_fail_max is not None or confidence_flag_max is not None:
        print(
            f"Using manually-overridden thresholds: fail<={thresholds.confidence_fail_max:.6g}, "
            f"flag<={thresholds.confidence_flag_max:.6g}"
        )
    elif thresholds_path is not None:
        print(
            f"Loaded thresholds from {thresholds_path}: fail<={thresholds.confidence_fail_max:.6g}, "
            f"flag<={thresholds.confidence_flag_max:.6g}"
        )

    if not thresholds.calibrated:
        print(
            "WARNING: decoder thresholds are UNCALIBRATED placeholders. "
            "Run scripts/run_validation.py before trusting any verdict."
        )

    event_bus = EventBus()
    dashboard = None
    capture_body_ids: list[int] | None = None

    if launch_dashboard:
        from fly_qa.webapp.server import DashboardServer

        viz_subset = pick_visualization_subset(
            graph,
            config.r1_r6_ids,
            config.r8_ids,
            [config.dnp20_left_id, config.dnp20_right_id],
            config.dnpe017_ids,
        )
        capture_body_ids = [n.body_id for n in viz_subset.nodes]
        dashboard_root = root if root.is_dir() else root.parent
        dashboard = DashboardServer(
            event_bus, root=dashboard_root, viz_subset=viz_subset, port=port
        )
        dashboard.start(open_browser=True)
        print(
            f"Dashboard: {dashboard.url} ({len(viz_subset.nodes)} real neurons, {len(viz_subset.edges)} real edges)"
        )
        time.sleep(0.3)

    print(
        f"Processing {len(pngs)} images (connectome simulation is expensive per image)..."
    )
    event_bus.publish(ResultEvent(type="started", total_images=len(pngs)))

    tally = {"pass": 0, "fail": 0, "flag": 0}
    rows = []
    for i, path in enumerate(pngs, 1):
        result = run_pipeline(
            path, simulator, config, thresholds, capture_body_ids=capture_body_ids
        )
        tally[result.verdict] += 1
        rows.append(build_result_row(result))
        print(
            f"  [{i}/{len(pngs)}] {path.name}: {result.verdict} "
            f"(precheck={'pass' if result.precheck_passed else 'FAIL'}, {result.duration_sec}s)"
        )

        event_bus.publish(
            ResultEvent(
                type="result",
                path=str(result.path.resolve()),
                precheck_passed=result.precheck_passed,
                precheck_findings=result.precheck_findings,
                fed_to_connectome=result.fed_to_connectome,
                defect_score=result.defect_score,
                confidence_signal=result.confidence_signal,
                verdict=result.verdict,
                calibrated=result.calibrated,
                total_images=len(pngs),
                processed_images=i,
                tally=dict(tally),
                step_activity=result.step_activity.tolist()
                if result.step_activity is not None
                else None,
            )
        )

    event_bus.publish(ResultEvent(type="finished", tally=dict(tally)))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n{tally['pass']} pass, {tally['fail']} fail, {tally['flag']} flag -- wrote {output_csv}"
    )
    print(BANNER)

    if dashboard is not None:
        print("Dashboard is running...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        dashboard.stop()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fly-qa", description=__doc__)
    parser.add_argument(
        "path", type=Path, help="A PNG file or a directory to scan recursively"
    )
    parser.add_argument(
        "--connectome-export",
        type=Path,
        default=Path.home() / ".cache" / "fly_qa" / "connectome_export",
        help="Directory with the CSVs from scripts/fetch_connectome.py",
    )
    parser.add_argument("--output", type=Path, default=Path("fly_qa_results.csv"))
    parser.add_argument("--leak", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--grid-size", type=int, nargs=2, default=(256, 256), metavar=("W", "H")
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Don't launch the live web dashboard",
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Dashboard port (default: auto-pick)"
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="Path to a decoder_thresholds.json file (default: the shipped calibrated thresholds). "
        "e.g. .devtest/fruit_demo/decoder_thresholds.json for the fruit-calibrated ones.",
    )
    parser.add_argument(
        "--confidence-fail-max",
        type=float,
        default=None,
        help="Override: verdict is 'fail' when confidence_signal <= this value (bypasses --thresholds)",
    )
    parser.add_argument(
        "--confidence-flag-max",
        type=float,
        default=None,
        help="Override: verdict is 'flag' when confidence_signal <= this value (bypasses --thresholds)",
    )
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 1
    if not (args.connectome_export / "neurons_full.csv").exists():
        print(
            f"error: no connectome export found at {args.connectome_export}. "
            f"Run: python scripts/fetch_connectome.py --output-dir {args.connectome_export}",
            file=sys.stderr,
        )
        return 1

    return run(
        args.path,
        args.connectome_export,
        args.output,
        args.leak,
        args.steps,
        tuple(args.grid_size),
        not args.no_dashboard,
        args.port,
        thresholds_path=args.thresholds,
        confidence_fail_max=args.confidence_fail_max,
        confidence_flag_max=args.confidence_flag_max,
    )


if __name__ == "__main__":
    raise SystemExit(main())
