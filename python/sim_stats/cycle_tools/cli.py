# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CLI wiring for cycle estimation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .hardware_profile import DEFAULT, resolve_profile
from .model import estimate_kernel_cycles, group_kernel_estimates
from .parse import extract_kernel_features, extract_kernel_work, parse_trace
from .report import print_peak_report, print_report, write_json_report
from .types import EstimatorConfig, KernelEstimate, KernelWork


def build_estimation_pipeline(
    trace_path: Path,
    config: EstimatorConfig,
    include_zero_kernels: bool = False,
) -> list[KernelEstimate]:
    events = parse_trace(trace_path)
    features = extract_kernel_features(events)
    return estimate_kernel_cycles(features, config, include_zero_kernels)


def build_peak_pipeline(trace_path: Path) -> list[KernelWork]:
    """v1.0 analytical peak pipeline: trace -> per-kernel work records."""
    events = parse_trace(trace_path)
    return list(extract_kernel_work(events).values())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tt-lang-sim-cycles",
        description=(
            "Estimate kernel cycles from a tt-lang simulator trace using a "
            "higher-level roofline-style model."
        ),
        epilog=(
            "Advanced model-tuning flags (roofline peaks, per-event costs, "
            "blocking scales) are hidden from this help; see "
            "docs/development/CycleEstimator.md."
        ),
    )
    parser.add_argument(
        "trace",
        metavar="FILE",
        help="JSON Lines trace file produced by tt-lang-sim --trace",
    )
    parser.add_argument(
        "--model",
        choices=("phase", "peak"),
        default="phase",
        help=(
            "Estimation model: 'phase' (v0.1, default) or 'peak' "
            "(v1.0 analytical ideal-peak, hardware-cycle oriented)"
        ),
    )
    parser.add_argument(
        "--hw-profile",
        default=DEFAULT.name,
        help=(
            "Hardware profile for the peak model: a built-in name or a path to a "
            ".json profile file (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write a JSON report",
    )
    parser.add_argument(
        "--include-zero-kernels",
        action="store_true",
        help="Include kernels with both measured and estimated cycles equal to zero",
    )

    # Advanced model-tuning knobs. Hidden from --help (help=SUPPRESS) so the
    # common path matches the sibling tt-lang-sim-stats tool; still accepted on
    # the command line and documented in docs/development/CycleEstimator.md.
    tuning = parser.add_argument_group("advanced tuning")
    tuning.add_argument(
        "--flops-per-tile",
        type=float,
        default=EstimatorConfig.flops_per_tile,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--bytes-per-tile",
        type=float,
        default=EstimatorConfig.bytes_per_tile,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--peak-flops-per-cycle",
        type=float,
        default=EstimatorConfig.peak_flops_per_cycle,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--memory-bytes-per-cycle",
        type=float,
        default=EstimatorConfig.memory_bytes_per_cycle,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--wait-event-cycles",
        type=float,
        default=EstimatorConfig.wait_event_cycles,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--reserve-event-cycles",
        type=float,
        default=EstimatorConfig.reserve_event_cycles,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--sync-event-cycles",
        type=float,
        default=EstimatorConfig.sync_event_cycles,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--copy-call-cycles",
        type=float,
        default=EstimatorConfig.copy_call_cycles,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--blocked-cycle-weight",
        type=float,
        default=EstimatorConfig.blocked_cycle_weight,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--kernel-launch-cycles",
        type=float,
        default=EstimatorConfig.kernel_launch_cycles,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--dfb-wait-block-scale",
        type=float,
        default=EstimatorConfig.dfb_wait_block_scale,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--dfb-reserve-block-scale",
        type=float,
        default=EstimatorConfig.dfb_reserve_block_scale,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--copy-duration-scale",
        type=float,
        default=EstimatorConfig.copy_duration_scale,
        help=argparse.SUPPRESS,
    )
    tuning.add_argument(
        "--mismatch-threshold-pct",
        type=float,
        default=EstimatorConfig.mismatch_threshold_pct,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    trace_path = Path(args.trace).resolve()
    if not trace_path.exists():
        print(f"Error: trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    if args.model == "peak":
        try:
            hw = resolve_profile(args.hw_profile)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        if args.json_out is not None:
            print(
                "note: --json-out is not yet supported for the peak model; "
                "printing report only.",
                file=sys.stderr,
            )
        kernels = build_peak_pipeline(trace_path)
        print_peak_report(kernels, hw)
        return

    config = EstimatorConfig(
        flops_per_tile=args.flops_per_tile,
        bytes_per_tile=args.bytes_per_tile,
        peak_flops_per_cycle=args.peak_flops_per_cycle,
        memory_bytes_per_cycle=args.memory_bytes_per_cycle,
        wait_event_cycles=args.wait_event_cycles,
        reserve_event_cycles=args.reserve_event_cycles,
        sync_event_cycles=args.sync_event_cycles,
        copy_call_cycles=args.copy_call_cycles,
        blocked_cycle_weight=args.blocked_cycle_weight,
        kernel_launch_cycles=args.kernel_launch_cycles,
        dfb_wait_block_scale=args.dfb_wait_block_scale,
        dfb_reserve_block_scale=args.dfb_reserve_block_scale,
        copy_duration_scale=args.copy_duration_scale,
        mismatch_threshold_pct=args.mismatch_threshold_pct,
    )
    rows = build_estimation_pipeline(
        trace_path,
        config,
        include_zero_kernels=args.include_zero_kernels,
    )
    print_report(rows, config.mismatch_threshold_pct)

    if args.json_out is not None:
        out_path = args.json_out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        groups = group_kernel_estimates(rows)
        write_json_report(out_path, rows, groups, config)
        print(f"\nWrote JSON report: {out_path}")
