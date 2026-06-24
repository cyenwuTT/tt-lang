# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CLI wiring for cycle estimation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model import estimate_kernel_cycles, group_kernel_estimates
from .parse import extract_kernel_features, parse_trace
from .report import print_report, write_json_report
from .types import EstimatorConfig, KernelEstimate


def build_estimation_pipeline(
    trace_path: Path,
    config: EstimatorConfig,
    include_zero_kernels: bool = False,
) -> list[KernelEstimate]:
    events = parse_trace(trace_path)
    features = extract_kernel_features(events)
    return estimate_kernel_cycles(features, config, include_zero_kernels)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tt-lang-sim-cycles",
        description=(
            "Estimate kernel cycles from a tt-lang simulator trace using a "
            "higher-level roofline-style model."
        ),
    )
    parser.add_argument(
        "trace",
        metavar="FILE",
        help="JSON Lines trace file produced by tt-lang-sim --trace",
    )
    parser.add_argument(
        "--flops-per-tile",
        type=float,
        default=EstimatorConfig.flops_per_tile,
        help="Model flops per compute tile (default: %(default)s)",
    )
    parser.add_argument(
        "--bytes-per-tile",
        type=float,
        default=EstimatorConfig.bytes_per_tile,
        help="Bytes transferred per tile (default: %(default)s)",
    )
    parser.add_argument(
        "--peak-flops-per-cycle",
        type=float,
        default=EstimatorConfig.peak_flops_per_cycle,
        help="Compute roofline peak in flops/cycle (default: %(default)s)",
    )
    parser.add_argument(
        "--memory-bytes-per-cycle",
        type=float,
        default=EstimatorConfig.memory_bytes_per_cycle,
        help="Memory roofline peak in bytes/cycle (default: %(default)s)",
    )
    parser.add_argument(
        "--wait-event-cycles",
        type=float,
        default=EstimatorConfig.wait_event_cycles,
        help="Cycle cost per dfb_wait_begin event (default: %(default)s)",
    )
    parser.add_argument(
        "--reserve-event-cycles",
        type=float,
        default=EstimatorConfig.reserve_event_cycles,
        help="Cycle cost per dfb_reserve_begin event (default: %(default)s)",
    )
    parser.add_argument(
        "--sync-event-cycles",
        type=float,
        default=EstimatorConfig.sync_event_cycles,
        help="Cycle cost per dfb_push/dfb_pop event (default: %(default)s)",
    )
    parser.add_argument(
        "--copy-call-cycles",
        type=float,
        default=EstimatorConfig.copy_call_cycles,
        help="Cycle cost per copy_end event (default: %(default)s)",
    )
    parser.add_argument(
        "--blocked-cycle-weight",
        type=float,
        default=EstimatorConfig.blocked_cycle_weight,
        help=(
            "Optional diagnostic weight applied to blocked trace spans; "
            "keep at 0 to avoid duration leakage (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--kernel-launch-cycles",
        type=float,
        default=EstimatorConfig.kernel_launch_cycles,
        help="Fixed launch/scheduler overhead per kernel row (default: %(default)s)",
    )
    parser.add_argument(
        "--dfb-wait-block-scale",
        type=float,
        default=EstimatorConfig.dfb_wait_block_scale,
        help="Scaling factor for dfb_wait blocking durations (default: %(default)s)",
    )
    parser.add_argument(
        "--dfb-reserve-block-scale",
        type=float,
        default=EstimatorConfig.dfb_reserve_block_scale,
        help="Scaling factor for dfb_reserve blocking durations (default: %(default)s)",
    )
    parser.add_argument(
        "--copy-duration-scale",
        type=float,
        default=EstimatorConfig.copy_duration_scale,
        help="Scaling factor for copy_start/copy_end durations (default: %(default)s)",
    )
    parser.add_argument(
        "--mismatch-threshold-pct",
        type=float,
        default=EstimatorConfig.mismatch_threshold_pct,
        help="Threshold for significant mismatch (default: %(default)s)",
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
    args = parser.parse_args()

    trace_path = Path(args.trace).resolve()
    if not trace_path.exists():
        print(f"Error: trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

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
        groups = group_kernel_estimates(rows)
        write_json_report(out_path, rows, groups, config)
        print(f"\nWrote JSON report: {out_path}")
