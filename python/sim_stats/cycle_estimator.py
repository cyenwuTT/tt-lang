# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Higher-level cycle estimation from tt-lang simulator JSONL traces.

This module provides a roofline-style v0 estimator designed for simulator traces,
not detailed microarchitectural modeling. It separates four stages:

1. Trace parsing
2. Kernel-level feature extraction
3. Higher-level cycle estimation with roofline metrics
4. Estimated-vs-measured mismatch analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    tick: int
    event: str
    kernel: str | None
    data: dict[str, Any]


@dataclass
class KernelFeatures:
    kernel: str
    role: str
    measured_cycles: int = 0
    blocked_cycles: int = 0
    active_cycles: int = 0
    wait_count: int = 0
    reserve_count: int = 0
    push_count: int = 0
    pop_count: int = 0
    wait_tiles: int = 0
    reserve_tiles: int = 0
    copy_calls: int = 0
    copy_tiles: int = 0
    local_l1_tiles: int = 0
    remote_l1_tiles: int = 0
    dram_tiles: int = 0


@dataclass(frozen=True)
class EstimatorConfig:
    flops_per_tile: float = 2048.0
    bytes_per_tile: float = 2048.0
    peak_flops_per_cycle: float = 4096.0
    memory_bytes_per_cycle: float = 1024.0
    wait_event_cycles: float = 2.0
    reserve_event_cycles: float = 2.0
    sync_event_cycles: float = 1.0
    mismatch_threshold_pct: float = 20.0


@dataclass
class KernelEstimate:
    kernel: str
    role: str
    measured_cycles: int
    estimated_cycles: float
    abs_error_pct: float
    signed_error_pct: float
    roofline_efficiency: float
    operational_intensity: float
    bound_classification: str
    compute_ceiling_cycles: float
    memory_ceiling_cycles: float
    stall_cycles: float
    sync_cycles: float
    mismatch_reason: str
    needs_lower_level_model: bool


@dataclass
class _OpenKernelState:
    start_tick: int
    block_start_tick: int | None = None


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_trace(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed line {lineno}: {exc}",
                    file=sys.stderr,
                )
                continue

            tick = _as_int(obj.get("tick", 0))
            event = str(obj.get("event", ""))
            kernel_obj = obj.get("kernel")
            kernel = str(kernel_obj) if kernel_obj is not None else None
            data = {
                k: v
                for k, v in obj.items()
                if k not in {"tick", "event", "kernel", "node"}
            }
            if "node" in obj:
                data["node"] = obj["node"]

            events.append(TraceEvent(tick=tick, event=event, kernel=kernel, data=data))
    return events


def _infer_kernel_role(kernel: str) -> str:
    if kernel.endswith("-compute"):
        return "compute"
    if kernel.endswith("-read"):
        return "read"
    if kernel.endswith("-write"):
        return "write"
    return "other"


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_kernel_features(events: list[TraceEvent]) -> dict[str, KernelFeatures]:
    features: dict[str, KernelFeatures] = {}
    # Track open kernel intervals to accumulate measured/blocked time.
    open_states: dict[str, list[_OpenKernelState]] = {}
    max_tick = 0

    def get_features(kernel: str) -> KernelFeatures:
        if kernel not in features:
            features[kernel] = KernelFeatures(
                kernel=kernel,
                role=_infer_kernel_role(kernel),
            )
        return features[kernel]

    for ev in events:
        max_tick = max(max_tick, ev.tick)
        kernel = ev.kernel

        if ev.event == "kernel_start" and kernel:
            get_features(kernel)
            open_states.setdefault(kernel, []).append(_OpenKernelState(start_tick=ev.tick))
            continue

        if ev.event == "kernel_end" and kernel:
            get_features(kernel)
            stack = open_states.get(kernel, [])
            if stack:
                st = stack.pop()
                if st.block_start_tick is not None:
                    features[kernel].blocked_cycles += max(0, ev.tick - st.block_start_tick)
                features[kernel].measured_cycles += max(0, ev.tick - st.start_tick)
            continue

        if not kernel:
            continue

        f = get_features(kernel)

        if ev.event == "kernel_block":
            stack = open_states.get(kernel, [])
            if stack and stack[-1].block_start_tick is None:
                stack[-1].block_start_tick = ev.tick
            continue

        if ev.event == "kernel_unblock":
            stack = open_states.get(kernel, [])
            if stack and stack[-1].block_start_tick is not None:
                f.blocked_cycles += max(0, ev.tick - stack[-1].block_start_tick)
                stack[-1].block_start_tick = None
            continue

        if ev.event == "dfb_wait_begin":
            f.wait_count += 1
            continue

        if ev.event == "dfb_wait_end":
            f.wait_tiles += _as_int(ev.data.get("tiles", 0))
            continue

        if ev.event == "dfb_reserve_begin":
            f.reserve_count += 1
            continue

        if ev.event == "dfb_reserve_end":
            f.reserve_tiles += _as_int(ev.data.get("tiles", 0))
            continue

        if ev.event == "dfb_push":
            f.push_count += 1
            continue

        if ev.event == "dfb_pop":
            f.pop_count += 1
            continue

        if ev.event == "copy_end":
            f.copy_calls += 1
            tiles = _as_int(ev.data.get("tiles", 0))
            local_l1 = _as_int(ev.data.get("local_l1", 0))
            remote_l1 = _as_int(ev.data.get("remote_l1", 0))
            dram = _as_int(ev.data.get("dram", 0))
            f.local_l1_tiles += local_l1
            f.remote_l1_tiles += remote_l1
            f.dram_tiles += dram
            f.copy_tiles += tiles

    for kernel, stack in open_states.items():
        f = get_features(kernel)
        for st in stack:
            # If the trace ended before kernel_end, close intervals at max_tick.
            if st.block_start_tick is not None:
                f.blocked_cycles += max(0, max_tick - st.block_start_tick)
            f.measured_cycles += max(0, max_tick - st.start_tick)

    for f in features.values():
        f.active_cycles = max(0, f.measured_cycles - f.blocked_cycles)

    return features


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def _bound_classification(compute_ceiling: float, memory_ceiling: float) -> str:
    if compute_ceiling > memory_ceiling:
        return "compute-bound"
    if memory_ceiling > compute_ceiling:
        return "memory-bound"
    return "balanced"


def _mismatch_reason(
    feature: KernelFeatures,
    abs_error_pct: float,
    threshold_pct: float,
    compute_ceiling_cycles: float,
    memory_ceiling_cycles: float,
) -> str:
    if abs_error_pct <= threshold_pct:
        return "within-threshold"
    if feature.measured_cycles > 0:
        blocked_fraction = feature.blocked_cycles / feature.measured_cycles
        if blocked_fraction >= 0.30:
            return "stall-dominated: refine wait/sync model before Tensix-level"
    if feature.role == "other":
        return "unknown-kernel-role: add semantic tagging"
    if compute_ceiling_cycles == 0.0 and memory_ceiling_cycles == 0.0:
        return "no work signal in trace: add op-level counters"
    return "roofline-parameter mismatch"


def estimate_kernel_cycles(
    features: dict[str, KernelFeatures],
    config: EstimatorConfig,
    include_zero_kernels: bool = False,
) -> list[KernelEstimate]:
    rows: list[KernelEstimate] = []

    for name in sorted(features):
        f = features[name]
        # v0 proxy: compute kernels use wait_tiles as the only compute work signal.
        compute_tiles = f.wait_tiles if f.role == "compute" else 0
        # Prefer locality-aware tile counters; fall back to aggregate copy tiles.
        effective_memory_tiles = (
            f.local_l1_tiles + f.remote_l1_tiles + f.dram_tiles
            if (f.local_l1_tiles + f.remote_l1_tiles + f.dram_tiles) > 0
            else f.copy_tiles
        )

        flops = float(compute_tiles) * config.flops_per_tile
        bytes_moved = float(effective_memory_tiles) * config.bytes_per_tile

        compute_ceiling_cycles = (
            flops / config.peak_flops_per_cycle
            if config.peak_flops_per_cycle > 0.0
            else 0.0
        )
        memory_ceiling_cycles = (
            bytes_moved / config.memory_bytes_per_cycle
            if config.memory_bytes_per_cycle > 0.0
            else 0.0
        )

        roofline_base_cycles = max(compute_ceiling_cycles, memory_ceiling_cycles)
        stall_cycles = (
            f.wait_count * config.wait_event_cycles
            + f.reserve_count * config.reserve_event_cycles
        )
        sync_cycles = (f.push_count + f.pop_count) * config.sync_event_cycles
        estimated_cycles = roofline_base_cycles + stall_cycles + sync_cycles

        if f.measured_cycles > 0:
            signed_error_pct = (
                (estimated_cycles - float(f.measured_cycles))
                / float(f.measured_cycles)
                * 100.0
            )
            abs_error_pct = abs(signed_error_pct)
        else:
            signed_error_pct = math.inf if estimated_cycles > 0.0 else 0.0
            abs_error_pct = math.inf if estimated_cycles > 0.0 else 0.0

        if bytes_moved > 0.0:
            operational_intensity = flops / bytes_moved
        elif flops > 0.0:
            operational_intensity = math.inf
        else:
            operational_intensity = 0.0

        lower_bound = roofline_base_cycles
        roofline_efficiency = (
            min(1.0, lower_bound / estimated_cycles)
            if estimated_cycles > 0.0
            else 1.0
        )
        bound_class = _bound_classification(compute_ceiling_cycles, memory_ceiling_cycles)

        mismatch_reason = _mismatch_reason(
            f,
            abs_error_pct,
            config.mismatch_threshold_pct,
            compute_ceiling_cycles,
            memory_ceiling_cycles,
        )
        # Escalate only true model mismatches, not known trace/semantic gaps.
        needs_lower_level_model = (
            abs_error_pct > config.mismatch_threshold_pct
            and mismatch_reason == "roofline-parameter mismatch"
        )

        rows.append(
            KernelEstimate(
                kernel=f.kernel,
                role=f.role,
                measured_cycles=f.measured_cycles,
                estimated_cycles=estimated_cycles,
                abs_error_pct=abs_error_pct,
                signed_error_pct=signed_error_pct,
                roofline_efficiency=roofline_efficiency,
                operational_intensity=operational_intensity,
                bound_classification=bound_class,
                compute_ceiling_cycles=compute_ceiling_cycles,
                memory_ceiling_cycles=memory_ceiling_cycles,
                stall_cycles=stall_cycles,
                sync_cycles=sync_cycles,
                mismatch_reason=mismatch_reason,
                needs_lower_level_model=needs_lower_level_model,
            )
        )

    if include_zero_kernels:
        return rows
    return [r for r in rows if not (r.measured_cycles == 0 and r.estimated_cycles == 0.0)]


def _format_float(value: float, digits: int = 2) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_report(rows: list[KernelEstimate], threshold_pct: float) -> None:
    if not rows:
        print("No kernel-level rows were produced from the trace.")
        return

    header = (
        f"{'Kernel':<28} {'Role':<8} {'Measured':>10} {'Estimated':>10} "
        f"{'Err%':>8} {'Eff':>7} {'OI':>8} {'Bound':<14}"
    )
    width = len(header)
    print("\n" + "=" * width)
    print("TT-Lang Trace Cycle Estimation (v0 roofline model)")
    print("=" * width)
    print(header)
    print("-" * width)

    weighted_abs_error_num = 0.0
    weighted_abs_error_den = 0
    refine_count = 0
    above_threshold_count = 0

    for row in rows:
        if row.needs_lower_level_model:
            refine_count += 1
        if row.abs_error_pct > threshold_pct:
            above_threshold_count += 1
        if math.isfinite(row.abs_error_pct):
            weighted_abs_error_num += row.abs_error_pct * row.measured_cycles
            weighted_abs_error_den += row.measured_cycles

        print(
            f"{row.kernel:<28} {row.role:<8} {row.measured_cycles:>10d} "
            f"{_format_float(row.estimated_cycles):>10} "
            f"{_format_float(row.signed_error_pct):>8} "
            f"{_format_float(row.roofline_efficiency):>7} "
            f"{_format_float(row.operational_intensity):>8} "
            f"{row.bound_classification:<14}"
        )

    print("-" * width)
    if weighted_abs_error_den > 0:
        wape = weighted_abs_error_num / weighted_abs_error_den
        print(f"Weighted abs error %: {_format_float(wape)}")
    else:
        print("Weighted abs error %: n/a")
    print(
        "Kernels above mismatch threshold "
        f"({_format_float(threshold_pct)}%): {above_threshold_count}/{len(rows)}"
    )
    print(f"Kernels that need lower-level model: {refine_count}/{len(rows)}")
    mismatch_rows = [r for r in rows if r.abs_error_pct > threshold_pct]
    mismatch_rows.sort(key=lambda r: r.abs_error_pct, reverse=True)
    print("\nMismatch notes (top 15 by abs error):")
    for row in mismatch_rows[:15]:
        print(f"- {row.kernel}: {row.mismatch_reason}")
    if len(mismatch_rows) > 15:
        print(f"... {len(mismatch_rows) - 15} additional kernels exceed threshold")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _write_json_report(path: Path, rows: list[KernelEstimate], config: EstimatorConfig) -> None:
    payload = {
        "config": asdict(config),
        "kernels": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_estimation_pipeline(
    trace_path: Path,
    config: EstimatorConfig,
    include_zero_kernels: bool = False,
) -> list[KernelEstimate]:
    events = parse_trace(trace_path)
    features = extract_kernel_features(events)
    return estimate_kernel_cycles(features, config, include_zero_kernels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        _write_json_report(out_path, rows, config)
        print(f"\nWrote JSON report: {out_path}")


if __name__ == "__main__":
    main()