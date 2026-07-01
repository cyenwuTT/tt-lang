# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Reporting and diagnostics for cycle estimation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from .model import group_kernel_estimates
from .types import (
    EstimatorConfig,
    KernelEstimate,
    KernelGroupEstimate,
    PeakKernel,
    PeakResult,
)
from ..utils import format_float, node_sort_key


# WAPE formula (weighted by measured cycles).
def _weighted_abs_error_pct(pairs: list[tuple[float, int]]) -> float:
    numerator = 0.0
    denominator = 0

    for estimate, measured in pairs:
        if measured <= 0:
            continue
        numerator += (
            abs((estimate - float(measured)) / float(measured) * 100.0) * measured
        )
        denominator += measured

    if denominator == 0:
        return math.nan

    return numerator / denominator


# Ablation: measure how much the blocked-cycle term contributes.
def ablation_metrics(rows: list[KernelEstimate]) -> dict[str, float]:
    full_pairs: list[tuple[float, int]] = []
    no_blocked_pairs: list[tuple[float, int]] = []
    blocked_only_pairs: list[tuple[float, int]] = []
    blocked_share_num = 0.0
    blocked_share_den = 0.0

    for row in rows:
        full_pairs.append((row.estimated_cycles, row.measured_cycles))
        no_blocked_est = max(0.0, row.estimated_cycles - row.blocked_cycles_term)
        no_blocked_pairs.append((no_blocked_est, row.measured_cycles))
        blocked_only_pairs.append((row.blocked_cycles_term, row.measured_cycles))
        blocked_share_num += row.blocked_cycles_term
        blocked_share_den += max(row.estimated_cycles, 0.0)

    blocked_share_pct = (
        blocked_share_num / blocked_share_den * 100.0
        if blocked_share_den > 0.0
        else math.nan
    )

    return {
        "full_wape_pct": _weighted_abs_error_pct(full_pairs),
        "no_blocked_wape_pct": _weighted_abs_error_pct(no_blocked_pairs),
        "blocked_only_wape_pct": _weighted_abs_error_pct(blocked_only_pairs),
        "blocked_share_pct": blocked_share_pct,
    }


def feature_provenance() -> dict[str, str]:
    return {
        "measured_cycles": "ground-truth(trace interval from kernel_start/kernel_end)",
        "blocked_cycles": "trace-derived(kernel_block/kernel_unblock spans)",
        "active_cycles": "derived(measured_cycles - blocked_cycles)",
        "wait_count/reserve_count/push_count/pop_count": "trace-derived(event counters)",
        "wait_tiles/reserve_tiles": "trace-derived(tile metadata from wait/reserve end)",
        "local_l1_tiles/remote_l1_tiles/dram_tiles/copy_tiles": "trace-derived(copy_end metadata)",
        "flops_per_tile/bytes_per_tile/peaks/event costs": "config-derived",
        "estimated_cycles": "model-derived from trace + config",
    }


# Calibration scale suggestions per role.
def role_calibration_suggestions(rows: list[KernelEstimate]) -> dict[str, float]:
    measured_sum: dict[str, float] = {
        "compute": 0.0,
        "read": 0.0,
        "write": 0.0,
        "other": 0.0,
    }

    base_sum: dict[str, float] = {
        "compute": 0.0,
        "read": 0.0,
        "write": 0.0,
        "other": 0.0,
    }

    for row in rows:
        role = row.role if row.role in measured_sum else "other"
        non_blocked_base = max(0.0, row.estimated_cycles - row.blocked_cycles_term)
        measured_sum[role] += float(max(row.measured_cycles, 0))
        base_sum[role] += non_blocked_base

    out: dict[str, float] = {}

    for role in ("compute", "read", "write", "other"):
        if base_sum[role] > 0.0:
            out[role] = measured_sum[role] / base_sum[role]
        else:
            out[role] = math.nan

    return out


def print_report(rows: list[KernelEstimate], threshold_pct: float) -> None:
    if not rows:
        print("No kernel-level rows were produced from the trace.")
        return

    header = (
        f"{'Kernel':<28} {'Role':<8} {'Measured':>10} {'Estimated':>10} "
        f"{'Err%':>8} {'Eff':>7} {'M-Eff':>7} {'OI':>8} {'Bound':<14}"
    )

    width = len(header)

    print("\n" + "=" * width)
    print("TT-Lang Trace Cycle Estimation (v0.1 roofline model)")
    print("=" * width)
    print(header)
    print("-" * width)

    refine_count = 0
    above_threshold_count = 0
    full_pairs: list[tuple[float, int]] = []

    for row in rows:
        full_pairs.append((row.estimated_cycles, row.measured_cycles))
        if row.needs_lower_level_model:
            refine_count += 1
        if row.abs_error_pct > threshold_pct:
            above_threshold_count += 1

        print(
            f"{row.kernel:<28} {row.role:<8} {row.measured_cycles:>10d} "
            f"{format_float(row.estimated_cycles):>10} "
            f"{format_float(row.signed_error_pct):>8} "
            f"{format_float(row.roofline_efficiency):>7} "
            f"{format_float(row.measured_roofline_efficiency):>7} "
            f"{format_float(row.operational_intensity):>8} "
            f"{row.bound_classification:<14}"
        )

    groups = group_kernel_estimates(rows)
    wape = _weighted_abs_error_pct(full_pairs)

    print("-" * width)

    if math.isfinite(wape):
        print(f"Weighted abs error %: {format_float(wape)}")
    else:
        print("Weighted abs error %: n/a")

    print(
        "Kernels above mismatch threshold "
        f"({format_float(threshold_pct)}%): {above_threshold_count}/{len(rows)}"
    )

    print(f"Kernels that need lower-level model: {refine_count}/{len(rows)}")

    mismatch_rows = [r for r in rows if r.abs_error_pct > threshold_pct]
    mismatch_rows.sort(key=lambda r: r.abs_error_pct, reverse=True)

    print("\nMismatch notes (top 15 by abs error):")

    for row in mismatch_rows[:15]:
        print(f"- {row.kernel}: {row.mismatch_reason}")
    if len(mismatch_rows) > 15:
        print(f"... {len(mismatch_rows) - 15} additional kernels exceed threshold")

    group_header = (
        f"{'Node':<28} {'Kernels':>7} {'Measured':>10} {'Estimated':>10} {'Err%':>8}"
    )
    group_width = len(group_header)

    print("\n" + "=" * group_width)
    print("Kernel-Level Group Totals (heuristic critical-path v0.1)")
    print("=" * group_width)
    print(group_header)
    print("-" * group_width)

    for group in groups:
        print(
            f"{group.node:<28} {group.kernel_count:>7d} "
            f"{group.measured_cycles:>10d} {format_float(group.estimated_cycles):>10} "
            f"{format_float(group.signed_error_pct):>8}"
        )

    print("-" * group_width)

    ablation = ablation_metrics(rows)

    print("\nAblation Diagnostics (internal)")
    print(f"- Full model WAPE%: {format_float(ablation['full_wape_pct'])}")
    print(f"- No-blocked-term WAPE%: {format_float(ablation['no_blocked_wape_pct'])}")
    print(f"- Blocked-only WAPE%: {format_float(ablation['blocked_only_wape_pct'])}")
    print(
        "- Blocked term share of estimated cycles %: "
        f"{format_float(ablation['blocked_share_pct'])}"
    )

    provenance = feature_provenance()

    print("\nFeature Provenance Audit (internal)")
    for key in sorted(provenance):
        print(f"- {key}: {provenance[key]}")

    role_scales = role_calibration_suggestions(rows)

    print("\nCalibration Suggestions (non-blocked base -> measured)")
    print("- Use profiler/perf-summary wall-time as ground truth when available.")
    for role in ("compute", "read", "write", "other"):
        print(f"- Suggested {role} scale: {format_float(role_scales[role], digits=3)}")


def write_json_report(
    path: Path,
    rows: list[KernelEstimate],
    groups: list[KernelGroupEstimate],
    config: EstimatorConfig,
) -> None:
    payload = {
        "config": asdict(config),
        "kernels": [asdict(r) for r in rows],
        "kernel_groups": [asdict(g) for g in groups],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# v1.0 analytical peak model — render / serialize (all pure over PeakResult)
# ---------------------------------------------------------------------------

_PEAK_TOOL = "tt-lang-sim-cycles"
_PEAK_SCHEMA_VERSION = 1
_PEAK_W = 78


def _peak_header(result: PeakResult, unit: str) -> None:
    print("\n" + "=" * _PEAK_W)
    print(f"Cycle Estimate — ideal-peak model (profile: {result.profile_name})")
    print("=" * _PEAK_W)
    print(f"{unit:<28} {'Compute':>12} {'Movement':>12} {'Cycles':>12}  Bound")
    print("-" * _PEAK_W)


def _peak_footer(result: PeakResult) -> None:
    print("-" * _PEAK_W)
    print(
        f"{'Program (throughput-bound)':<28} {'':>12} {'':>12} "
        f"{result.program_cycles:>12.2f}"
    )
    idle = result.total_nodes - result.active_nodes
    print(f"Active nodes: {result.active_nodes} / {result.total_nodes}   ({idle} idle)")
    print("=" * _PEAK_W)
    if sum(k.compute_cycles for k in result.kernels) == 0.0:
        print(
            "note: compute path is 0 — no compute_op events in this trace "
            "(sim instrumentation pending); movement-only estimate."
        )


def print_peak_report(result: PeakResult) -> None:
    """Detailed per-kernel view of the peak result — complete, includes zero rows."""
    _peak_header(result, "Kernel")
    for pk in result.kernels:
        print(
            f"{pk.kernel:<28} {pk.compute_cycles:>12.2f} {pk.movement_cycles:>12.2f} "
            f"{pk.cycles:>12.2f}  {pk.bound}"
        )
    _peak_footer(result)


def print_peak_summary(result: PeakResult, include_zero: bool = False) -> None:
    """Clean per-node rollup of the peak result (the default view).

    Each node's columns are the max over its kernels (concurrent RISCs), matching
    the program combiner.
    """
    per_node: dict[str, list[float]] = {}
    for pk in result.kernels:
        entry = per_node.setdefault(pk.node, [0.0, 0.0, 0.0])
        entry[0] = max(entry[0], pk.compute_cycles)
        entry[1] = max(entry[1], pk.movement_cycles)
        entry[2] = max(entry[2], pk.cycles)

    _peak_header(result, "Node")
    for node in sorted(per_node, key=node_sort_key):
        compute, movement, cyc = per_node[node]
        if not include_zero and cyc == 0.0:
            continue
        bound = "compute-bound" if compute > movement else "memory-bound"
        print(f"{node:<28} {compute:>12.2f} {movement:>12.2f} {cyc:>12.2f}  {bound}")
    _peak_footer(result)


def write_peak_json_report(path: Path, result: PeakResult) -> None:
    """Serialize the full, self-describing peak result (for analysis reuse)."""
    payload = {
        "tool": _PEAK_TOOL,
        "schema_version": _PEAK_SCHEMA_VERSION,
        "model": "peak-ideal",
        "profile": result.profile,
        "program_cycles": result.program_cycles,
        "total_nodes": result.total_nodes,
        "active_nodes": result.active_nodes,
        "kernels": [asdict(k) for k in result.kernels],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_peak_result(path: Path | str) -> PeakResult:
    """Load a saved peak report JSON back into a PeakResult, with validation.

    Raises FileNotFoundError if the file is missing, or ValueError if it is not a
    tt-lang-sim-cycles report (including the common mistake of passing a raw
    JSON-Lines trace instead of a saved report).
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"report file not found: {p}") from None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(
            f"{p} is not a cycle report: not a single JSON object "
            "(a raw --trace file is JSON Lines, not a report)"
        ) from None

    if (
        not isinstance(data, dict)
        or data.get("tool") != _PEAK_TOOL
        or "kernels" not in data
    ):
        raise ValueError(
            f"{p} is not a tt-lang-sim-cycles report (missing tool marker or kernels)"
        )

    try:
        kernels = [PeakKernel(**k) for k in data["kernels"]]
        profile = data.get("profile", {})
        return PeakResult(
            profile_name=profile.get("name", data.get("profile_name", "?")),
            profile=profile,
            program_cycles=float(data["program_cycles"]),
            total_nodes=int(data.get("total_nodes", 0)),
            active_nodes=int(data.get("active_nodes", 0)),
            kernels=kernels,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed cycle report {p}: {exc}") from None
