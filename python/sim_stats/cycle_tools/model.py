# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Core cycle estimation and aggregation logic."""

from __future__ import annotations

import math

from .schedule import kernel_paths, program_cycles
from .types import (
    EstimatorConfig,
    HardwareProfile,
    KernelEstimate,
    KernelFeatures,
    KernelGroupEstimate,
    KernelWork,
    PeakKernel,
    PeakResult,
)
from ..utils import node_from_kernel


def _role_from_kernel(kernel: str) -> str:
    """Derive read/compute/write/other from the kernel-name suffix."""
    node = node_from_kernel(kernel)
    suffix = kernel.removeprefix(f"{node}-") if node != kernel else ""
    return suffix if suffix in {"compute", "read", "write"} else "other"


def build_peak_result(kernels: list[KernelWork], hw: HardwareProfile) -> PeakResult:
    """Assemble the canonical v1.0 PeakResult from per-kernel work + a profile."""
    peak_kernels: list[PeakKernel] = []
    for kw in sorted(kernels, key=lambda k: k.kernel):
        compute, movement = kernel_paths(kw, hw)
        peak_kernels.append(
            PeakKernel(
                kernel=kw.kernel,
                node=node_from_kernel(kw.kernel),
                role=_role_from_kernel(kw.kernel),
                compute_cycles=compute,
                movement_cycles=movement,
                cycles=max(compute, movement),
                bound="compute-bound" if compute > movement else "memory-bound",
            )
        )

    node_cycles: dict[str, float] = {}
    for pk in peak_kernels:
        node_cycles[pk.node] = max(node_cycles.get(pk.node, 0.0), pk.cycles)

    profile = {
        "name": hw.name,
        "clock_ghz": hw.clock_ghz,
        "bytes_per_tile": hw.bytes_per_tile,
        "compute_rate_default": hw.compute_rate_default,
        "noc_bw": dict(hw.noc_bw),
        "noc_latency": dict(hw.noc_latency),
    }

    return PeakResult(
        profile_name=hw.name,
        profile=profile,
        program_cycles=program_cycles(kernels, hw),
        total_nodes=len(node_cycles),
        active_nodes=sum(1 for v in node_cycles.values() if v > 0.0),
        kernels=peak_kernels,
    )


def _error_pcts(estimated_cycles: float, measured_cycles: int) -> tuple[float, float]:
    if measured_cycles > 0:
        signed_error_pct = (
            (estimated_cycles - float(measured_cycles)) / float(measured_cycles) * 100.0
        )
        return signed_error_pct, abs(signed_error_pct)

    signed_error_pct = math.inf if estimated_cycles > 0.0 else 0.0
    abs_error_pct = math.inf if estimated_cycles > 0.0 else 0.0

    return signed_error_pct, abs_error_pct


def mismatch_reason(
    feature: KernelFeatures,
    abs_error_pct: float,
    threshold_pct: float,
    blocked_cycles_term: float,
    estimated_cycles: float,
    compute_ceiling_cycles: float,
    memory_ceiling_cycles: float,
) -> str:
    if abs_error_pct <= threshold_pct:
        return "within-threshold"

    if feature.measured_cycles > 0:
        blocked_fraction = feature.blocked_cycles / feature.measured_cycles
        if blocked_fraction >= 0.30:
            return "stall-dominated: refine wait/sync model before Tensix-level"

    if estimated_cycles > 0.0 and blocked_cycles_term / estimated_cycles >= 0.50:
        return "blocked-term-dominated: audit blocked_cycles signal before escalation"

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
    out: list[KernelEstimate] = []

    for name in sorted(features):
        f = features[name]
        compute_tiles = 0

        # Compute tile proxy: wait + reserve tiles for compute role only.
        if f.role == "compute":
            compute_tiles = f.wait_tiles + f.reserve_tiles

        # Prefer locality-aware tile counters; fall back to aggregate copy tiles.
        effective_memory_tiles = (
            f.local_l1_tiles + f.remote_l1_tiles + f.dram_tiles
            if (f.local_l1_tiles + f.remote_l1_tiles + f.dram_tiles) > 0
            else f.copy_tiles
        )

        flops = float(compute_tiles) * config.flops_per_tile
        bytes_moved = float(effective_memory_tiles) * config.bytes_per_tile

        # Roofline ceilings: compute and memory.
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
        copy_overhead_cycles = f.copy_calls * config.copy_call_cycles

        # Phase-duration primary terms.
        dfb_wait_contribution = f.dfb_wait_block_cycles * config.dfb_wait_block_scale
        dfb_reserve_contribution = (
            f.dfb_reserve_block_cycles * config.dfb_reserve_block_scale
        )
        copy_duration_contribution = f.copy_duration_cycles * config.copy_duration_scale

        # Blocked-cycle term: disabled by default (blocked_cycle_weight=0) to prevent leakage.
        blocked_cycles_term = float(f.blocked_cycles) * config.blocked_cycle_weight
        launch_cycles = config.kernel_launch_cycles

        # Final estimate formula.
        estimated_cycles = (
            dfb_wait_contribution
            + dfb_reserve_contribution
            + copy_duration_contribution
            + roofline_base_cycles
            + stall_cycles
            + sync_cycles
            + copy_overhead_cycles
            + blocked_cycles_term
            + launch_cycles
        )

        # Mismatch analysis
        signed_error_pct, abs_error_pct = _error_pcts(
            estimated_cycles,
            f.measured_cycles,
        )

        if bytes_moved > 0.0:
            operational_intensity = flops / bytes_moved
        elif flops > 0.0:
            operational_intensity = math.inf
        else:
            operational_intensity = 0.0

        lower_bound = roofline_base_cycles
        roofline_efficiency = (
            min(1.0, lower_bound / estimated_cycles) if estimated_cycles > 0.0 else 1.0
        )
        measured_roofline_efficiency = (
            min(1.0, lower_bound / float(f.measured_cycles))
            if f.measured_cycles > 0
            else math.nan
        )

        bound_class = (
            "compute-bound"
            if compute_ceiling_cycles > memory_ceiling_cycles
            else "memory-bound"
        )

        mismatch_note = mismatch_reason(
            f,
            abs_error_pct,
            config.mismatch_threshold_pct,
            blocked_cycles_term,
            estimated_cycles,
            compute_ceiling_cycles,
            memory_ceiling_cycles,
        )

        # Escalation gate.
        needs_lower_level_model = (
            abs_error_pct > config.mismatch_threshold_pct
            and mismatch_note == "roofline-parameter mismatch"
        )

        out.append(
            KernelEstimate(
                kernel=f.kernel,
                role=f.role,
                measured_cycles=f.measured_cycles,
                estimated_cycles=estimated_cycles,
                abs_error_pct=abs_error_pct,
                signed_error_pct=signed_error_pct,
                roofline_efficiency=roofline_efficiency,
                measured_roofline_efficiency=measured_roofline_efficiency,
                operational_intensity=operational_intensity,
                bound_classification=bound_class,
                roofline_base_cycles=roofline_base_cycles,
                compute_ceiling_cycles=compute_ceiling_cycles,
                memory_ceiling_cycles=memory_ceiling_cycles,
                stall_cycles=stall_cycles,
                sync_cycles=sync_cycles,
                copy_overhead_cycles=copy_overhead_cycles,
                blocked_cycles_term=blocked_cycles_term,
                launch_cycles=launch_cycles,
                dfb_wait_block_contribution=dfb_wait_contribution,
                dfb_reserve_block_contribution=dfb_reserve_contribution,
                copy_duration_contribution=copy_duration_contribution,
                mismatch_reason=mismatch_note,
                needs_lower_level_model=needs_lower_level_model,
            )
        )

    if not include_zero_kernels:
        out = [
            r for r in out if not (r.measured_cycles == 0 and r.estimated_cycles == 0.0)
        ]

    return out


def group_kernel_estimates(rows: list[KernelEstimate]) -> list[KernelGroupEstimate]:
    grouped: dict[str, list[KernelEstimate]] = {}
    for row in rows:
        node = node_from_kernel(row.kernel)
        grouped.setdefault(node, []).append(row)

    out: list[KernelGroupEstimate] = []
    for node in sorted(grouped):
        # Group critical-path aggregation.
        group_rows = grouped[node]
        measured_cycles = max((r.measured_cycles for r in group_rows), default=0)
        work_estimate = max(
            (max(0.0, r.estimated_cycles - r.sync_cycles) for r in group_rows),
            default=0.0,
        )
        coordination_sync = max((r.sync_cycles for r in group_rows), default=0.0)
        estimated_cycles = work_estimate + coordination_sync

        signed_error_pct, abs_error_pct = _error_pcts(
            estimated_cycles,
            measured_cycles,
        )

        out.append(
            KernelGroupEstimate(
                node=node,
                kernel_count=len(group_rows),
                measured_cycles=measured_cycles,
                estimated_cycles=estimated_cycles,
                abs_error_pct=abs_error_pct,
                signed_error_pct=signed_error_pct,
                aggregation_model="heuristic_critical_path_v0_1",
            )
        )

    return out
