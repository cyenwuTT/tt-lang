# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Data types for cycle estimation from simulator traces."""

from __future__ import annotations

from dataclasses import dataclass
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
    # Phase-duration features (new): trace-derived event pairs.
    dfb_wait_block_cycles: int = 0
    dfb_reserve_block_cycles: int = 0
    copy_duration_cycles: int = 0
    node_index: int = 0


@dataclass(frozen=True)
class EstimatorConfig:
    flops_per_tile: float = 2048.0
    bytes_per_tile: float = 2048.0
    peak_flops_per_cycle: float = 4096.0
    memory_bytes_per_cycle: float = 1024.0
    wait_event_cycles: float = 2.0
    reserve_event_cycles: float = 2.0
    sync_event_cycles: float = 1.0
    copy_call_cycles: float = 4.0
    blocked_cycle_weight: float = 0.0
    kernel_launch_cycles: float = 0.0
    # Phase-duration scaling coefficients.
    dfb_wait_block_scale: float = 1.0
    dfb_reserve_block_scale: float = 1.0
    copy_duration_scale: float = 1.0
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
    measured_roofline_efficiency: float
    operational_intensity: float
    bound_classification: str
    roofline_base_cycles: float
    compute_ceiling_cycles: float
    memory_ceiling_cycles: float
    stall_cycles: float
    sync_cycles: float
    copy_overhead_cycles: float
    blocked_cycles_term: float
    launch_cycles: float
    # Phase-duration contributions to estimate.
    dfb_wait_block_contribution: float
    dfb_reserve_block_contribution: float
    copy_duration_contribution: float
    mismatch_reason: str
    needs_lower_level_model: bool


@dataclass
class KernelGroupEstimate:
    node: str
    kernel_count: int
    measured_cycles: int
    estimated_cycles: float
    abs_error_pct: float
    signed_error_pct: float
    aggregation_model: str
