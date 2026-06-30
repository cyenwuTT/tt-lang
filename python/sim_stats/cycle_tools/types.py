# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Data types for cycle estimation from simulator traces."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    blocked_cycle_weight: float = 0.0  # Keep at 0 to prevent leakage into predictions.
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

    # Phase-duration contributions captured in the estimate.
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


# ---------------------------------------------------------------------------
# v1.0 analytical peak model (scaffolding; not yet wired into the pipeline)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """Static hardware spec rates for the analytical peak model.

    Holds peak throughput/bandwidth constants that cannot be derived from a
    trace (the trace says *how much* work happens, not *how fast* the part runs).
    Canonical profiles live in ``hardware_profile.py`` and are looked up by name.
    A file-based loader for custom profiles is deferred until the field set is
    settled and sensitivity sweeps actually need it.
    """

    name: str
    # Compute throughput in tiles/cycle, keyed by (op_type, dtype).
    compute_rate: dict[tuple[str, str], float]
    # Fallback tiles/cycle when an (op_type, dtype) pair is not listed.
    compute_rate_default: float
    # Data-movement bandwidth in bytes/cycle, keyed by locality
    # ("local_l1", "remote_l1", "dram").
    noc_bw: dict[str, float]
    # Per-transfer fixed latency in cycles, keyed by locality.
    noc_latency: dict[str, float]
    # Core clock in GHz; used only for cycle<->ns reporting, never in the model.
    clock_ghz: float
    # Number of data-movement engines (reserved for future overlap modelling).
    dm_engines: int = 1

    def rate_for(self, op_type: str, dtype: str) -> float:
        """Peak tiles/cycle for an (op_type, dtype), falling back to the default."""
        return self.compute_rate.get((op_type, dtype), self.compute_rate_default)

    def bandwidth_for(self, locality: str) -> float:
        """Peak bytes/cycle for a locality, or 0.0 if unknown."""
        return self.noc_bw.get(locality, 0.0)

    def latency_for(self, locality: str) -> float:
        """Fixed per-transfer latency in cycles for a locality, or 0.0 if unknown."""
        return self.noc_latency.get(locality, 0.0)


@dataclass(frozen=True)
class OpWork:
    """A single operation extracted from the trace (v1.0 per-op work record)."""

    kind: str  # "compute" | "movement"
    op_type: str  # e.g. "matmul", "add", "exp", "copy"
    dtype: str = ""  # e.g. "bf16", "fp32" (compute ops)
    tiles: int = 0  # compute work
    bytes: int = 0  # movement work
    locality: str = ""  # "local_l1" | "remote_l1" | "dram" (movement ops)


@dataclass
class KernelWork:
    """Per-kernel collection of op records plus dependency structure (v1.0)."""

    kernel: str
    node_index: int = 0
    ops: list[OpWork] = field(default_factory=list[OpWork])
    # DFB / pipe names this kernel blocks on; filled by the dependency pass.
    blocks_on: list[str] = field(default_factory=list[str])
