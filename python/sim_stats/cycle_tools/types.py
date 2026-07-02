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


@dataclass(frozen=True)
class HardwareProfile:
    """Static hardware spec: the rates the trace can't provide (how fast the part runs).

    Built-in profiles live in ``hardware_profile.py`` (looked up by name); custom
    ones load from JSON via ``load_profile_json``.
    """

    name: str
    compute_rate: dict[tuple[str, str], float]  # tiles/cycle by (op_type, dtype)
    compute_rate_default: float  # fallback tiles/cycle
    noc_bw: dict[str, float]  # bytes/cycle by locality (local_l1/remote_l1/dram)
    noc_latency: dict[str, float]  # fixed cycles per transfer, by locality
    clock_ghz: float  # cycle<->ns reporting only, not used in the model
    bytes_per_tile: float  # movement tile size (provisional; bf16 = 2048 B)
    dm_engines: int = 1  # reserved for future overlap modelling

    def rate_for(self, op_type: str, dtype: str = "") -> float:
        """Peak tiles/cycle for an op.

        Tiered lookup: exact ``(op_type, dtype)``, then an op-type-only entry
        ``(op_type, "")``, then ``compute_rate_default``. The op-type-only tier
        lets rates be keyed by op_type alone when the trace carries no dtype.
        """
        for key in ((op_type, dtype), (op_type, "")):
            if key in self.compute_rate:
                return self.compute_rate[key]
        return self.compute_rate_default

    def bandwidth_for(self, locality: str) -> float:
        """Peak bytes/cycle for a locality, or 0.0 if unknown."""
        return self.noc_bw.get(locality, 0.0)

    def latency_for(self, locality: str) -> float:
        """Fixed per-transfer latency in cycles for a locality, or 0.0 if unknown."""
        return self.noc_latency.get(locality, 0.0)


@dataclass(frozen=True)
class OpWork:
    """A single operation extracted from the trace (per-op work record)."""

    kind: str  # "compute" | "movement"
    op_type: str  # e.g. "matmul", "add", "exp", "copy"
    dtype: str = ""  # e.g. "bf16", "fp32" (compute ops)
    tiles: int = 0  # work in tiles (compute tiles, or tiles moved)
    locality: str = ""  # "local_l1" | "remote_l1" | "dram" (movement ops)


@dataclass
class KernelWork:
    """Per-kernel collection of op records plus dependency structure."""

    kernel: str
    node_index: int = 0
    ops: list[OpWork] = field(default_factory=list[OpWork])
    # DFB / pipe names this kernel blocks on; filled by the dependency pass.
    blocks_on: list[str] = field(default_factory=list[str])


@dataclass(frozen=True)
class KernelEstimate:
    """Per-kernel cycle decomposition (a rendered result row)."""

    kernel: str
    node: str
    role: str
    compute_cycles: float
    movement_cycles: float
    cycles: float
    bound: str


@dataclass(frozen=True)
class CycleEstimate:
    """Canonical estimate result: the intermediate that render + JSON share.

    Produced fresh from a trace (:func:`model.build_estimate`) or loaded back from
    a saved JSON report (:func:`report.load_estimate`). All views (summary /
    detailed / JSON) are pure functions of this.
    """

    profile_name: str
    profile: dict[str, Any]  # resolved rates, embedded for reproducibility
    program_cycles: float
    total_nodes: int
    active_nodes: int
    kernels: list[KernelEstimate] = field(default_factory=list[KernelEstimate])
