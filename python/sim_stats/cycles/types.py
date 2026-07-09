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

    Built-in profile instances (``WORMHOLE_B0``, ``DEFAULT``) are defined at the
    bottom of this module (data only). Name/path resolution and JSON loading live
    in :mod:`model` (``resolve_profile`` / ``load_profile_json`` / ``get_profile``).
    """

    name: str
    compute_rate: dict[tuple[str, str], float]  # tiles/cycle by (op_type, dtype)
    compute_rate_default: float  # fallback tiles/cycle
    noc_bw: dict[str, float]  # bytes/cycle by locality (local_l1/remote_l1/dram)
    noc_latency: dict[str, float]  # fixed cycles per transfer, by locality
    clock_ghz: float  # cycle<->ns reporting only, not used in the model
    bytes_per_tile: float  # movement tile size (provisional; bf16 = 2048 B)
    dm_engines: int = 1  # reserved for future overlap modelling
    dram_aggregate_bw: float = 0.0  # shared DRAM peak, bytes/cycle (0 = off)

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

    def aggregate_dram_bandwidth(self) -> float:
        """Program-wide shared DRAM peak in bytes/cycle, or 0.0 if unmodeled.

        Unlike ``noc_bw["dram"]`` (a per-core NoC lane), this is the single
        GDDR6 controller pool shared by all cores. 0.0 means no program-level
        ceiling (legacy behavior).
        """
        return self.dram_aggregate_bw

    def summary(self) -> dict[str, Any]:
        """Serializable snapshot embedded in a report for reproducibility."""
        return {
            "name": self.name,
            "clock_ghz": self.clock_ghz,
            "bytes_per_tile": self.bytes_per_tile,
            "compute_rate_default": self.compute_rate_default,
            "noc_bw": dict(self.noc_bw),
            "noc_latency": dict(self.noc_latency),
            "dram_aggregate_bw": self.dram_aggregate_bw,
        }


@dataclass(frozen=True)
class OpWork:
    """A single operation extracted from the trace (per-op work record)."""

    kind: str  # "compute" | "movement"
    op_type: str  # e.g. "matmul", "add", "exp", "copy"
    dtype: str = ""  # e.g. "bf16", "fp32" (compute ops)
    tiles: int = 0  # work in tiles (compute tiles, or tiles moved)
    locality: str = ""  # "local_l1" | "remote_l1" | "dram" (movement ops)
    direction: str = ""  # "read" | "write" (movement ops)


@dataclass
class KernelWork:
    """Per-kernel collection of op records extracted from the trace."""

    kernel: str
    ops: list[OpWork] = field(default_factory=list[OpWork])


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
class NodeEstimate:
    """Per-node rollup row: the max over a node's kernels (concurrent RISCs)."""

    node: str
    compute: float
    movement: float
    cycles: float
    bound: str  # "compute" | "memory"


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
    program_bound: str = "per-node"  # "per-node" | "aggregate-dram"
    dram_floor: float = 0.0
    total_dram_bytes: float = 0.0
    dram_read_bytes: float = 0.0
    dram_write_bytes: float = 0.0
    nodes: list[NodeEstimate] = field(default_factory=list[NodeEstimate])
    node_bound: float = 0.0  # max over nodes of per-node cycles (throughput)
    node_bound_reason: str = "compute"  # bound of the slowest node ("compute"|"memory")
    node_fill_drain: float = 0.0  # Tier-1 fill/drain on the per-node path


# ---------------------------------------------------------------------------
# Built-in hardware profiles
# ---------------------------------------------------------------------------

# Wormhole B0 (80 Tensix cores). Value sources + caveats: see the "wormhole_b0
# provenance" table in docs/development/CycleEstimator.md.
# Tile units differ per family: matmul = MAC volume (M*K*N), eltwise/unary/reduce
# = output tiles.
WORMHOLE_B0 = HardwareProfile(
    name="wormhole_b0",
    compute_rate={("matmul", ""): 1.0 / 64},  # tiles/cycle; HiFi4 (tt-lang default)
    compute_rate_default=1.0 / 32,  # tiles/cycle (SFPU)
    noc_bw={"local_l1": 25.3, "remote_l1": 25.3, "dram": 25.3},  # bytes/cycle
    noc_latency={"local_l1": 293.0, "remote_l1": 293.0, "dram": 293.0},  # cycles
    clock_ghz=1.0,  # GHz
    bytes_per_tile=2048.0,  # bytes (bf16)
    dm_engines=2,  # engines
    dram_aggregate_bw=288.0,  # bytes/cycle (shared GDDR6 pool); see provenance table
)

_PROFILES: dict[str, HardwareProfile] = {
    WORMHOLE_B0.name: WORMHOLE_B0,
}

DEFAULT = WORMHOLE_B0
