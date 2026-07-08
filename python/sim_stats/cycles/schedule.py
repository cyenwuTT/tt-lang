# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Overlap combiner for the analytical ideal-peak model.

Turns per-op work into cycles and combines them:

- :func:`op_cycles` — one op's ideal-peak cycles (work / peak-rate).
- :func:`kernel_cycles` — per-kernel ``max(compute, movement)`` (concurrent engines).
- :func:`program_cycles` — throughput-bound ``max`` within a node and across nodes,
  floored by the shared aggregate-DRAM ceiling (:func:`program_breakdown`).

The dependency-DAG latency regime (fill/drain, cross-node serialization) is out of
scope; see docs/development/CycleEstimator.md.
"""

from __future__ import annotations

from .types import HardwareProfile, KernelWork, OpWork
from ..utils import node_from_kernel


def op_cycles(op: OpWork, hw: HardwareProfile) -> float:
    """Ideal-peak cycles for a single op: work / peak-rate."""
    if op.kind == "compute":
        rate = hw.rate_for(op.op_type, op.dtype)
        return op.tiles / rate if rate > 0.0 else 0.0
    if op.kind == "movement":
        bw = hw.bandwidth_for(op.locality)
        moved_bytes = op.tiles * hw.bytes_per_tile
        transfer = moved_bytes / bw if bw > 0.0 else 0.0
        return hw.latency_for(op.locality) + transfer
    return 0.0


def kernel_paths(work: KernelWork, hw: HardwareProfile) -> tuple[float, float]:
    """Return the (compute_path, movement_path) cycle totals for a kernel."""
    compute_path = sum(op_cycles(o, hw) for o in work.ops if o.kind == "compute")
    movement_path = sum(op_cycles(o, hw) for o in work.ops if o.kind == "movement")
    return compute_path, movement_path


def total_dram_bytes(kernels: list[KernelWork], hw: HardwareProfile) -> float:
    """Program-wide bytes that hit the shared GDDR6 pool.

    Only ``locality == "dram"`` movement counts: local-L1 and remote-L1
    (multicast) traffic never touches the DRAM controller and must be excluded
    from the aggregate ceiling.
    """
    return sum(
        o.tiles * hw.bytes_per_tile
        for k in kernels
        for o in k.ops
        if o.kind == "movement" and o.locality == "dram"
    )


def kernel_cycles(work: KernelWork, hw: HardwareProfile) -> float:
    """Ideal-peak kernel cycles: the larger of the compute and movement paths.

    The compute engine and the data-movement engine run concurrently, so the
    kernel time is ``max`` of the two serial paths, not their sum.
    """
    compute_path, movement_path = kernel_paths(work, hw)
    return max(compute_path, movement_path)


def program_cycles(kernels: list[KernelWork], hw: HardwareProfile) -> float:
    """Program-level cycles under the ideal-peak, throughput-bound model.

    Two levels of overlap:
      - within a node: the reader / compute / writer kernels run on that core's
        concurrent RISCs, so the node's time is the ``max`` of its kernels.
      - across nodes: distinct nodes are separate cores running in parallel, so
        the program time is the ``max`` over nodes.

    A third bound sits above the two overlaps: the shared GDDR6 pool, taken as a
    ``max`` with the per-node bound (see :func:`program_breakdown`). Rationale
    and the deferred latency regime: docs/development/CycleEstimator.md.
    """
    return program_breakdown(kernels, hw)[0]


def program_breakdown(
    kernels: list[KernelWork], hw: HardwareProfile
) -> tuple[float, str, float, float]:
    """Program cycles plus which bound set them.

    Returns ``(program_cycles, program_bound, dram_floor, node_bound)`` where
    ``program_bound`` is ``"aggregate-dram"`` if the shared DRAM floor dominates,
    else ``"per-node"``. See :func:`program_cycles` for the model rationale.
    """
    per_node: dict[str, float] = {}
    for k in kernels:
        node = node_from_kernel(k.kernel)
        per_node[node] = max(per_node.get(node, 0.0), kernel_cycles(k, hw))
    node_bound = max(per_node.values(), default=0.0)
    return program_from_node_bound(kernels, hw, node_bound)


def program_from_node_bound(
    kernels: list[KernelWork], hw: HardwareProfile, node_bound: float
) -> tuple[float, str, float, float]:
    """Select the program bound given an already-computed per-node ``node_bound``.

    Takes the ``max`` of the per-node throughput bound and the shared aggregate
    DRAM floor. Callers that have already rolled up per-node cycles (e.g.
    :func:`model.build_estimate`) pass ``node_bound`` here to avoid re-walking
    the kernels; :func:`program_breakdown` computes it and delegates.
    """
    agg_bw = hw.aggregate_dram_bandwidth()
    dram_floor = total_dram_bytes(kernels, hw) / agg_bw if agg_bw > 0.0 else 0.0

    if dram_floor > node_bound:
        return dram_floor, "aggregate-dram", dram_floor, node_bound
    return node_bound, "per-node", dram_floor, node_bound
