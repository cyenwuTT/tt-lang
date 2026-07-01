# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Overlap and critical-path combiner for the v1.0 analytical peak model.

Scaffolding for the v0.1 -> v1.0 restructure:

- :func:`kernel_cycles` implements the per-kernel ``max(compute, movement)``
  combine and is ready for use.
- :func:`program_cycles` is a PLACEHOLDER pending the dependency-DAG critical
  path (Step 5 of the restructure plan).

Not yet wired into the live estimation pipeline.
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


def kernel_cycles(work: KernelWork, hw: HardwareProfile) -> float:
    """Ideal-peak kernel cycles: the larger of the compute and movement paths.

    The compute engine and the data-movement engine run concurrently, so the
    kernel time is ``max`` of the two serial paths, not their sum.
    """
    compute_path = sum(op_cycles(o, hw) for o in work.ops if o.kind == "compute")
    movement_path = sum(op_cycles(o, hw) for o in work.ops if o.kind == "movement")
    return max(compute_path, movement_path)


def program_cycles(kernels: list[KernelWork], hw: HardwareProfile) -> float:
    """Program-level cycles under the ideal-peak, throughput-bound model.

    Two levels of overlap:
      - within a node: the reader / compute / writer kernels run on that core's
        concurrent RISCs, so the node's time is the ``max`` of its kernels.
      - across nodes: distinct nodes are separate cores running in parallel, so
        the program time is the ``max`` over nodes.

    Net: the program is throughput-bound by its slowest kernel. Under ideal-peak
    with full pipelining, connected producer/consumer kernels overlap in steady
    state, so there is no serial sum along a dependency chain.

    Deferred (would need the dependency DAG from kernel_block.on / dfb push-pop /
    pipe send-recv): fill/drain latency for small workloads, and explicit
    cross-node serialization — i.e. the latency regime rather than throughput.
    """
    per_node: dict[str, float] = {}
    for k in kernels:
        node = node_from_kernel(k.kernel)
        per_node[node] = max(per_node.get(node, 0.0), kernel_cycles(k, hw))
    return max(per_node.values(), default=0.0)
