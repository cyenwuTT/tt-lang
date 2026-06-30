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
        transfer = op.bytes / bw if bw > 0.0 else 0.0
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
    """Program-level cycles across all kernels.

    PLACEHOLDER (Step 5): the real version walks the kernel dependency DAG
    (edges from ``kernel_block.on`` + DFB push/pop + pipe send/recv) and returns
    its critical path. For now it approximates each node by its slowest kernel
    and sums across nodes, pending the dependency reconstruction.
    """
    per_node: dict[str, float] = {}
    for k in kernels:
        node = node_from_kernel(k.kernel)
        per_node[node] = max(per_node.get(node, 0.0), kernel_cycles(k, hw))
    return sum(per_node.values())
