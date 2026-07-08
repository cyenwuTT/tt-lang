# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Assemble a CycleEstimate from per-kernel work records + a hardware profile."""

from __future__ import annotations

from .schedule import kernel_paths, program_from_node_bound, total_dram_bytes
from .types import CycleEstimate, HardwareProfile, KernelEstimate, KernelWork
from ..utils import node_from_kernel, role_from_kernel


def build_estimate(kernels: list[KernelWork], hw: HardwareProfile) -> CycleEstimate:
    """Assemble the canonical CycleEstimate from per-kernel work + a profile."""
    kernel_estimates: list[KernelEstimate] = []
    for kw in sorted(kernels, key=lambda k: k.kernel):
        compute, movement = kernel_paths(kw, hw)
        kernel_estimates.append(
            KernelEstimate(
                kernel=kw.kernel,
                node=node_from_kernel(kw.kernel),
                role=role_from_kernel(kw.kernel),
                compute_cycles=compute,
                movement_cycles=movement,
                cycles=max(compute, movement),
                bound="compute-bound" if compute > movement else "memory-bound",
            )
        )

    # Per-node rollup from the kernel estimates we just built — the per-node
    # bound is the max over nodes, reusing those cycles rather than re-walking
    # the ops through the schedule a second time.
    node_cycles: dict[str, float] = {}
    for ke in kernel_estimates:
        node_cycles[ke.node] = max(node_cycles.get(ke.node, 0.0), ke.cycles)
    node_bound = max(node_cycles.values(), default=0.0)

    prog_cycles, program_bound, dram_floor, _node_bound = program_from_node_bound(
        kernels, hw, node_bound
    )

    return CycleEstimate(
        profile_name=hw.name,
        profile=hw.summary(),
        program_cycles=prog_cycles,
        total_nodes=len(node_cycles),
        active_nodes=sum(1 for v in node_cycles.values() if v > 0.0),
        kernels=kernel_estimates,
        program_bound=program_bound,
        dram_floor=dram_floor,
        total_dram_bytes=total_dram_bytes(kernels, hw),
    )
