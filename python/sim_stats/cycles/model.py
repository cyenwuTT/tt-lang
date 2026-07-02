# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Assemble a CycleEstimate from per-kernel work records + a hardware profile."""

from __future__ import annotations

from .schedule import kernel_paths, program_cycles
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

    node_cycles: dict[str, float] = {}
    for ke in kernel_estimates:
        node_cycles[ke.node] = max(node_cycles.get(ke.node, 0.0), ke.cycles)

    return CycleEstimate(
        profile_name=hw.name,
        profile=hw.summary(),
        program_cycles=program_cycles(kernels, hw),
        total_nodes=len(node_cycles),
        active_nodes=sum(1 for v in node_cycles.values() if v > 0.0),
        kernels=kernel_estimates,
    )
