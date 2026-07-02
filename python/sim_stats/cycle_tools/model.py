# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Assemble a CycleEstimate from per-kernel work records + a hardware profile."""

from __future__ import annotations

from .schedule import kernel_paths, program_cycles
from .types import CycleEstimate, HardwareProfile, KernelEstimate, KernelWork
from ..utils import node_from_kernel


def _role_from_kernel(kernel: str) -> str:
    """Derive read/compute/write/other from the kernel-name suffix."""
    node = node_from_kernel(kernel)
    suffix = kernel.removeprefix(f"{node}-") if node != kernel else ""
    return suffix if suffix in {"compute", "read", "write"} else "other"


def build_estimate(kernels: list[KernelWork], hw: HardwareProfile) -> CycleEstimate:
    """Assemble the canonical CycleEstimate from per-kernel work + a profile."""
    kernel_estimates: list[KernelEstimate] = []
    for kw in sorted(kernels, key=lambda k: k.kernel):
        compute, movement = kernel_paths(kw, hw)
        kernel_estimates.append(
            KernelEstimate(
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
    for ke in kernel_estimates:
        node_cycles[ke.node] = max(node_cycles.get(ke.node, 0.0), ke.cycles)

    profile = {
        "name": hw.name,
        "clock_ghz": hw.clock_ghz,
        "bytes_per_tile": hw.bytes_per_tile,
        "compute_rate_default": hw.compute_rate_default,
        "noc_bw": dict(hw.noc_bw),
        "noc_latency": dict(hw.noc_latency),
    }

    return CycleEstimate(
        profile_name=hw.name,
        profile=profile,
        program_cycles=program_cycles(kernels, hw),
        total_nodes=len(node_cycles),
        active_nodes=sum(1 for v in node_cycles.values() if v > 0.0),
        kernels=kernel_estimates,
    )
