# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cycle estimation package for simulator traces."""

from .cli import build_pipeline, main
from .model import build_estimate
from .parse import extract_kernel_work, parse_trace
from .report import (
    load_estimate,
    print_detailed,
    print_summary,
    write_json,
)
from .types import (
    CycleEstimate,
    HardwareProfile,
    KernelEstimate,
    KernelWork,
    OpWork,
    TraceEvent,
)

__all__ = [
    "TraceEvent",
    "HardwareProfile",
    "OpWork",
    "KernelWork",
    "KernelEstimate",
    "CycleEstimate",
    "parse_trace",
    "extract_kernel_work",
    "build_estimate",
    "build_pipeline",
    "print_summary",
    "print_detailed",
    "write_json",
    "load_estimate",
    "main",
]
