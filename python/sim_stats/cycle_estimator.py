# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compatibility module for cycle estimation from simulator JSONL traces.

Preserves the ``tt-lang-sim-cycles`` entry point and the public API while the
implementation lives in focused modules under ``sim_stats.cycle_tools.*``.
"""

from __future__ import annotations

from .cycle_tools import (
    CycleEstimate,
    HardwareProfile,
    KernelEstimate,
    TraceEvent,
    build_estimate,
    load_estimate,
    main,
    parse_trace,
    print_detailed,
    print_summary,
    write_json,
)

__all__ = [
    "TraceEvent",
    "HardwareProfile",
    "KernelEstimate",
    "CycleEstimate",
    "parse_trace",
    "build_estimate",
    "print_summary",
    "print_detailed",
    "write_json",
    "load_estimate",
    "main",
]


if __name__ == "__main__":
    main()
