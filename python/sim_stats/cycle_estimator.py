# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compatibility module for cycle estimation from simulator JSONL traces.

This file preserves the public entrypoint and historical symbols while the
implementation lives in focused modules under ``sim_stats.cycle_tools.*``.
"""

from __future__ import annotations

from .cycle_tools import (
    EstimatorConfig,
    KernelEstimate,
    TraceEvent,
    ablation_metrics,
    bound_classification,
    build_estimation_pipeline,
    estimate_kernel_cycles,
    extract_kernel_features,
    feature_provenance,
    group_kernel_estimates,
    main,
    mismatch_reason,
    parse_trace,
    print_report,
    role_calibration_suggestions,
    write_json_report,
)

__all__ = [
    "TraceEvent",
    "EstimatorConfig",
    "KernelEstimate",
    "parse_trace",
    "extract_kernel_features",
    "bound_classification",
    "mismatch_reason",
    "estimate_kernel_cycles",
    "group_kernel_estimates",
    "ablation_metrics",
    "feature_provenance",
    "role_calibration_suggestions",
    "print_report",
    "write_json_report",
    "build_estimation_pipeline",
    "main",
]


if __name__ == "__main__":
    main()
