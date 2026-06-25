"""Cycle estimation package for simulator traces."""

from .cli import build_estimation_pipeline, main
from .model import (
    estimate_kernel_cycles,
    group_kernel_estimates,
    mismatch_reason,
)
from .parse import extract_kernel_features, parse_trace
from .report import (
    ablation_metrics,
    feature_provenance,
    print_report,
    role_calibration_suggestions,
    write_json_report,
)
from .types import (
    EstimatorConfig,
    KernelEstimate,
    KernelFeatures,
    KernelGroupEstimate,
    TraceEvent,
)

__all__ = [
    "TraceEvent",
    "KernelFeatures",
    "EstimatorConfig",
    "KernelEstimate",
    "KernelGroupEstimate",
    "parse_trace",
    "extract_kernel_features",
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
