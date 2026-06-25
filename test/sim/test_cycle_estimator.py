# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sim_stats cycle estimator logic.

Tests cover model behavior and parser edge cases:
- include/exclude zero-kernel rows
- bound classification tie behavior
- mismatch reason gating branches
- compute tile proxy and roofline contribution
- group critical-path aggregation
- feature extraction from event spans
"""

import math

from python.sim_stats.cycle_tools.model import (
    estimate_kernel_cycles,
    group_kernel_estimates,
    mismatch_reason,
)
from python.sim_stats.cycle_tools.parse import extract_kernel_features
from python.sim_stats.cycle_tools.types import (
    EstimatorConfig,
    KernelEstimate,
    KernelFeatures,
)
from python.sim_stats.cycle_tools.types import TraceEvent


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_zero_kernels_filtered_by_default() -> None:
    features = {
        "node0-compute": KernelFeatures(kernel="node0-compute", role="compute"),
    }

    rows = estimate_kernel_cycles(features, EstimatorConfig(), include_zero_kernels=False)

    assert rows == []


def test_zero_kernels_included_when_enabled() -> None:
    features = {
        "node0-compute": KernelFeatures(kernel="node0-compute", role="compute"),
    }

    rows = estimate_kernel_cycles(features, EstimatorConfig(), include_zero_kernels=True)

    assert len(rows) == 1
    assert rows[0].measured_cycles == 0
    assert rows[0].estimated_cycles == 0.0


def test_tied_roofline_defaults_to_memory_bound() -> None:
    features = {
        "node0-compute": KernelFeatures(
            kernel="node0-compute",
            role="compute",
            measured_cycles=10,
            wait_tiles=1,
            local_l1_tiles=1,
        ),
    }
    config = EstimatorConfig(
        flops_per_tile=1.0,
        bytes_per_tile=1.0,
        peak_flops_per_cycle=1.0,
        memory_bytes_per_cycle=1.0,
        wait_event_cycles=0.0,
        reserve_event_cycles=0.0,
        sync_event_cycles=0.0,
        copy_call_cycles=0.0,
    )

    rows = estimate_kernel_cycles(features, config, include_zero_kernels=True)

    assert len(rows) == 1
    assert rows[0].compute_ceiling_cycles == rows[0].memory_ceiling_cycles
    assert rows[0].bound_classification == "memory-bound"


def test_mismatch_reason_stall_dominated_threshold() -> None:
    feature = KernelFeatures(
        kernel="node0-compute",
        role="compute",
        measured_cycles=100,
        blocked_cycles=30,
    )

    reason = mismatch_reason(
        feature=feature,
        abs_error_pct=25.0,
        threshold_pct=20.0,
        blocked_cycles_term=0.0,
        estimated_cycles=50.0,
        compute_ceiling_cycles=1.0,
        memory_ceiling_cycles=0.5,
    )

    assert reason.startswith("stall-dominated")


def test_mismatch_reason_blocked_term_dominated_threshold() -> None:
    feature = KernelFeatures(
        kernel="node0-compute",
        role="compute",
        measured_cycles=100,
        blocked_cycles=0,
    )

    reason = mismatch_reason(
        feature=feature,
        abs_error_pct=25.0,
        threshold_pct=20.0,
        blocked_cycles_term=5.0,
        estimated_cycles=10.0,
        compute_ceiling_cycles=1.0,
        memory_ceiling_cycles=0.5,
    )

    assert reason.startswith("blocked-term-dominated")


def test_compute_tile_proxy_uses_wait_plus_reserve_tiles() -> None:
    features = {
        "node0-compute": KernelFeatures(
            kernel="node0-compute",
            role="compute",
            wait_tiles=2,
            reserve_tiles=3,
        ),
    }
    config = EstimatorConfig(
        flops_per_tile=10.0,
        peak_flops_per_cycle=10.0,
        bytes_per_tile=1.0,
        memory_bytes_per_cycle=1.0,
        wait_event_cycles=0.0,
        reserve_event_cycles=0.0,
        sync_event_cycles=0.0,
        copy_call_cycles=0.0,
    )

    rows = estimate_kernel_cycles(features, config, include_zero_kernels=True)

    assert len(rows) == 1
    assert rows[0].compute_ceiling_cycles == 5.0
    assert rows[0].estimated_cycles == 5.0


def test_group_aggregation_critical_path_formula() -> None:
    rows = [
        KernelEstimate(
            kernel="node0-compute",
            role="compute",
            measured_cycles=100,
            estimated_cycles=100.0,
            abs_error_pct=0.0,
            signed_error_pct=0.0,
            roofline_efficiency=0.0,
            measured_roofline_efficiency=0.0,
            operational_intensity=math.inf,
            bound_classification="compute-bound",
            roofline_base_cycles=0.0,
            compute_ceiling_cycles=0.0,
            memory_ceiling_cycles=0.0,
            stall_cycles=0.0,
            sync_cycles=20.0,
            copy_overhead_cycles=0.0,
            blocked_cycles_term=0.0,
            launch_cycles=0.0,
            dfb_wait_block_contribution=0.0,
            dfb_reserve_block_contribution=0.0,
            copy_duration_contribution=0.0,
            mismatch_reason="within-threshold",
            needs_lower_level_model=False,
        ),
        KernelEstimate(
            kernel="node0-read",
            role="read",
            measured_cycles=120,
            estimated_cycles=80.0,
            abs_error_pct=0.0,
            signed_error_pct=0.0,
            roofline_efficiency=0.0,
            measured_roofline_efficiency=0.0,
            operational_intensity=0.0,
            bound_classification="memory-bound",
            roofline_base_cycles=0.0,
            compute_ceiling_cycles=0.0,
            memory_ceiling_cycles=0.0,
            stall_cycles=0.0,
            sync_cycles=40.0,
            copy_overhead_cycles=0.0,
            blocked_cycles_term=0.0,
            launch_cycles=0.0,
            dfb_wait_block_contribution=0.0,
            dfb_reserve_block_contribution=0.0,
            copy_duration_contribution=0.0,
            mismatch_reason="within-threshold",
            needs_lower_level_model=False,
        ),
    ]

    groups = group_kernel_estimates(rows)

    assert len(groups) == 1
    g = groups[0]
    assert g.measured_cycles == 120
    assert g.estimated_cycles == 120.0
    assert g.signed_error_pct == 0.0
    assert g.abs_error_pct == 0.0


# ---------------------------------------------------------------------------
# Parse/extraction tests
# ---------------------------------------------------------------------------


def test_extract_features_accumulates_wait_reserve_and_copy_durations() -> None:
    events = [
        TraceEvent(0, "kernel_start", "node0-compute", {}),
        TraceEvent(1, "dfb_wait_begin", "node0-compute", {}),
        TraceEvent(5, "dfb_wait_end", "node0-compute", {"tiles": 2}),
        TraceEvent(6, "dfb_reserve_begin", "node0-compute", {}),
        TraceEvent(10, "dfb_reserve_end", "node0-compute", {"tiles": 3}),
        TraceEvent(12, "copy_start", "node0-compute", {}),
        TraceEvent(
            15,
            "copy_end",
            "node0-compute",
            {"tiles": 4, "local_l1": 1, "remote_l1": 2, "dram": 1},
        ),
        TraceEvent(16, "kernel_end", "node0-compute", {}),
    ]

    features = extract_kernel_features(events)
    f = features["node0-compute"]

    assert f.measured_cycles == 16
    assert f.dfb_wait_block_cycles == 4
    assert f.dfb_reserve_block_cycles == 4
    assert f.copy_duration_cycles == 3
    assert f.wait_tiles == 2
    assert f.reserve_tiles == 3
    assert f.copy_tiles == 4
    assert f.local_l1_tiles == 1
    assert f.remote_l1_tiles == 2
    assert f.dram_tiles == 1


def test_extract_features_closes_open_kernel_at_trace_end() -> None:
    events = [
        TraceEvent(10, "kernel_start", "node0-compute", {}),
        TraceEvent(20, "dfb_push", "node0-compute", {}),
    ]

    features = extract_kernel_features(events)
    f = features["node0-compute"]

    assert f.measured_cycles == 10
    assert f.blocked_cycles == 0
    assert f.active_cycles == 10


def test_extract_features_ignores_unpaired_phase_begin_events() -> None:
    events = [
        TraceEvent(0, "kernel_start", "node0-read", {}),
        TraceEvent(1, "dfb_wait_begin", "node0-read", {}),
        TraceEvent(2, "dfb_reserve_begin", "node0-read", {}),
        TraceEvent(3, "copy_start", "node0-read", {}),
        TraceEvent(4, "kernel_end", "node0-read", {}),
    ]

    features = extract_kernel_features(events)
    f = features["node0-read"]

    assert f.wait_count == 1
    assert f.reserve_count == 1
    assert f.copy_calls == 0
    assert f.dfb_wait_block_cycles == 0
    assert f.dfb_reserve_block_cycles == 0
    assert f.copy_duration_cycles == 0