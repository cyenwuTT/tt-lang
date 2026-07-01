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
from python.sim_stats.cycle_tools.parse import (
    extract_kernel_features,
    extract_kernel_work,
)
from python.sim_stats.cycle_tools.schedule import (
    kernel_cycles,
    op_cycles,
    program_cycles,
)
from python.sim_stats.cycle_tools.types import (
    EstimatorConfig,
    HardwareProfile,
    KernelEstimate,
    KernelFeatures,
    KernelWork,
    OpWork,
)
from python.sim_stats.cycle_tools.types import TraceEvent


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_zero_kernels_filtered_by_default() -> None:
    features = {
        "node0-compute": KernelFeatures(kernel="node0-compute", role="compute"),
    }

    rows = estimate_kernel_cycles(
        features, EstimatorConfig(), include_zero_kernels=False
    )

    assert rows == []


def test_zero_kernels_included_when_enabled() -> None:
    features = {
        "node0-compute": KernelFeatures(kernel="node0-compute", role="compute"),
    }

    rows = estimate_kernel_cycles(
        features, EstimatorConfig(), include_zero_kernels=True
    )

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


# ---------------------------------------------------------------------------
# v1.0 analytical peak model (Step 2: movement path)
# ---------------------------------------------------------------------------


def _hw() -> HardwareProfile:
    """Deterministic test profile (zero latency for clean arithmetic)."""
    return HardwareProfile(
        name="test",
        compute_rate={("matmul", "bf16"): 2.0},
        compute_rate_default=1.0,
        noc_bw={"local_l1": 8.0, "remote_l1": 4.0, "dram": 2.0},
        noc_latency={"local_l1": 0.0, "remote_l1": 0.0, "dram": 0.0},
        clock_ghz=1.0,
        bytes_per_tile=2.0,
    )


def test_extract_kernel_work_emits_movement_op_per_locality() -> None:
    events = [
        TraceEvent(0, "kernel_start", "node0-read", {}),
        TraceEvent(
            5,
            "copy_end",
            "node0-read",
            {"tiles": 4, "local_l1": 1, "remote_l1": 2, "dram": 1},
        ),
        TraceEvent(6, "kernel_end", "node0-read", {}),
    ]

    work = extract_kernel_work(events)
    kw = work["node0-read"]

    assert kw.node_index == 0
    assert [(o.locality, o.tiles) for o in kw.ops] == [
        ("local_l1", 1),
        ("remote_l1", 2),
        ("dram", 1),
    ]
    assert all(o.kind == "movement" for o in kw.ops)


def test_movement_op_cost_is_tiles_times_bytes_over_bandwidth() -> None:
    hw = _hw()
    op = OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")
    # bytes = 4 tiles * 2 B/tile = 8; bw(dram) = 2 -> 4.0 cycles
    assert op_cycles(op, hw) == 4.0


def test_movement_cost_monotonic_in_tiles() -> None:
    hw = _hw()
    one = op_cycles(
        OpWork(kind="movement", op_type="copy", tiles=4, locality="dram"), hw
    )
    two = op_cycles(
        OpWork(kind="movement", op_type="copy", tiles=8, locality="dram"), hw
    )
    assert two == 2 * one


def test_kernel_cycles_is_max_of_compute_and_movement_paths() -> None:
    hw = _hw()
    kw = KernelWork(
        kernel="node0-compute",
        ops=[
            OpWork(
                kind="compute", op_type="matmul", dtype="bf16", tiles=10
            ),  # 10/2 = 5
            OpWork(
                kind="movement", op_type="copy", tiles=4, locality="dram"
            ),  # 8/2 = 4
        ],
    )
    # max(compute_path=5, movement_path=4) = 5
    assert kernel_cycles(kw, hw) == 5.0


def test_zero_work_zero_cycles() -> None:
    hw = _hw()
    kw = KernelWork(kernel="node0-compute")
    assert kernel_cycles(kw, hw) == 0.0
    assert program_cycles([kw], hw) == 0.0


# ---------------------------------------------------------------------------
# v1.0 invariants / regression fixtures (Step 3)
#
# These assert structural PROPERTIES that must hold for any valid input and
# survive later steps (e.g. Step 5 replacing the program_cycles combiner). They
# test bounds and guarantees, not a placeholder's exact arithmetic, so they do
# not need rewriting when the combiner changes.
# ---------------------------------------------------------------------------


def _mixed_kernel() -> KernelWork:
    """A kernel with both a compute and a movement op (compute path dominates)."""
    return KernelWork(
        kernel="node0-compute",
        ops=[
            OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10),  # 5
            OpWork(kind="movement", op_type="copy", tiles=4, locality="dram"),  # 4
        ],
    )


def test_compute_cost_monotonic_in_tiles() -> None:
    # Doubling compute tiles doubles the compute cost (rate fixed).
    hw = _hw()
    one = op_cycles(
        OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10), hw
    )
    two = op_cycles(
        OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=20), hw
    )
    assert one == 5.0
    assert two == 2 * one


def test_kernel_cycles_never_double_counts() -> None:
    # The headline of the restructure: overlap (max), never additive (sum).
    # max(compute, movement) <= kernel <= compute + movement, and strictly less
    # than the additive sum when both paths are non-zero.
    hw = _hw()
    kw = _mixed_kernel()
    compute_path, movement_path = 5.0, 4.0
    k = kernel_cycles(kw, hw)

    assert max(compute_path, movement_path) <= k <= compute_path + movement_path
    assert k < compute_path + movement_path


def test_kernel_cycles_non_negative() -> None:
    hw = _hw()
    assert kernel_cycles(_mixed_kernel(), hw) >= 0.0
    assert kernel_cycles(KernelWork(kernel="node0-read"), hw) >= 0.0


def test_kernel_cycles_deterministic() -> None:
    # Same (work, profile) -> identical output (pure function, no hidden state).
    hw = _hw()
    kw = _mixed_kernel()
    assert kernel_cycles(kw, hw) == kernel_cycles(kw, hw)


def test_program_cycles_bounded_by_max_and_sum_of_kernels() -> None:
    # Bound holds for the current per-node-max placeholder AND the future
    # dependency-DAG critical path (Step 5): a program can be no faster than its
    # slowest kernel and no slower than running every kernel serially.
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),  # 4
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10)],
        ),  # 5
    ]
    ks = [kernel_cycles(k, hw) for k in kernels]
    prog = program_cycles(kernels, hw)

    assert max(ks) <= prog <= sum(ks)


def test_hand_derived_simple_kernel_value() -> None:
    # Hand-derived golden: a read kernel moving 4 dram tiles.
    # bytes = 4 * 2 = 8; bw(dram) = 2 -> 4.0 cycles. Locks kernel-level output
    # (stable across the Step 5 program-combiner change).
    hw = _hw()
    kw = KernelWork(
        kernel="node0-read",
        ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
    )
    assert kernel_cycles(kw, hw) == 4.0


# ---------------------------------------------------------------------------
# v1.0 compute path (Step 4) — consumer built against synthetic compute_op
# events (the agreed sim instrumentation contract). These tests are final: when
# the simulator starts emitting compute_op events, the same parser handles them
# unchanged.
# ---------------------------------------------------------------------------


def test_extract_kernel_work_reads_compute_op_events() -> None:
    events = [
        TraceEvent(0, "kernel_start", "node0-compute", {}),
        TraceEvent(
            2,
            "compute_op",
            "node0-compute",
            {"op_type": "matmul", "dtype": "bf16", "tiles": 10},
        ),
        TraceEvent(3, "kernel_end", "node0-compute", {}),
    ]

    work = extract_kernel_work(events)
    ops = work["node0-compute"].ops

    assert len(ops) == 1
    assert ops[0].kind == "compute"
    assert ops[0].op_type == "matmul"
    assert ops[0].dtype == "bf16"
    assert ops[0].tiles == 10


def test_kernel_cycles_from_trace_is_max_of_compute_and_movement() -> None:
    # End-to-end through the parser: a kernel that both computes and moves data.
    events = [
        TraceEvent(0, "kernel_start", "node0-compute", {}),
        TraceEvent(
            1,
            "compute_op",
            "node0-compute",
            {"op_type": "matmul", "dtype": "bf16", "tiles": 10},  # 10/2 = 5
        ),
        TraceEvent(
            2, "copy_end", "node0-compute", {"tiles": 4, "dram": 4}
        ),  # 4*2/2 = 4
        TraceEvent(3, "kernel_end", "node0-compute", {}),
    ]

    work = extract_kernel_work(events)
    kw = work["node0-compute"]
    kinds = sorted(o.kind for o in kw.ops)

    assert kinds == ["compute", "movement"]
    assert kernel_cycles(kw, _hw()) == 5.0  # max(5, 4)


def test_compute_op_with_zero_tiles_is_ignored() -> None:
    events = [
        TraceEvent(0, "kernel_start", "node0-compute", {}),
        TraceEvent(1, "compute_op", "node0-compute", {"op_type": "add", "tiles": 0}),
        TraceEvent(2, "kernel_end", "node0-compute", {}),
    ]

    work = extract_kernel_work(events)
    assert work["node0-compute"].ops == []


def test_compute_op_missing_dtype_falls_back_to_default_rate() -> None:
    # op_type + tiles only (the minimum-viable contract); dtype defaults to "".
    # rate_for("matmul", "") misses the (matmul, bf16) entry -> default rate 1.0.
    hw = _hw()
    events = [
        TraceEvent(0, "kernel_start", "node0-compute", {}),
        TraceEvent(1, "compute_op", "node0-compute", {"op_type": "matmul", "tiles": 4}),
        TraceEvent(2, "kernel_end", "node0-compute", {}),
    ]

    work = extract_kernel_work(events)
    # 4 tiles / default rate 1.0 = 4.0
    assert kernel_cycles(work["node0-compute"], hw) == 4.0
