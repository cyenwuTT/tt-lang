# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the sim_stats cycle estimator (analytical ideal-peak model).

Covers: per-op cost, kernel/program combiners, invariants, the compute path
(against synthetic compute_op events), report rendering, JSON round-trip, and
hardware-profile loading.
"""

import json

import pytest

from python.sim_stats.cycle_tools.hardware_profile import (
    load_profile_json,
    resolve_profile,
)
from python.sim_stats.cycle_tools.model import build_estimate
from python.sim_stats.cycle_tools.parse import extract_kernel_work
from python.sim_stats.cycle_tools.report import (
    load_estimate,
    print_detailed,
    print_summary,
    write_json,
)
from python.sim_stats.cycle_tools.schedule import (
    kernel_cycles,
    kernel_paths,
    op_cycles,
    program_cycles,
)
from python.sim_stats.cycle_tools.types import (
    HardwareProfile,
    KernelWork,
    OpWork,
    TraceEvent,
)


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


# ---------------------------------------------------------------------------
# Movement path
# ---------------------------------------------------------------------------


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
            OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10),  # 5
            OpWork(kind="movement", op_type="copy", tiles=4, locality="dram"),  # 4
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
# Invariants / regression fixtures
#
# These assert structural PROPERTIES that hold for any valid input (bounds and
# guarantees, not exact placeholder arithmetic).
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
    # Overlap (max), never additive (sum): max(c, m) <= kernel <= c + m, and
    # strictly less than the additive sum when both paths are non-zero.
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


def test_program_cycles_is_max_across_parallel_nodes() -> None:
    # Distinct nodes are separate cores running in parallel -> max, not sum.
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10)],
        ),  # 5
        KernelWork(
            kernel="node1-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=16)],
        ),  # 8
    ]
    assert program_cycles(kernels, hw) == 8.0  # max(5, 8), not 13


def test_program_cycles_within_node_is_max_of_kernels() -> None:
    # Reader / compute / writer share one core's concurrent RISCs -> max.
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
        KernelWork(
            kernel="node0-write",
            ops=[OpWork(kind="movement", op_type="copy", tiles=2, locality="dram")],
        ),  # 2
    ]
    assert program_cycles(kernels, hw) == 5.0  # max(4, 5, 2)


def test_program_cycles_bounded_by_max_and_sum_of_kernels() -> None:
    # A program is no faster than its slowest kernel and no slower than running
    # every kernel serially.
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
    # bytes = 4 * 2 = 8; bw(dram) = 2 -> 4.0 cycles.
    hw = _hw()
    kw = KernelWork(
        kernel="node0-read",
        ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
    )
    assert kernel_cycles(kw, hw) == 4.0


# ---------------------------------------------------------------------------
# Compute path (consumer built against synthetic compute_op events)
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


# ---------------------------------------------------------------------------
# rate_for + report rendering
# ---------------------------------------------------------------------------


def test_rate_for_is_dtype_optional() -> None:
    hw = HardwareProfile(
        name="t",
        compute_rate={("matmul", "bf16"): 8.0, ("add", ""): 32.0},
        compute_rate_default=2.0,
        noc_bw={},
        noc_latency={},
        clock_ghz=1.0,
        bytes_per_tile=1.0,
    )
    assert hw.rate_for("matmul", "bf16") == 8.0  # exact (op, dtype)
    assert hw.rate_for("matmul", "fp32") == 2.0  # no (matmul, "") -> default
    assert hw.rate_for("add", "bf16") == 32.0  # op-type-only entry serves any dtype
    assert hw.rate_for("add") == 32.0  # dtype omitted -> (add, "")
    assert hw.rate_for("exp") == 2.0  # unknown op -> default


def test_kernel_paths_splits_compute_and_movement() -> None:
    hw = _hw()
    kw = KernelWork(
        kernel="node0-compute",
        ops=[
            OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10),  # 5
            OpWork(kind="movement", op_type="copy", tiles=4, locality="dram"),  # 4
        ],
    )
    assert kernel_paths(kw, hw) == (5.0, 4.0)


def test_detailed_report_shows_decomposition_and_program_total(capsys) -> None:
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10)],
        ),
    ]

    print_detailed(build_estimate(kernels, hw))
    out = capsys.readouterr().out

    assert "ideal-peak model" in out
    assert "node0-compute" in out
    assert "node0-read" in out
    assert "Program cycles" in out


def test_detailed_report_notes_empty_compute_path(capsys) -> None:
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),
    ]

    print_detailed(build_estimate(kernels, hw))
    out = capsys.readouterr().out

    assert "compute path is 0" in out


def test_summary_rolls_up_per_node_and_reports_utilization(capsys) -> None:
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),
        KernelWork(
            kernel="node0-compute",
            ops=[OpWork(kind="compute", op_type="matmul", dtype="bf16", tiles=10)],
        ),
        KernelWork(kernel="node1-read", ops=[]),  # idle node
        KernelWork(kernel="node1-compute", ops=[]),
    ]

    print_summary(build_estimate(kernels, hw))
    out = capsys.readouterr().out

    assert "Node" in out
    assert "node0" in out
    assert "1 / 2" in out
    assert "Bottleneck" in out
    assert "node1" not in out  # idle node hidden by default


def test_json_write_then_load_round_trip(tmp_path) -> None:
    hw = _hw()
    kernels = [
        KernelWork(
            kernel="node0-read",
            ops=[OpWork(kind="movement", op_type="copy", tiles=4, locality="dram")],
        ),
    ]
    estimate = build_estimate(kernels, hw)

    p = tmp_path / "report.json"
    write_json(p, estimate)
    loaded = load_estimate(p)

    assert loaded.profile_name == estimate.profile_name
    assert loaded.program_cycles == estimate.program_cycles
    assert [k.kernel for k in loaded.kernels] == [k.kernel for k in estimate.kernels]
    assert loaded.kernels[0].movement_cycles == 4.0


def test_load_estimate_rejects_non_report(tmp_path) -> None:
    p = tmp_path / "other.json"
    p.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_estimate(p)


def test_load_estimate_rejects_jsonl_trace(tmp_path) -> None:
    # A raw trace is JSON Lines (multiple objects), not a single report object.
    p = tmp_path / "trace.jsonl"
    p.write_text(
        '{"event": "kernel_start"}\n{"event": "kernel_end"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_estimate(p)


# ---------------------------------------------------------------------------
# Custom hardware profile loading (--hw-profile <name|path.json>)
# ---------------------------------------------------------------------------


def _write_profile(path, **overrides) -> None:
    data = {
        "name": "custom",
        "compute_rate": [["matmul", "bf16", 8.0]],
        "compute_rate_default": 4.0,
        "noc_bw": {"dram": 2.0},
        "noc_latency": {"dram": 1.0},
        "clock_ghz": 1.0,
        "bytes_per_tile": 2048.0,
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_profile_json_round_trip(tmp_path) -> None:
    p = tmp_path / "custom.json"
    _write_profile(p)

    hw = load_profile_json(p)

    assert hw.name == "custom"
    assert hw.rate_for("matmul", "bf16") == 8.0  # listed
    assert hw.rate_for("add", "bf16") == 4.0  # falls back to default
    assert hw.bandwidth_for("dram") == 2.0
    assert hw.latency_for("dram") == 1.0
    assert hw.bytes_per_tile == 2048.0


def test_resolve_profile_accepts_builtin_name_and_json_path(tmp_path) -> None:
    assert resolve_profile("wormhole_b0").name == "wormhole_b0"

    p = tmp_path / "mine.json"
    _write_profile(p, name="mine")
    assert resolve_profile(str(p)).name == "mine"


def test_load_profile_json_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_profile_json(tmp_path / "nope.json")


def test_load_profile_json_malformed_raises(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('{"name": "bad"}', encoding="utf-8")  # missing required keys
    with pytest.raises(ValueError):
        load_profile_json(p)
