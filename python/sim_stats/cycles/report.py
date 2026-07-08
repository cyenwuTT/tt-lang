# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Rendering and JSON serialization for cycle estimates (pure over CycleEstimate)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from .types import CycleEstimate, KernelEstimate
from ..utils import abbrev_count, node_sort_key

_TOOL = "tt-lang-sim-cycles"
_SCHEMA_VERSION = 1
_MIN_WIDTH = 78
_NUM_W = 10  # numeric column width (fits headers + abbreviated values)
_LABEL_PAD = 16  # gap between the label column and the first numeric column
# label + 3 numeric cols (each led by a space) + two-space gap + widest bound.
_ROW_TAIL = 3 * (_NUM_W + 1) + 2 + len("compute")


def _short_bound(bound: str) -> str:
    """ "compute-bound" -> "compute"; the table header already says "Bound"."""
    return bound.split("-", 1)[0]


def _label_width(labels: list[str], header: str) -> int:
    """Label column sized to the longest label (or header), plus a small pad."""
    longest = max((len(x) for x in labels), default=0)
    return max(len(header), longest) + _LABEL_PAD


def _row(
    label: str, compute: float, movement: float, cycles: float, bound: str, label_w: int
) -> str:
    return (
        f"{label:<{label_w}} {abbrev_count(compute):>{_NUM_W}} "
        f"{abbrev_count(movement):>{_NUM_W}} {abbrev_count(cycles):>{_NUM_W}}  {bound}"
    )


def _per_node_rollup(
    estimate: CycleEstimate,
) -> dict[str, tuple[float, float, float, str]]:
    """Per-node (compute, movement, cycles, bound) — max over the node's kernels."""
    agg: dict[str, tuple[float, float, float]] = {}
    for ke in estimate.kernels:
        c, m, cy = agg.get(ke.node, (0.0, 0.0, 0.0))
        agg[ke.node] = (
            max(c, ke.compute_cycles),
            max(m, ke.movement_cycles),
            max(cy, ke.cycles),
        )
    return {
        node: (c, m, cy, "compute" if c > m else "memory")
        for node, (c, m, cy) in agg.items()
    }


def _header(estimate: CycleEstimate, unit: str, label_w: int, width: int) -> None:
    print("\n" + "=" * width)
    print("Cycle Estimate — ideal-peak model")
    print(f"hw-profile: {estimate.profile_name}")
    print("=" * width)  # title block / tables separator
    print(
        f"{unit:<{label_w}} {'Compute':>{_NUM_W}} "
        f"{'Movement':>{_NUM_W}} {'Cycles':>{_NUM_W}}  Bound"
    )
    print("." * width)


def _kv(label: str, value: str, note: str) -> str:
    """A summary line: left label, value column, parenthetical note."""
    return f"{label:<15}:  {value:<10}({note})"


def _human_bytes(n: float) -> str:
    """Bytes as a compact decimal magnitude with a 1-decimal unit (e.g. 50.3 MB)."""
    for unit, divisor in (("B", 1.0), ("KB", 1e3), ("MB", 1e6), ("GB", 1e9)):
        if abs(n) / divisor < 1000.0:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / divisor:.1f} {unit}"
    return f"{n / 1e9:.1f} GB"


def _per_node_max(
    active: dict[str, tuple[float, float, float, str]],
) -> tuple[float, str]:
    """The slowest node's cycles and its bound reason (compute/memory).

    This is schedule's per-node bound; ties break by node order for a stable
    reason. Empty (no active nodes) -> (0.0, "-").
    """
    if not active:
        return 0.0, "-"
    max_cy = max(v[2] for v in active.values())
    at_max = (n for n, v in active.items() if v[2] == max_cy)
    slowest = sorted(at_max, key=node_sort_key)
    return max_cy, active[slowest[0]][3]


def _stats_footer(
    estimate: CycleEstimate,
    width: int,
    rollup: dict[str, tuple[float, float, float, str]] | None = None,
) -> None:
    """Bound summary table, optional DRAM block, and program/per-node stats.

    ``rollup`` may be passed by a caller that already computed it (the summary view)
    to avoid recomputing; the detailed view lets it default.
    """
    if rollup is None:
        rollup = _per_node_rollup(estimate)
    active = {n: v for n, v in rollup.items() if v[2] > 0.0}

    # Bound summary table (active nodes only) — its own section.
    print("-" * width)
    print(f"{'Type':<10}{'Nodes':>8}{'Avg Cycles':>14}{'Max':>14}   Max node")
    print("." * width)
    by_bound: dict[str, list[tuple[str, float]]] = {}
    for node, (_c, _m, cy, bound) in active.items():
        by_bound.setdefault(bound, []).append((node, cy))
    for bound in ("compute", "memory"):  # always show both types
        rows = by_bound.get(bound, [])
        count = len(rows)
        if rows:
            avg = sum(cy for _, cy in rows) / count
            max_cy = max(cy for _, cy in rows)
            max_node = sorted((n for n, cy in rows if cy == max_cy), key=node_sort_key)[
                0
            ]
            avg_s, max_s = abbrev_count(avg), abbrev_count(max_cy)
        else:
            # Empty bound: dashes rather than 0.00, matching the "Max node" column.
            avg_s = max_s = max_node = "-"
        print(f"{bound:<10}{count:>8}{avg_s:>14}{max_s:>14}   {max_node}")

    # DRAM (shared) block — only when the profile models an aggregate ceiling.
    agg_bw = float(estimate.profile.get("dram_aggregate_bw", 0.0))
    if agg_bw > 0.0:
        clock = float(estimate.profile.get("clock_ghz", 1.0))
        gbps = agg_bw * clock
        print("-" * width)
        print("DRAM (shared)")
        print("." * width)
        print(f"  {'traffic':<15}:  {_human_bytes(estimate.total_dram_bytes)}")
        print(
            f"  {'bandwidth':<15}:  {agg_bw:g} B/cyc   "
            f"({gbps:g} GB/s @ {clock:.1f} GHz)"
        )
        print(f"  {'floor':<15}:  {abbrev_count(estimate.dram_floor)}")

    # Summary — its own section.
    idle = estimate.total_nodes - estimate.active_nodes
    node_max, node_reason = _per_node_max(active)
    nodes = f"{estimate.active_nodes} / {estimate.total_nodes}"
    print("-" * width)
    prog = abbrev_count(estimate.program_cycles)
    print(_kv("Program cycles", prog, estimate.program_bound))
    print(_kv("Per-node max", abbrev_count(node_max), node_reason))
    print(_kv("Active nodes", nodes, f"{idle} idle"))
    print("=" * width)
    if sum(k.compute_cycles for k in estimate.kernels) == 0.0:
        print(
            "note: compute path is 0 — the trace has no compute_op events "
            "(compute category filtered out, or a pre-instrumentation trace); "
            "movement-only estimate."
        )


def print_detailed(estimate: CycleEstimate) -> None:
    """Detailed per-kernel view — complete, includes zero rows."""
    label_w = _label_width([ke.kernel for ke in estimate.kernels], "Kernel")
    width = max(_MIN_WIDTH, label_w + _ROW_TAIL)
    _header(estimate, "Kernel", label_w, width)
    for ke in estimate.kernels:
        print(
            _row(
                ke.kernel,
                ke.compute_cycles,
                ke.movement_cycles,
                ke.cycles,
                _short_bound(ke.bound),
                label_w,
            )
        )
    _stats_footer(estimate, width)


def print_summary(estimate: CycleEstimate, include_zero: bool = False) -> None:
    """Per-node rollup (the default view).

    Each node's columns are the max over its kernels (concurrent RISCs), matching
    the program combiner.
    """
    rollup = _per_node_rollup(estimate)
    label_w = _label_width(list(rollup), "Node")
    width = max(_MIN_WIDTH, label_w + _ROW_TAIL)
    _header(estimate, "Node", label_w, width)
    for node in sorted(rollup, key=node_sort_key):
        compute, movement, cyc, bound = rollup[node]
        if not include_zero and cyc == 0.0:
            continue
        print(_row(node, compute, movement, cyc, bound, label_w))
    _stats_footer(estimate, width, rollup)


def write_json(path: Path, estimate: CycleEstimate) -> None:
    """Serialize the full, self-describing estimate (for analysis reuse)."""
    payload = {
        "tool": _TOOL,
        "schema_version": _SCHEMA_VERSION,
        "model": "ideal-peak",
        "profile": estimate.profile,
        "program_cycles": estimate.program_cycles,
        "program_bound": estimate.program_bound,
        "dram_floor": estimate.dram_floor,
        "total_dram_bytes": estimate.total_dram_bytes,
        "total_nodes": estimate.total_nodes,
        "active_nodes": estimate.active_nodes,
        "kernels": [asdict(k) for k in estimate.kernels],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_estimate(path: Path | str) -> CycleEstimate:
    """Load a saved report JSON back into a CycleEstimate, with validation.

    Raises FileNotFoundError if the file is missing, or ValueError if it is not a
    tt-lang-sim-cycles report (including the common mistake of passing a raw
    JSON-Lines trace instead of a saved report).
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"report file not found: {p}") from None

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(
            f"{p} is not a cycle report: not a single JSON object "
            "(a raw --trace file is JSON Lines, not a report)"
        ) from None

    if not isinstance(raw, dict):
        raise ValueError(f"{p} is not a tt-lang-sim-cycles report (not a JSON object)")

    # Give the decoded JSON a concrete type so the reads below are not "unknown".
    data = cast("dict[str, Any]", raw)
    if data.get("tool") != _TOOL or "kernels" not in data:
        raise ValueError(
            f"{p} is not a tt-lang-sim-cycles report (missing tool marker or kernels)"
        )

    try:
        raw_kernels: list[dict[str, Any]] = data["kernels"]
        kernels = [KernelEstimate(**k) for k in raw_kernels]
        profile: dict[str, Any] = data.get("profile", {})
        return CycleEstimate(
            profile_name=str(profile.get("name", data.get("profile_name", "?"))),
            profile=profile,
            program_cycles=float(data["program_cycles"]),
            total_nodes=int(data.get("total_nodes", 0)),
            active_nodes=int(data.get("active_nodes", 0)),
            kernels=kernels,
            program_bound=str(data.get("program_bound", "per-node")),
            dram_floor=float(data.get("dram_floor", 0.0)),
            total_dram_bytes=float(data.get("total_dram_bytes", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed cycle report {p}: {exc}") from None
