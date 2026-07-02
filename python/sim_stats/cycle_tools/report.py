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
from ..utils import node_sort_key

_TOOL = "tt-lang-sim-cycles"
_SCHEMA_VERSION = 1
_WIDTH = 78
_FRAME = "=" * _WIDTH  # report frame (top / bottom only)
_SECT = "-" * _WIDTH  # section break
_HDR = "." * _WIDTH  # column-header underline


def _short_bound(bound: str) -> str:
    """ "compute-bound" -> "compute"; the table header already says "Bound"."""
    return bound.split("-", 1)[0]


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


def _header(estimate: CycleEstimate, unit: str) -> None:
    print("\n" + _FRAME)
    print("Cycle Estimate — ideal-peak model")
    print(f"hw-profile: {estimate.profile_name}")
    print(_FRAME)  # title block / tables separator
    print(f"{unit:<28} {'Compute':>12} {'Movement':>12} {'Cycles':>12}  Bound")
    print(_HDR)


def _bottleneck(active: dict[str, tuple[float, float, float, str]]) -> str:
    """Node(s) setting program time. Ties (common under ideal-peak) are reported
    as a count + resource, not a single arbitrary node."""
    if not active:
        return "none (no active nodes)"
    max_cy = max(v[2] for v in active.values())
    at_max = [(n, v[3]) for n, v in active.items() if v[2] == max_cy]
    bounds = sorted({b for _, b in at_max})
    bound_str = bounds[0] if len(bounds) == 1 else "/".join(bounds)
    if len(at_max) == 1:
        return f"{at_max[0][0]} @ {max_cy:.2f} ({bound_str}-bound)"
    return f"{len(at_max)} nodes @ {max_cy:.2f} ({bound_str}-bound)"


def _stats_footer(estimate: CycleEstimate) -> None:
    """Bound summary table + program/active/bottleneck stats. Shared by both views."""
    rollup = _per_node_rollup(estimate)
    active = {n: v for n, v in rollup.items() if v[2] > 0.0}

    # Bound summary table (active nodes only) — its own section.
    print(_SECT)
    print(f"{'Type':<10}{'Nodes':>8}{'Avg Cycles':>14}{'Max':>14}   Max node")
    print(_HDR)
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
        else:
            avg = max_cy = 0.0
            max_node = "-"
        print(f"{bound:<10}{count:>8}{avg:>14.2f}{max_cy:>14.2f}   {max_node}")

    # Summary — its own section.
    idle = estimate.total_nodes - estimate.active_nodes
    print(_SECT)
    print(f"Program cycles : {estimate.program_cycles:.2f}")
    print(
        f"Active nodes   : {estimate.active_nodes} / {estimate.total_nodes}  ({idle} idle)"
    )
    print(f"Bottleneck     : {_bottleneck(active)}")
    print(_FRAME)
    if sum(k.compute_cycles for k in estimate.kernels) == 0.0:
        print(
            "note: compute path is 0 — no compute_op events in this trace "
            "(sim instrumentation pending); movement-only estimate."
        )


def print_detailed(estimate: CycleEstimate) -> None:
    """Detailed per-kernel view — complete, includes zero rows."""
    _header(estimate, "Kernel")
    for ke in estimate.kernels:
        print(
            f"{ke.kernel:<28} {ke.compute_cycles:>12.2f} {ke.movement_cycles:>12.2f} "
            f"{ke.cycles:>12.2f}  {_short_bound(ke.bound)}"
        )
    _stats_footer(estimate)


def print_summary(estimate: CycleEstimate, include_zero: bool = False) -> None:
    """Per-node rollup (the default view).

    Each node's columns are the max over its kernels (concurrent RISCs), matching
    the program combiner.
    """
    rollup = _per_node_rollup(estimate)
    _header(estimate, "Node")
    for node in sorted(rollup, key=node_sort_key):
        compute, movement, cyc, bound = rollup[node]
        if not include_zero and cyc == 0.0:
            continue
        print(f"{node:<28} {compute:>12.2f} {movement:>12.2f} {cyc:>12.2f}  {bound}")
    _stats_footer(estimate)


def write_json(path: Path, estimate: CycleEstimate) -> None:
    """Serialize the full, self-describing estimate (for analysis reuse)."""
    payload = {
        "tool": _TOOL,
        "schema_version": _SCHEMA_VERSION,
        "model": "ideal-peak",
        "profile": estimate.profile,
        "program_cycles": estimate.program_cycles,
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
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed cycle report {p}: {exc}") from None
