# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cycle math, per-node rollup, profile resolution, and estimate assembly.

Turns per-kernel work into cycles and combines them into a ``CycleEstimate``:

- :func:`op_cycles` — one op's ideal-peak cycles (work / peak-rate).
- :func:`kernel_cycles` — per-kernel ``max(compute, movement)`` (concurrent engines).
- :func:`program_cycles` — throughput-bound ``max`` within a node and across nodes,
  floored by the shared aggregate-DRAM ceiling (:func:`program_breakdown`).
- :func:`per_node_rollup` — the single per-node aggregation (max over a node).
- :func:`build_estimate` — assemble the canonical ``CycleEstimate``.

Profile resolution (:func:`resolve_profile` / :func:`load_profile_json` /
:func:`get_profile`) also lives here; the profile *data* is in :mod:`types`.

The dependency-DAG latency regime (fill/drain, cross-node serialization) is out of
scope; see docs/development/CycleEstimator.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import (
    DEFAULT,
    _PROFILES,
    CycleEstimate,
    HardwareProfile,
    KernelEstimate,
    KernelWork,
    NodeEstimate,
    OpWork,
)
from ..utils import node_from_kernel, node_sort_key, role_from_kernel


# ---------------------------------------------------------------------------
# Cycle math
# ---------------------------------------------------------------------------


def op_cycles(op: OpWork, hw: HardwareProfile) -> float:
    """Ideal-peak cycles for a single op: work / peak-rate."""
    if op.kind == "compute":
        rate = hw.rate_for(op.op_type, op.dtype)
        return op.tiles / rate if rate > 0.0 else 0.0
    if op.kind == "movement":
        bw = hw.bandwidth_for(op.locality)
        moved_bytes = op.tiles * hw.bytes_per_tile
        transfer = moved_bytes / bw if bw > 0.0 else 0.0
        return hw.latency_for(op.locality) + transfer
    return 0.0


def kernel_paths(work: KernelWork, hw: HardwareProfile) -> tuple[float, float]:
    """Return the (compute_path, movement_path) cycle totals for a kernel."""
    compute_path = sum(op_cycles(o, hw) for o in work.ops if o.kind == "compute")
    movement_path = sum(op_cycles(o, hw) for o in work.ops if o.kind == "movement")
    return compute_path, movement_path


def total_dram_bytes(kernels: list[KernelWork], hw: HardwareProfile) -> float:
    """Program-wide bytes that hit the shared GDDR6 pool.

    Only ``locality == "dram"`` movement counts: local-L1 and remote-L1
    (multicast) traffic never touches the DRAM controller and must be excluded
    from the aggregate ceiling.
    """
    return sum(
        o.tiles * hw.bytes_per_tile
        for k in kernels
        for o in k.ops
        if o.kind == "movement" and o.locality == "dram"
    )


def dram_bytes_by_direction(
    kernels: list[KernelWork], hw: HardwareProfile
) -> tuple[float, float]:
    """Split the DRAM-locality movement bytes into (read, write).

    Direction comes from the trace's ``copy_end`` ``direction`` field. Traffic with
    no direction (older traces) falls into read, so read + write always equals
    :func:`total_dram_bytes`. Reporting/validation only — the ceiling is unsplit.
    """
    read = write = 0.0
    for k in kernels:
        for o in k.ops:
            if o.kind == "movement" and o.locality == "dram":
                b = o.tiles * hw.bytes_per_tile
                if o.direction == "write":
                    write += b
                else:
                    read += b
    return read, write


def kernel_cycles(work: KernelWork, hw: HardwareProfile) -> float:
    """Ideal-peak kernel cycles: the larger of the compute and movement paths.

    The compute engine and the data-movement engine run concurrently, so the
    kernel time is ``max`` of the two serial paths, not their sum.
    """
    compute_path, movement_path = kernel_paths(work, hw)
    return max(compute_path, movement_path)


def program_cycles(kernels: list[KernelWork], hw: HardwareProfile) -> float:
    """Program-level cycles under the ideal-peak, throughput-bound model.

    Two levels of overlap:
      - within a node: the reader / compute / writer kernels run on that core's
        concurrent RISCs, so the node's time is the ``max`` of its kernels.
      - across nodes: distinct nodes are separate cores running in parallel, so
        the program time is the ``max`` over nodes.

    A third bound sits above the two overlaps: the shared GDDR6 pool, taken as a
    ``max`` with the per-node bound (see :func:`program_breakdown`). Rationale
    and the deferred latency regime: docs/development/CycleEstimator.md.
    """
    return program_breakdown(kernels, hw)[0]


def program_breakdown(
    kernels: list[KernelWork], hw: HardwareProfile
) -> tuple[float, str, float, float]:
    """Program cycles plus which bound set them.

    Returns ``(program_cycles, program_bound, dram_floor, node_bound)`` where
    ``program_bound`` is ``"aggregate-dram"`` if the shared DRAM floor dominates,
    else ``"per-node"``. See :func:`program_cycles` for the model rationale.
    """
    per_node: dict[str, float] = {}
    for k in kernels:
        node = node_from_kernel(k.kernel)
        per_node[node] = max(per_node.get(node, 0.0), kernel_cycles(k, hw))
    node_bound = max(per_node.values(), default=0.0)
    return program_from_node_bound(kernels, hw, node_bound)


def program_from_node_bound(
    kernels: list[KernelWork], hw: HardwareProfile, node_bound: float
) -> tuple[float, str, float, float]:
    """Select the program bound given an already-computed per-node ``node_bound``.

    Takes the ``max`` of the per-node throughput bound and the shared aggregate
    DRAM floor. Callers that have already rolled up per-node cycles (e.g.
    :func:`build_estimate`) pass ``node_bound`` here to avoid re-walking the
    kernels; :func:`program_breakdown` computes it and delegates.
    """
    agg_bw = hw.aggregate_dram_bandwidth()
    dram_floor = total_dram_bytes(kernels, hw) / agg_bw if agg_bw > 0.0 else 0.0

    if dram_floor > node_bound:
        return dram_floor, "aggregate-dram", dram_floor, node_bound
    return node_bound, "per-node", dram_floor, node_bound


# ---------------------------------------------------------------------------
# Per-node rollup + estimate assembly
# ---------------------------------------------------------------------------


def per_node_rollup(kernel_estimates: list[KernelEstimate]) -> list[NodeEstimate]:
    """Group kernel estimates by node, taking the ``max`` per column.

    The reader / compute / writer kernels share one core's concurrent RISCs, so a
    node's time is the ``max`` over its kernels. This is the single place per-node
    cycles are computed; both program selection and rendering read it.
    """
    agg: dict[str, tuple[float, float, float]] = {}
    for ke in kernel_estimates:
        c, m, cy = agg.get(ke.node, (0.0, 0.0, 0.0))
        agg[ke.node] = (
            max(c, ke.compute_cycles),
            max(m, ke.movement_cycles),
            max(cy, ke.cycles),
        )
    return [
        NodeEstimate(
            node=node,
            compute=c,
            movement=m,
            cycles=cy,
            bound="compute" if c > m else "memory",
        )
        for node, (c, m, cy) in agg.items()
    ]


def _pipeline_items(kernels: list[KernelWork]) -> dict[str, int]:
    """Pipeline-item count N per node: movement ops in its write-role kernel.

    Each write op is one output block flowing through the pipeline. Nodes with no
    write kernel get N=1 (no pipelining -> serial stages).
    """
    items: dict[str, int] = {}
    for kw in kernels:
        if role_from_kernel(kw.kernel) == "write":
            n = sum(1 for o in kw.ops if o.kind == "movement")
            node = node_from_kernel(kw.kernel)
            items[node] = items.get(node, 0) + n
    return items


def per_node_fill_drain_bound(
    kernel_estimates: list[KernelEstimate], kernels: list[KernelWork]
) -> float:
    """Tier-1 per-node bound including crude pipeline fill/drain.

    For each node, treat its kernels as pipeline stages with cycles ``C_i`` and let
    ``N`` be the node's pipeline-item count. The standard pipeline time is
    ``max_i(C_i) + (sum_i C_i - max_i C_i) / N``: large N recovers the throughput
    bound ``max_i C_i``; N=1 gives the serial sum. Returns the max over nodes.

    Crude approximation (assumes a read/compute/write stage structure by role); the
    rigorous CB-DAG fill/drain is deferred. See docs/development/CycleEstimator.md.
    """
    items = _pipeline_items(kernels)
    by_node: dict[str, list[float]] = {}
    for ke in kernel_estimates:
        by_node.setdefault(ke.node, []).append(ke.cycles)

    fd_bound = 0.0
    for node, cyc in by_node.items():
        stage_max = max(cyc, default=0.0)
        n = max(1, items.get(node, 1))
        node_time = stage_max + (sum(cyc) - stage_max) / n
        fd_bound = max(fd_bound, node_time)
    return fd_bound


def build_estimate(kernels: list[KernelWork], hw: HardwareProfile) -> CycleEstimate:
    """Assemble the canonical CycleEstimate from per-kernel work + a profile."""
    kernel_estimates: list[KernelEstimate] = []
    for kw in sorted(kernels, key=lambda k: k.kernel):
        compute, movement = kernel_paths(kw, hw)
        kernel_estimates.append(
            KernelEstimate(
                kernel=kw.kernel,
                node=node_from_kernel(kw.kernel),
                role=role_from_kernel(kw.kernel),
                compute_cycles=compute,
                movement_cycles=movement,
                cycles=max(compute, movement),
                bound="compute-bound" if compute > movement else "memory-bound",
            )
        )

    # One per-node rollup, reused for program selection and rendering — no second
    # walk of the ops through the cycle math.
    nodes = per_node_rollup(kernel_estimates)
    node_bound = max((n.cycles for n in nodes), default=0.0)
    at_max = sorted(
        (n for n in nodes if n.cycles == node_bound and n.cycles > 0.0),
        key=lambda n: node_sort_key(n.node),
    )
    node_bound_reason = at_max[0].bound if at_max else "-"

    # Tier-1 pipeline fill/drain feeds the per-node path; dram_floor is unchanged.
    fd_bound = per_node_fill_drain_bound(kernel_estimates, kernels)
    node_fill_drain = fd_bound - node_bound

    prog_cycles, program_bound, dram_floor, _fd = program_from_node_bound(
        kernels, hw, fd_bound
    )

    dram_read, dram_write = dram_bytes_by_direction(kernels, hw)

    return CycleEstimate(
        profile_name=hw.name,
        profile=hw.summary(),
        program_cycles=prog_cycles,
        total_nodes=len(nodes),
        active_nodes=sum(1 for n in nodes if n.cycles > 0.0),
        kernels=kernel_estimates,
        program_bound=program_bound,
        dram_floor=dram_floor,
        total_dram_bytes=dram_read + dram_write,
        dram_read_bytes=dram_read,
        dram_write_bytes=dram_write,
        nodes=nodes,
        node_bound=node_bound,
        node_bound_reason=node_bound_reason,
        node_fill_drain=node_fill_drain,
    )


# ---------------------------------------------------------------------------
# Hardware profile resolution (profile data lives in types.py)
# ---------------------------------------------------------------------------


def get_profile(name: str) -> HardwareProfile:
    """Return a built-in profile by name, or raise with the known names."""
    try:
        return _PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(_PROFILES))
        raise KeyError(f"unknown hardware profile {name!r}; known: {known}") from None


def load_profile_json(path: Path | str) -> HardwareProfile:
    """Load a custom HardwareProfile from a JSON file.

    JSON keys mirror the dataclass fields; ``compute_rate`` is a list of
    ``[op_type, dtype, rate]`` triples. Raises FileNotFoundError / ValueError
    with the file path on a missing or malformed profile.
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"hardware profile file not found: {p}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in hardware profile {p}: {exc}") from None

    try:
        compute_rate = {
            (str(op), str(dt)): float(rate)
            for op, dt, rate in data.get("compute_rate", [])
        }
        return HardwareProfile(
            name=str(data.get("name", p.stem)),
            compute_rate=compute_rate,
            compute_rate_default=float(data["compute_rate_default"]),
            noc_bw={str(k): float(v) for k, v in data["noc_bw"].items()},
            noc_latency={
                str(k): float(v) for k, v in data.get("noc_latency", {}).items()
            },
            clock_ghz=float(data["clock_ghz"]),
            bytes_per_tile=float(data["bytes_per_tile"]),
            dm_engines=int(data.get("dm_engines", 1)),
            dram_aggregate_bw=float(data.get("dram_aggregate_bw", 0.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed hardware profile {p}: {exc}") from None


def resolve_profile(name_or_path: str | None) -> HardwareProfile:
    """Resolve a built-in profile name, a path to a JSON profile file, or None.

    ``None`` (or empty) yields :data:`types.DEFAULT`; a value ending in ``.json``
    is loaded from file; otherwise it is looked up as a built-in name. Used by the
    CLI ``--hw-profile`` option and the ``tt-lang-sim --cycles`` path.
    """
    if not name_or_path:
        return DEFAULT
    if name_or_path.endswith(".json"):
        return load_profile_json(name_or_path)
    return get_profile(name_or_path)
