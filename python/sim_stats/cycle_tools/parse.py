# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Trace parsing and per-kernel work extraction for cycle estimation."""

from __future__ import annotations

from pathlib import Path

from .types import KernelWork, OpWork, TraceEvent
from ..utils import as_int, iter_events, node_from_kernel


def parse_trace(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for obj in iter_events(path):
        tick = as_int(obj.get("tick", 0))
        event = str(obj.get("event", ""))
        kernel_obj = obj.get("kernel")
        kernel = str(kernel_obj) if kernel_obj is not None else None
        data = {
            k: v for k, v in obj.items() if k not in {"tick", "event", "kernel", "node"}
        }

        if "node" in obj:
            data["node"] = obj["node"]

        events.append(TraceEvent(tick=tick, event=event, kernel=kernel, data=data))

    return events


def extract_kernel_work(events: list[TraceEvent]) -> dict[str, KernelWork]:
    """Build per-kernel work records from trace events.

    Reads two event kinds:
      - ``copy_end``   -> movement OpWork, one per non-zero locality tile count.
      - ``compute_op`` -> compute OpWork (op_type, dtype, tiles).

    ``compute_op`` events are emitted by the simulator once per math op. When a
    trace lacks them (e.g. generated before the instrumentation), the compute path
    is simply empty and the estimate is movement-only.
    """
    work: dict[str, KernelWork] = {}

    for ev in events:
        kernel = ev.kernel
        if not kernel:
            continue

        kw = work.get(kernel)
        if kw is None:
            node = node_from_kernel(kernel)
            try:
                node_idx = int(node.replace("node", ""))
            except (ValueError, AttributeError):
                node_idx = 0
            kw = KernelWork(kernel=kernel, node_index=node_idx)
            work[kernel] = kw

        if ev.event == "compute_op":
            tiles = as_int(ev.data.get("tiles", 0))
            if tiles > 0:
                kw.ops.append(
                    OpWork(
                        kind="compute",
                        op_type=str(ev.data.get("op_type", "")),
                        dtype=str(ev.data.get("dtype", "")),
                        tiles=tiles,
                    )
                )
        elif ev.event == "copy_end":
            # One movement op per locality with a non-zero tile count.
            for locality in ("local_l1", "remote_l1", "dram"):
                tiles = as_int(ev.data.get(locality, 0))
                if tiles > 0:
                    kw.ops.append(
                        OpWork(
                            kind="movement",
                            op_type="copy",
                            tiles=tiles,
                            locality=locality,
                        )
                    )

    return work
