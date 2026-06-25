# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Trace parsing and kernel feature extraction for cycle estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .types import KernelFeatures, TraceEvent
from ..utils import as_int, iter_events, node_from_kernel


@dataclass
class _OpenKernelState:
    start_tick: int
    block_start_tick: int | None = None


def parse_trace(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for obj in iter_events(path):
        tick = as_int(obj.get("tick", 0))
        event = str(obj.get("event", ""))
        kernel_obj = obj.get("kernel")
        kernel = str(kernel_obj) if kernel_obj is not None else None
        data = {
            k: v
            for k, v in obj.items()
            if k not in {"tick", "event", "kernel", "node"}
        }
        
        if "node" in obj:
            data["node"] = obj["node"]

        events.append(TraceEvent(tick=tick, event=event, kernel=kernel, data=data))
    
    return events


def extract_kernel_features(events: list[TraceEvent]) -> dict[str, KernelFeatures]:
    features: dict[str, KernelFeatures] = {}
    # Track open kernel intervals to accumulate measured/blocked time.
    open_states: dict[str, list[_OpenKernelState]] = {}
    # Track phase-duration signals: unpaired DFB and copy events.
    dfb_wait_starts: dict[str, int] = {}
    dfb_reserve_starts: dict[str, int] = {}
    copy_starts: dict[str, int] = {}
    max_tick = 0

    for ev in events:
        max_tick = max(max_tick, ev.tick)
        kernel = ev.kernel
        
        if not kernel:
            continue

        feature = features.get(kernel)
        
        if feature is None:
            node = node_from_kernel(kernel)
            suffix = kernel.removeprefix(f"{node}-") if node != kernel else ""
            role = suffix if suffix in {"compute", "read", "write"} else "other"
            
            try:
                node_idx = int(node.replace("node", ""))
            except (ValueError, AttributeError):
                node_idx = 0
            
            feature = KernelFeatures(kernel=kernel, role=role, node_index=node_idx)
            features[kernel] = feature

        if ev.event == "kernel_start":
            open_states.setdefault(kernel, []).append(_OpenKernelState(start_tick=ev.tick))
            continue

        if ev.event == "kernel_end":
            stack = open_states.get(kernel, [])
            if stack:
                st = stack.pop()
                if st.block_start_tick is not None:
                    feature.blocked_cycles += max(0, ev.tick - st.block_start_tick)
                feature.measured_cycles += max(0, ev.tick - st.start_tick)
            continue

        if ev.event == "kernel_block":
            stack = open_states.get(kernel, [])
            if stack and stack[-1].block_start_tick is None:
                stack[-1].block_start_tick = ev.tick
            continue

        if ev.event == "kernel_unblock":
            stack = open_states.get(kernel, [])
            if stack and stack[-1].block_start_tick is not None:
                feature.blocked_cycles += max(0, ev.tick - stack[-1].block_start_tick)
                stack[-1].block_start_tick = None
            continue

        if ev.event == "dfb_wait_begin":
            feature.wait_count += 1
            dfb_wait_starts[kernel] = ev.tick
            continue

        if ev.event == "dfb_wait_end":
            feature.wait_tiles += as_int(ev.data.get("tiles", 0))
            if kernel in dfb_wait_starts:
                feature.dfb_wait_block_cycles += max(0, ev.tick - dfb_wait_starts[kernel])
                del dfb_wait_starts[kernel]
            continue

        if ev.event == "dfb_reserve_begin":
            feature.reserve_count += 1
            dfb_reserve_starts[kernel] = ev.tick
            continue

        if ev.event == "dfb_reserve_end":
            feature.reserve_tiles += as_int(ev.data.get("tiles", 0))
            if kernel in dfb_reserve_starts:
                feature.dfb_reserve_block_cycles += max(0, ev.tick - dfb_reserve_starts[kernel])
                del dfb_reserve_starts[kernel]
            continue

        if ev.event == "dfb_push":
            feature.push_count += 1
            continue

        if ev.event == "dfb_pop":
            feature.pop_count += 1
            continue

        if ev.event == "copy_start":
            copy_starts[kernel] = ev.tick
            continue

        if ev.event == "copy_end":
            feature.copy_calls += 1
            tiles = as_int(ev.data.get("tiles", 0))
            local_l1 = as_int(ev.data.get("local_l1", 0))
            remote_l1 = as_int(ev.data.get("remote_l1", 0))
            dram = as_int(ev.data.get("dram", 0))
            feature.local_l1_tiles += local_l1
            feature.remote_l1_tiles += remote_l1
            feature.dram_tiles += dram
            feature.copy_tiles += tiles
            if kernel in copy_starts:
                feature.copy_duration_cycles += max(0, ev.tick - copy_starts[kernel])
                del copy_starts[kernel]
            continue

    # Close any intervals still open at trace end.
    for kernel, stack in open_states.items():
        feature = features.get(kernel)
        assert feature is not None
        
        for st in stack:
            if st.block_start_tick is not None:
                feature.blocked_cycles += max(0, max_tick - st.block_start_tick)
            feature.measured_cycles += max(0, max_tick - st.start_tick)

    # active_cycles = measured - blocked.
    for feature in features.values():
        feature.active_cycles = max(0, feature.measured_cycles - feature.blocked_cycles)

    return features
