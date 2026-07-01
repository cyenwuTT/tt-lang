# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hardware spec profiles for the v1.0 analytical peak cycle model.

Canonical, reviewed peak-rate profiles for known parts live here as typed
``HardwareProfile`` objects (the source of truth), looked up by name via
:func:`get_profile`. Custom/experimental profiles can be supplied as a JSON file
and loaded with :func:`load_profile_json`; :func:`resolve_profile` accepts either
a built-in name or a ``.json`` path, which is what the CLI ``--hw-profile`` uses.

The JSON mirrors the dataclass fields, with ``compute_rate`` encoded as a list of
``[op_type, dtype, rate]`` triples (JSON has no tuple keys). It is a thin
serialization of ``HardwareProfile`` and may evolve as the field set does.

NOTE: built-in **movement** rates are derived from tt-metal's measured NoC data
(cited inline). Built-in **compute** rates are still placeholders pending arch/ISA
throughput numbers and the compute_op instrumentation, so absolute estimates are
not yet hardware-validated end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import HardwareProfile

# Wormhole B0 (80 worker cores). Movement rates cited below; compute is a
# placeholder (see module docstring). Sources under third-party/tt-metal/:
#   noc_bw/noc_latency: impl/experimental/noc_estimator/latencies/noc_latencies.yaml
#     (baseline row) -> 64 KB / 2589 cyc = 25.3 B/cyc; 64 B floor = 293 cyc.
#   topology/clock:     soc_descriptors/wormhole_b0_80_arch.yaml; AICLK ~1 GHz.
WORMHOLE_B0 = HardwareProfile(
    name="wormhole_b0",
    # COMPUTE — PROVISIONAL placeholder rates (tiles/cycle), NOT hardware-validated;
    # keyed by op_type with dtype="" (any dtype). Fill real per-(op, dtype) rates
    # from arch/ISA docs later. Note the tile conventions differ and are not
    # comparable: eltwise/unary/reduce emit output tiles; matmul emits MAC-tile
    # volume (M*K*N).
    compute_rate={("matmul", ""): 4096.0},  # placeholder, MAC-tiles/cycle
    compute_rate_default=16.0,  # placeholder for eltwise / unary / reduce (tiles/cycle)
    # NoC bytes/cycle (large-transfer asymptote). One cited number shared across
    # localities for now; local_l1 is really faster — refine when sourced.
    noc_bw={"local_l1": 25.3, "remote_l1": 25.3, "dram": 25.3},
    # fixed per-transfer latency (cycles), from the small-transfer floor
    noc_latency={"local_l1": 293.0, "remote_l1": 293.0, "dram": 293.0},
    clock_ghz=1.0,  # nominal Wormhole AICLK
    bytes_per_tile=2048.0,  # bf16 32x32 tile = 2 B * 1024
    dm_engines=2,  # Tensix: NCRISC + BRISC
)

_PROFILES: dict[str, HardwareProfile] = {
    WORMHOLE_B0.name: WORMHOLE_B0,
}

DEFAULT = WORMHOLE_B0


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
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed hardware profile {p}: {exc}") from None


def resolve_profile(name_or_path: str) -> HardwareProfile:
    """Resolve a built-in profile name, or a path to a JSON profile file.

    A value ending in ``.json`` is loaded from file; otherwise it is looked up
    as a built-in name. Used by the CLI ``--hw-profile`` option.
    """
    if name_or_path.endswith(".json"):
        return load_profile_json(name_or_path)
    return get_profile(name_or_path)
