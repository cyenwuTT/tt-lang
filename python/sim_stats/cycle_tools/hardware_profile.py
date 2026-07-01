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

NOTE: the built-in rates below are PROVISIONAL placeholders pending the
target-part decision (Wormhole / Blackhole). They are not yet hardware-validated
and must not be trusted for real estimates.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import HardwareProfile

# Provisional placeholder profile — numbers are NOT hardware-validated yet.
WORMHOLE_B0 = HardwareProfile(
    name="wormhole_b0",
    compute_rate={},  # (op_type, dtype) -> tiles/cycle; filled once specs land
    compute_rate_default=1.0,  # placeholder
    noc_bw={"local_l1": 1.0, "remote_l1": 1.0, "dram": 1.0},  # placeholder
    noc_latency={"local_l1": 0.0, "remote_l1": 0.0, "dram": 0.0},  # placeholder
    clock_ghz=1.0,  # placeholder
    bytes_per_tile=2048.0,  # placeholder (bf16 tile = 32*32*2 B)
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
