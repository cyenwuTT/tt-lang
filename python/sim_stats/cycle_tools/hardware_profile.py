# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hardware spec profiles for the v1.0 analytical peak cycle model.

Canonical, reviewed peak-rate profiles for known parts live here as typed
``HardwareProfile`` objects (the source of truth), looked up by name.

A file-based loader (e.g. JSON) for custom/experimental profiles is intentionally
deferred: the ``HardwareProfile`` fields are still in flux pending the target-part
decision, and the only consumer (sensitivity sweeps) comes later — designing a
serialization format now would lock it before the requirements are known.

NOTE: the rates below are PROVISIONAL placeholders pending the target-part
decision (Wormhole / Blackhole). They are not yet hardware-validated and must
not be trusted for real estimates.
"""

from __future__ import annotations

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
