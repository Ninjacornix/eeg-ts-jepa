"""Electrode montages and 3D positions.

Hardware-agnostic by design: a montage is just a list of electrode *names*.
Each name maps to (a) a stable integer identity (index into a learned table)
and (b) an approximate 3D position on the unit sphere (10-20 layout).  The
model consumes arbitrary subsets / orderings of these, so different devices
and channel counts share one representation space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Approximate 10-20 spherical coordinates (azimuth deg, elevation deg).
# Not anatomically exact -- enough to give the spatial branch a consistent,
# device-independent geometry.  Azimuth: 0 = nose (front), +90 = left ear.
_TEN_TWENTY_AZ_EL: dict[str, tuple[float, float]] = {
    "Fp1": (-18, 60), "Fp2": (18, 60),
    "AF3": (-30, 50), "AF4": (30, 50),
    "F7": (-54, 40), "F3": (-39, 45), "Fz": (0, 50), "F4": (39, 45), "F8": (54, 40),
    "FC5": (-60, 22), "FC1": (-22, 30), "FC2": (22, 30), "FC6": (60, 22),
    "T7": (-90, 0), "C3": (-45, 0), "Cz": (0, 90), "C4": (45, 0), "T8": (90, 0),
    "CP5": (-120, 22), "CP1": (-22, -30), "CP2": (22, -30), "CP6": (120, 22),
    "P7": (-126, 40), "P3": (-39, -45), "Pz": (180, 50), "P4": (39, -45), "P8": (126, 40),
    "PO3": (-150, 50), "PO4": (150, 50),
    "O1": (-162, 60), "Oz": (180, 65), "O2": (162, 60),
}

# Stable identity index per electrode name -> shared across all montages/devices.
ELECTRODE_VOCAB: list[str] = sorted(_TEN_TWENTY_AZ_EL.keys())
ELECTRODE_TO_ID: dict[str, int] = {n: i for i, n in enumerate(ELECTRODE_VOCAB)}


def _sphere_xyz(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    az, el = math.radians(az_deg), math.radians(el_deg)
    r = math.cos(el)
    return (r * math.cos(az), r * math.sin(az), math.sin(el))


def channel_positions(names: list[str]) -> np.ndarray:
    """Return (C, 3) unit-sphere coords for the given electrode names."""
    out = np.zeros((len(names), 3), dtype=np.float32)
    for i, n in enumerate(names):
        if n not in _TEN_TWENTY_AZ_EL:
            raise KeyError(f"unknown electrode {n!r}; add it to montage.py")
        out[i] = _sphere_xyz(*_TEN_TWENTY_AZ_EL[n])
    return out


def channel_ids(names: list[str]) -> np.ndarray:
    """Return (C,) stable identity indices for the given electrode names."""
    return np.asarray([ELECTRODE_TO_ID[n] for n in names], dtype=np.int64)


@dataclass(frozen=True)
class Montage:
    name: str
    channels: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        return len(self.channels)


# A few stand-in device montages with different channel counts / layouts, used
# to simulate cross-site / cross-device heterogeneity during pretraining.
MONTAGES: dict[str, Montage] = {
    "clinical_19": Montage("clinical_19", (
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
    )),
    "consumer_8": Montage("consumer_8", (
        "AF3", "AF4", "F3", "F4", "C3", "C4", "O1", "O2",
    )),
    "highdensity_31": Montage("highdensity_31", tuple(ELECTRODE_VOCAB)),
    "frontal_6": Montage("frontal_6", (
        "Fp1", "Fp2", "AF3", "AF4", "F3", "F4",
    )),
}
