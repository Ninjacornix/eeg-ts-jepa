"""Global electrode registry from MNE's standard_1020 montage.

Real datasets (MOABB) use many electrode names with real scalp coordinates.
This builds a single device-agnostic vocabulary (stable id per electrode) and
unit-sphere positions, so any montage maps into the same id/position space the
model's identity/position embeddings consume.  Matching is case-insensitive.
"""

from __future__ import annotations

import functools
import re

import numpy as np


def _clean(name: str) -> str:
    """Normalize an electrode name for matching.

    Handles dataset quirks: PhysioNet pads names with dots ('Fc5.', 'Cz..'),
    others use 'T3'/'T4'/'T5'/'T6' (old 10-20) for T7/T8/P7/P8. Uppercase,
    strip non-alphanumerics, then remap deprecated labels.
    """
    k = re.sub(r"[^A-Za-z0-9]", "", name).upper()
    return {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(k, k)


@functools.lru_cache(maxsize=1)
def _registry():
    import mne

    m = mne.channels.make_standard_montage("standard_1020")
    pos = m.get_positions()["ch_pos"]            # name -> xyz (meters)
    names = list(m.ch_names)
    name_to_id = {_clean(n): i for i, n in enumerate(names)}
    xyz = np.zeros((len(names), 3), dtype=np.float32)
    for i, n in enumerate(names):
        v = np.asarray(pos[n], dtype=np.float32)
        nrm = np.linalg.norm(v)
        xyz[i] = v / nrm if nrm > 1e-8 else v
    return names, name_to_id, xyz


def vocab_size() -> int:
    return len(_registry()[0])


def unknown_id() -> int:
    """Stable identity used when a name is outside the standard registry."""
    return vocab_size()


def channel_ids(names: list[str], strict: bool = True) -> np.ndarray:
    _, name_to_id, _ = _registry()
    out = []
    for n in names:
        key = _clean(n)
        if key not in name_to_id:
            if strict:
                raise KeyError(f"electrode {n!r} (->{key}) not in standard_1020 registry")
            out.append(unknown_id())
        else:
            out.append(name_to_id[key])
    return np.asarray(out, dtype=np.int64)


def channel_positions(names: list[str], strict: bool = True) -> np.ndarray:
    _, name_to_id, xyz = _registry()
    out = []
    for n in names:
        key = _clean(n)
        if key not in name_to_id:
            if strict:
                raise KeyError(f"electrode {n!r} (->{key}) not in standard_1020 registry")
            out.append(np.zeros(3, dtype=np.float32))
        else:
            out.append(xyz[name_to_id[key]])
    return np.stack(out).astype(np.float32)


def known(names: list[str]) -> list[bool]:
    _, name_to_id, _ = _registry()
    return [_clean(n) in name_to_id for n in names]
