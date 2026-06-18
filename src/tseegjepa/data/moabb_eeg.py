"""MOABB -> EEGJepa dataset adapter.

Wraps a (small) MOABB motor-imagery dataset into the batch schema the model
expects.  One trial = one sample.  `domain` = subject id (the cross-site axis
for leave-one-dataset-out); `subgroup` = session (for disaggregated metrics).

Channels are mapped to the global standard_1020 registry, so a model pretrained
here accepts any other montage without changes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import electrodes


def _quiet():
    """Silence MNE/MOABB/urllib3 chatter so only training progress prints."""
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    for name in ("moabb", "mne", "urllib3", "pooch"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        import mne
        mne.set_log_level("ERROR")
    except Exception:
        pass
    try:
        import moabb
        moabb.set_log_level("ERROR")
    except Exception:
        pass


def load_moabb(
    dataset_name: str = "BNCI2014_001",
    subjects: list[int] | None = None,
    paradigm_name: str = "MotorImagery",
    n_classes: int | None = None,
    sample_rate: int = 200,
    tmax: float | None = None,
):
    """Return (X, y, meta, ch_names, label_names).

    X: float32 (n_trials, C, T) resampled to `sample_rate`.
    y: int64 labels; meta: list of dicts with subject/session.
    """
    _quiet()
    import moabb.datasets as ds_mod
    from moabb.paradigms import LeftRightImagery, MotorImagery

    dataset = getattr(ds_mod, dataset_name)()
    if subjects is not None:
        dataset.subject_list = subjects

    if paradigm_name == "LeftRightImagery":
        paradigm = LeftRightImagery(resample=sample_rate)
    else:
        kw = {"resample": sample_rate}
        if n_classes is not None:
            kw["n_classes"] = n_classes
        paradigm = MotorImagery(**kw)

    # return_epochs -> reliable channel names + info across MOABB versions
    epochs, labels, meta = paradigm.get_data(
        dataset=dataset, subjects=dataset.subject_list, return_epochs=True
    )
    ch_names = list(epochs.ch_names)
    X = epochs.get_data(copy=False).astype(np.float32)     # (n_trials, C, T)

    label_names = sorted(set(labels))
    lut = {n: i for i, n in enumerate(label_names)}
    y = np.asarray([lut[l] for l in labels], dtype=np.int64)

    if tmax is not None:
        X = X[..., : int(tmax * sample_rate)]

    # keep only channels present in the standard_1020 registry
    keep = np.asarray(electrodes.known(ch_names), dtype=bool)
    if not keep.all():
        X = X[:, keep, :]
        ch_names = [n for n, k in zip(ch_names, keep) if k]

    demo = subject_demographics(dataset, sorted({int(r["subject"]) for _, r in meta.iterrows()}))
    meta_list = [
        {"subject": int(r["subject"]),
         "session": str(r["session"]),
         **demo.get(int(r["subject"]), {})}
        for _, r in meta.iterrows()
    ]
    return X, y, meta_list, ch_names, label_names


def subject_demographics(dataset, subjects: list[int]) -> dict[int, dict]:
    """Pull per-subject demographics.

    Returns {subject: {"sex": "M"/"F"/"U", "age": int|None, "hand": str}}.
    Two sources:
      1. a rich subject-info TABLE (e.g. Dreyer2023.get_subject_info(): columns
         SUJ_gender, Birth_year, ...) -> preferred, no EEG download needed;
      2. else MNE raw.info['subject_info'] per subject (e.g. BNCI2014_001).
    Empty dict per subject if neither carries info (e.g. PhysionetMI).
    """
    _quiet()
    table = _demographics_from_table(dataset, subjects)
    if table is not None:
        return table

    sex_map = {0: "U", 1: "M", 2: "F"}
    hand_map = {1: "R", 2: "L", 3: "A"}
    out: dict[int, dict] = {}
    for s in subjects:
        try:
            rd = dataset.get_data([s])
            raw = next(iter(next(iter(rd[s].values())).values()))
            si = raw.info.get("subject_info") or {}
            age = None
            bday, meas = si.get("birthday"), raw.info.get("meas_date")
            if bday is not None and meas is not None:
                age = meas.year - bday.year
            out[s] = {
                "sex": sex_map.get(si.get("sex", 0), "U"),
                "age": age,
                "hand": hand_map.get(si.get("hand", 0), "U"),
            }
        except Exception:
            out[s] = {}
    return out


def _demographics_from_table(dataset, subjects, ref_year: int = 2019):
    """Parse a Dreyer2023-style subject-info table; None if unavailable.

    Assumes the table is ordered by global subject id (row i -> subject i+1),
    which matches MOABB's Dreyer2023 subject_list (1..87). gender: 1->M, 2->F.
    age = ref_year - Birth_year (ref only affects absolute age, not group order).
    """
    if not hasattr(dataset, "get_subject_info"):
        return None
    try:
        df = dataset.get_subject_info().reset_index(drop=True)
    except Exception:
        return None
    if "SUJ_gender" not in df.columns and "Birth_year" not in df.columns:
        return None
    import math
    sex_map = {1: "M", 2: "F"}
    out: dict[int, dict] = {}
    for s in subjects:
        if s - 1 >= len(df):
            out[s] = {}
            continue
        r = df.iloc[s - 1]
        by = r.get("Birth_year")
        age = None
        if by is not None and not (isinstance(by, float) and math.isnan(by)):
            age = ref_year - int(by)
        g = r.get("SUJ_gender")
        sex = sex_map.get(int(g), "U") if g is not None and not (
            isinstance(g, float) and math.isnan(g)) else "U"
        out[s] = {"sex": sex, "age": age, "hand": "U"}
    return out


class MoabbEEGDataset(Dataset):
    """norm modes:
      "global": one mean/std over ALL channels+time per trial. PRESERVES relative
                band-power across channels -> keeps the spatial MI signal. (default)
      "perchan": per-channel z-score. Robust to gain but ERASES inter-channel
                 power differences -> destroys lateralization (bad for MI).
      "none": raw signal.
    """

    def __init__(self, X, y, meta, ch_names, norm: str = "global",
                 subgroup_by: str = "sex", crop_t: int | None = None,
                 crop_align: str = "end", standardize: bool | None = None):
        self.X = X
        self.y = y
        self.meta = meta
        self.ch_names = ch_names
        self.ch_ids = torch.from_numpy(electrodes.channel_ids(ch_names))
        self.ch_pos = torch.from_numpy(electrodes.channel_positions(ch_names))
        # back-compat: standardize=False -> "none"
        if standardize is False:
            norm = "none"
        self.norm = norm
        self.subgroup_by = subgroup_by
        # crop every trial to crop_t samples -> uniform T for multi-dataset batches.
        # crop_align: 'end' = keep the LAST crop_t samples (sustained MI, past the
        # cue-onset transient -> aligns datasets with different cue offsets, e.g.
        # IV-2a [2,6] vs Dreyer [0,5]); 'start' = first; 'center' = middle.
        self.crop_t = crop_t
        self.crop_align = crop_align
        subs = sorted({m["subject"] for m in meta})
        self._sub_lut = {s: i for i, s in enumerate(subs)}
        self._build_subgroups()

    def _build_subgroups(self):
        """Map each trial to an integer subgroup id for disaggregated metrics.

        'session' = recording session; 'sex' = M/F/U; 'age' = below/above median.
        subgroup_names maps id -> human label.
        """
        field = self.subgroup_by
        if field == "age":
            ages = [m["age"] for m in self.meta if m.get("age") is not None]
            med = float(np.median(ages)) if ages else 0.0
            def key(m):
                a = m.get("age")
                if a is None:
                    return "age?"
                return f"age<={med:.0f}" if a <= med else f"age>{med:.0f}"
        elif field == "sex":
            def key(m): return f"sex={m.get('sex', 'U')}"
        else:  # session
            field = "session"
            def key(m): return f"sess={m.get('session', '?')}"
        labels = [key(m) for m in self.meta]
        uniq = sorted(set(labels))
        lut = {name: i for i, name in enumerate(uniq)}
        self._subgroups = [lut[name] for name in labels]
        self.subgroup_names = {i: name for name, i in lut.items()}

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])                  # (C, T)
        if self.crop_t is not None and x.shape[1] > self.crop_t:
            if self.crop_align == "end":
                x = x[:, -self.crop_t:]
            elif self.crop_align == "center":
                s = (x.shape[1] - self.crop_t) // 2
                x = x[:, s:s + self.crop_t]
            else:                                          # "start"
                x = x[:, : self.crop_t]
        if self.norm == "global":
            x = (x - x.mean()) / (x.std() + 1e-8)          # keep inter-channel power
        elif self.norm == "perchan":
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        m = self.meta[idx]
        return {
            "signal": x,
            "ch_ids": self.ch_ids,
            "ch_pos": self.ch_pos,
            "label": int(self.y[idx]),
            "subgroup": self._subgroups[idx],
            "domain": self._sub_lut[m["subject"]],
        }

    def subset_by_subject(self, subjects: list[int]) -> "MoabbEEGDataset":
        want = set(subjects)
        idx = [i for i, m in enumerate(self.meta) if m["subject"] in want]
        return MoabbEEGDataset(self.X[idx], self.y[idx],
                               [self.meta[i] for i in idx],
                               self.ch_names, norm=self.norm,
                               subgroup_by=self.subgroup_by, crop_t=self.crop_t,
                               crop_align=self.crop_align)

    def with_crop(self, crop_t: int | None, crop_align: str = "end") -> "MoabbEEGDataset":
        return MoabbEEGDataset(self.X, self.y, self.meta, self.ch_names,
                               norm=self.norm, subgroup_by=self.subgroup_by,
                               crop_t=crop_t, crop_align=crop_align)
