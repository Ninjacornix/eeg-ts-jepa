from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_experiment_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "experiment.py"
    spec = importlib.util.spec_from_file_location("experiment_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split_within = _load_experiment_script().split_within


def test_dreyer_subdataset_stratified_split_covers_every_domain():
    subjects = list(range(1, 88))
    domains = {
        subject: "A" if subject <= 60 else "B" if subject <= 81 else "C"
        for subject in subjects
    }

    train, val, test = split_within(
        subjects, [0.7, 0.15, 0.15], seed=0, strata=domains
    )

    assert len(train) == 61
    assert len(val) == 13
    assert len(test) == 13
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert {
        split: {
            domain: sum(domains[s] == domain for s in members)
            for domain in ("A", "B", "C")
        }
        for split, members in (
            ("train", train),
            ("validation", val),
            ("test", test),
        )
    } == {
        "train": {"A": 42, "B": 15, "C": 4},
        "validation": {"A": 9, "B": 3, "C": 1},
        "test": {"A": 9, "B": 3, "C": 1},
    }
