"""Compact temporal-head inputs with balanced learning and natural evaluation streams."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from vla_lego.isaac.campaign import EXPECTED
from vla_lego.pretrained.task_decisions import causal_history


def history_samples(times: list[float], kind: str):
    if kind not in ("recorded_instruction", "counterfactual_instruction"):
        raise ValueError("Unknown decision sequence kind")
    if (
        not times
        or not np.isfinite(times).all()
        or any(b <= a for a, b in zip(times, times[1:], strict=False))
    ):
        raise ValueError("Decision observations must advance the clock")
    return [
        (index, causal_history(times, index))
        for index, time_s in enumerate(times)
        # A cropped counterfactual is not a new startup: require actual past context.
        if kind == "recorded_instruction" or time_s - times[0] >= 3.0 - 1e-8
    ]


class DecisionFeatures(Dataset[dict]):
    def __init__(self, root: Path, partition: str, *, seed=20261026, draws=1_280_000):
        if partition not in ("train", "calibration", "confirmation"):
            raise ValueError("Unknown decision feature partition")
        protocol = json.loads((root / "protocol.json").read_text())
        index = json.loads((root / "index.json").read_text())
        sources = protocol["sources"]
        if (
            not index["complete"]
            or Counter((row["partition"], row["category"]) for row in sources) != EXPECTED
        ):
            raise ValueError("Decision learning waits for the complete620-source feature cache")
        members = {row["case"]: row for row in sources}
        if len(members) != 620:
            raise ValueError("Decision features require distinct admitted episodes")
        recorded = [
            row["case"] for row in index["sequences"] if row["kind"] == "recorded_instruction"
        ]
        if len(recorded) != 620 or set(recorded) != set(members):
            raise ValueError("Feature cache does not cover the admitted episodes exactly once")
        expected = {(case, "recorded_instruction") for case in members} | {
            (case, "counterfactual_instruction")
            for case, source in members.items()
            if source["category"] != "stop"
        }
        actual = [(row["case"], row["kind"]) for row in index["sequences"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Derived decision cache membership differs")
        self.seed, self.draws, self.partition = seed, draws, partition
        self.sequences: list[dict] = []
        self.pools: dict[int, dict[str, list[tuple[int, int]]]] = {
            label: defaultdict(list) for label in range(3)
        }
        self.original_class_counts: Counter = Counter()
        self.feature_width = None
        for row in index["sequences"]:
            source = members[row["case"]]
            if row["partition"] != source["partition"] or row["family"] != source["family"]:
                raise ValueError("Derived decision features cross source membership")
            if row["partition"] != partition:
                continue
            path = (root / row["file"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Decision cache path escapes its corpus")
            with np.load(path) as arrays:
                features, labels, times = arrays["features"], arrays["labels"], arrays["times"]
            if (
                features.ndim != 2
                or len(features) != len(labels)
                or len(features) != len(times)
                or len(features) != row["observations"]
                or not np.isfinite(features).all()
                or not np.isfinite(times).all()
                or not set(labels.tolist()) <= {0, 1, 2}
            ):
                raise ValueError("Invalid cached decision observations")
            if self.feature_width not in (None, features.shape[1]):
                raise ValueError("Frozen feature widths differ")
            self.feature_width = features.shape[1]
            samples = dict(history_samples(times.tolist(), row["kind"]))
            if not samples:
                raise ValueError("Decision sequence lacks a complete causal context")
            number = len(self.sequences)
            self.sequences.append(
                {
                    **row,
                    "features": features,
                    "labels": labels,
                    "times": times,
                    "histories": samples,
                }
            )
            for i in samples:
                self.pools[int(labels[i])][row["family"]].append((number, i))
            if row["kind"] == "recorded_instruction":
                self.original_class_counts.update(labels.tolist())
        if not self.sequences or any(not pool for pool in self.pools.values()):
            raise ValueError("All decision classes need actual recorded support")

    @property
    def protocol(self):
        return {
            "feature_width": self.feature_width,
            "partition": self.partition,
            "sampling": "uniform class, family, observation; counterfactuals inherit source family",
            "sampled_class_priors": [1 / 3] * 3,
            "recorded_class_counts": dict(self.original_class_counts),
            "calibration": "natural sequences; oversampling is not a calibrated posterior",
        }

    def __len__(self):
        return self.draws

    def __getitem__(self, draw):
        if self.partition != "train":
            raise ValueError("Evaluate complete natural sequences, not balanced development draws")
        rng = random.Random(self.seed + draw * 1_000_003)
        label = rng.randrange(3)
        family = rng.choice(sorted(self.pools[label]))
        number, index = rng.choice(self.pools[label][family])
        sequence = self.sequences[number]
        return {
            "history": sequence["features"][sequence["histories"][index]].astype(np.float32),
            "label": label,
        }
