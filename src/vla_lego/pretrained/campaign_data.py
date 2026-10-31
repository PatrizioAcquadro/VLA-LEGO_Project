"""Motor-only curriculum sampling of the admitted, family-separated 500/120 corpus."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from vla_lego.isaac.campaign import CONDITION, EXPECTED, RECOVERIES
from vla_lego.pretrained.pilot_dataset import PilotEpisode


def phase(stage: str) -> str:
    if "park" in stage:
        return "park"
    if any(word in stage for word in ("stabil", "restrain", "cooper")):
        return "cooperation"
    if "release" in stage:
        return "release"
    if any(word in stage for word in ("retreat", "exit", "withdraw")):
        return "withdraw"
    if any(word in stage for word in ("place", "support")):
        return "place"
    if "grip" in stage:
        return "grasp"
    return "move"


def motor_phase(stage: str) -> str:
    base = phase(stage)
    if base == "cooperation":
        return base
    if "lift" in stage:
        return "lift"
    if any(word in stage for word in ("transport", "carry")):
        return "transport"
    if any(word in stage for word in ("approach", "descend", "source_high", "edge")):
        return "approach"
    return base


def motor_supervision(annotations: list[dict], category: str) -> np.ndarray:
    """An impossible-task prefix is context; its verified released stop is the motor target."""
    return np.asarray(
        [
            a["positive_imitation"]
            and (category != "stop" or a["stage"] == "verified_justified_abstention")
            for a in annotations
        ],
        dtype=bool,
    )


def recovery_coverage(root: Path, entries: list[dict]) -> Counter:
    """Count verified observed errors, independently of the requested collection mechanism."""
    aliases = {
        "missed grasp": "missed_grasp",
        "wrong distractor picked": "wrong_piece",
        "physically consequential incomplete placement": "placement",
    }
    coverage: Counter = Counter()
    for entry in entries:
        if entry["category"] != "recovery":
            continue
        recording = (root / entry["recording"]).resolve()
        if not recording.is_relative_to(root.resolve()):
            raise ValueError("Correction source escapes the admitted corpus")
        report = json.loads((recording.parent / "report.json").read_text())
        fault = report.get("fault", {})
        mechanism = aliases.get(fault.get("type"), fault.get("type"))
        if (
            not report.get("passed")
            or not fault.get("physically_verified")
            or mechanism not in RECOVERIES
        ):
            raise ValueError("Recovery coverage requires a verified physical error and completion")
        observed = fault.get("policy_source", {}).get("observed_mechanism")
        if observed is not None and observed != mechanism:
            raise ValueError("Observed policy error differs from the recorded recovery mechanism")
        if entry.get("observed_mechanism", mechanism) != mechanism:
            raise ValueError("Admission labels differ from the physically observed recovery")
        coverage[(entry["partition"], mechanism)] += 1
    return coverage


def admitted_entries(root: Path) -> list[dict]:
    entries = json.loads((root / "admitted-episodes.json").read_text())["episodes"]
    if any(not entry.get("admitted") for entry in entries):
        raise ValueError("Only reviewed, admitted episodes satisfy the learning gate")
    if Counter((e["partition"], e["category"]) for e in entries) != EXPECTED:
        raise ValueError("Learning waits for the complete admitted500/60/60 corpus")
    if len({e["case"] for e in entries}) != 620:
        raise ValueError("Repeated case identities cannot satisfy the new corpus quota")
    family_partitions: dict[str, str] = {}
    for entry in entries:
        if family_partitions.setdefault(entry["family"], entry["partition"]) != entry["partition"]:
            raise ValueError("A generation family crosses learning partitions")
    from vla_lego.isaac.recording import verify_policy_correction

    policy_corrections = 0
    for entry in entries:
        if entry["category"] != "recovery":
            continue
        directory = (root / entry["recording"]).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("Correction source escapes the admitted corpus")
        metadata = json.loads((directory / "episode.json").read_text())
        row = metadata["collection_case"]
        if row.get("correction_origin") == "policy":
            if entry["partition"] != "train":
                raise ValueError("Policy-state correction quota belongs to TRAIN only")
            verify_policy_correction(directory, metadata)
            policy_corrections += 1
        elif row.get("mechanism") == "interrupted_progress":
            fault = json.loads((directory.parent / "report.json").read_text()).get("fault", {})
            if not (
                fault.get("mechanism") == "premature_stabilizer_withdrawal"
                and fault.get("physically_verified")
                and fault.get("stabilizer_grasp_before")
                and fault.get("stabilizer_grasp_at_intervention") is False
                and fault.get("placer_grasp_at_intervention")
                and fault.get("stabilization_restored")
            ):
                raise ValueError("A nominal pause cannot satisfy a teacher coordination recovery")
    if policy_corrections < 30:
        raise ValueError("Learning requires30 actual new TRAIN policy-state corrections")
    expected_coverage = Counter(
        {
            (partition, mechanism): count
            for partition, count in (("train", 15), ("calibration", 2), ("confirmation", 2))
            for mechanism in RECOVERIES
        }
    )
    if recovery_coverage(root, entries) != expected_coverage:
        raise ValueError(
            "Learning waits for actual recovery coverage:15 TRAIN and4 development per mechanism"
        )
    return list(entries)


class CampaignData(Dataset[dict]):
    """One deterministic sample stream changes curriculum after20k completed updates."""

    def __init__(self, root: Path, partition: str, *, seed: int, draws=6_400_000):
        if partition not in ("train", "calibration", "confirmation"):
            raise ValueError("Unknown campaign partition")
        entries = admitted_entries(root)
        self.seed, self.draws, self.partition = seed, draws, partition
        self.action_representation = "observation_arm_offset"
        self.episodes: list[PilotEpisode] = []
        self.entries: list[dict] = []
        self.pools: list[dict[str, list[int]]] = []
        self.categories: dict[str, list[int]] = defaultdict(list)
        for entry in entries:
            if entry["partition"] != partition:
                continue
            directory = (root / entry["recording"]).resolve()
            if not directory.is_relative_to(root.resolve()):
                raise ValueError("Indexed recording must belong to the transferred corpus")
            episode = PilotEpisode(
                directory, include_decisions=False, action_representation="observation_arm_offset"
            )
            metadata = episode.metadata
            row = metadata["collection_case"]
            if (
                metadata["collection_condition"] != CONDITION
                or row["partition"] != partition
                or row["case"] != entry["case"]
                or metadata["generation_family"] != entry["family"]
            ):
                raise ValueError("Episode differs from its admitted membership")
            pools: dict[str, list[int]] = defaultdict(list)
            supervised = motor_supervision(episode.annotations, entry["category"])
            for sample, raw in enumerate(episode.eligible_frames):
                if not supervised[raw]:
                    continue
                stage = episode.annotations[raw]["stage"]
                if entry["category"] != "stop" and any(
                    t in stage for t in ("verified_complete", "final_released_dwell")
                ):
                    continue
                match = re.search(r"level(\d+)", stage)
                level = match[1] if match else "shared"
                pools[f"{level}/{motor_phase(stage)}"].append(sample)
            if not pools:
                raise ValueError("No positive operational motor targets in admitted episode")
            self.categories[entry["category"]].append(len(self.episodes))
            self.episodes.append(episode)
            self.entries.append(entry)
            self.pools.append(dict(pools))

    @property
    def families(self):
        return {entry["family"] for entry in self.entries}

    @property
    def sampling_description(self):
        return self.protocol["sampling"]

    @property
    def protocol(self):
        return {
            "episodes": self.entries,
            "motor_width": 16,
            "history_inputs": "observations only",
            "control_updates": 20000,
            "full_category_weights": [0.30, 0.50, 0.15, 0.05],
            "sampling": "uniform families, episodes, available level/phase buckets",
            "stop_motor_targets": "verified released abstention only; prefix remains context",
        }

    def __len__(self):
        return self.draws

    def select(self, draw: int):
        rng = random.Random(self.seed + draw * 1_000_003)
        category = (
            "single_block"
            if draw < 20000 * 64
            else rng.choices(
                ("single_block", "tower", "recovery", "stop"), weights=(30, 50, 15, 5)
            )[0]
        )
        candidates = self.categories[category]
        family = rng.choice(sorted({self.entries[i]["family"] for i in candidates}))
        number = rng.choice([i for i in candidates if self.entries[i]["family"] == family])
        bucket = rng.choice(sorted(self.pools[number]))
        return number, rng.choice(self.pools[number][bucket]), bucket

    def __getitem__(self, draw: int):
        episode, sample, _ = self.select(draw)
        return self.episodes[episode][sample]

    def diagnostic_samples(self, *, single_only=False):
        result = []
        for number, _episode in enumerate(self.episodes):
            if single_only and self.entries[number]["category"] != "single_block":
                continue
            stop = self.entries[number]["category"] == "stop"
            first = min(min(values) for values in self.pools[number].values()) if stop else 0
            selected = {first: "released_stop" if stop else "startup"}
            for bucket, values in self.pools[number].items():
                if bucket.endswith(("grasp", "place", "release")) or bucket == "shared/cooperation":
                    selected.setdefault(values[len(values) // 2], bucket)
            result.extend((number, sample, bucket) for sample, bucket in sorted(selected.items()))
        return result

    def statistics(self) -> dict:
        if self.partition != "train":
            raise ValueError("Only training families fit normalization")
        state_total, state_squares, state_count = np.zeros(32), np.zeros(32), 0
        action_total, action_squares, action_count = np.zeros(16), np.zeros(16), 0
        for entry, episode in zip(self.entries, self.episodes, strict=True):
            supervised = motor_supervision(episode.annotations, entry["category"])
            positions = np.asarray([row["joint_position"][:14] for row in episode.frames])
            actions = np.asarray([row["action"] for row in episode.frames])
            states = np.asarray(
                [[*row["joint_position"], *row["joint_velocity"][:14]] for row in episode.frames]
            )[supervised]
            state_total += states.sum(0)
            state_squares += (states * states).sum(0)
            state_count += len(states)
            invalid = np.r_[0, np.cumsum(~supervised)]
            for offset in range(min(episode.horizon, len(actions))):
                valid = invalid[offset + 1 :] == invalid[: len(actions) - offset]
                values = actions[offset:][valid].copy()
                values[:, :14] -= positions[: len(actions) - offset][valid]
                action_total += values.sum(0)
                action_squares += (values * values).sum(0)
                action_count += len(values)
        result = {}
        for name, total, squares, count in (
            ("observation.state", state_total, state_squares, state_count),
            ("action", action_total, action_squares, action_count),
        ):
            mean = total / count
            std = np.sqrt(np.maximum(0, squares / count - mean * mean))
            std[std < 1e-4] = 1
            result[name] = {"mean": mean.tolist(), "std": std.tolist()}
        return result
