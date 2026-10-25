"""Balanced, deterministic sampling of admitted physical pilot episodes."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

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


def policy_acquisition_phase(stage: str) -> str | None:
    """Stratify the first physical acquisition of a policy-error continuation."""
    if "park" in stage:
        return "policy_start"
    if stage.startswith("level0_grip"):
        return "policy_grasp"
    if stage.startswith("level0_lift"):
        return "policy_lift"
    if stage.startswith("level0_") and any(
        word in stage for word in ("edge", "source_high", "approach", "descend")
    ):
        return "policy_approach"
    return None


class PilotTrainingData(Dataset[dict]):
    """Draw ids, rather than worker RNG state, determine the exact training stream."""

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        seed: int,
        draws: int = 6_400_000,
        action_representation: str = "absolute",
    ):
        if split not in ("train", "development"):
            raise ValueError("Unknown pilot split")
        self.root, self.split, self.seed, self.draws = root, split, seed, draws
        self.action_representation = action_representation
        cohort_path = root / "cohort.json"
        self.cohort = json.loads(cohort_path.read_text()) if cohort_path.exists() else None
        if self.cohort is not None and not self.cohort.get("name"):
            raise ValueError("A new cohort requires an explicit name")
        entries = json.loads((root / f"{split}-episodes.json").read_text())
        counts = self.cohort["episode_counts"] if self.cohort else {"train": 50, "development": 10}
        if (
            len(entries) != counts[split]
            or len({entry["case"] for entry in entries}) != len(entries)
            or len({Path(entry["path"]).parent.name for entry in entries}) != len(entries)
        ):
            raise ValueError("Episode membership differs from the declared cohort")
        self.policy_correction_probability = (
            float(self.cohort.get("policy_correction_probability", 0.0))
            if self.cohort and split == "train"
            else 0.0
        )
        self.policy_acquisition_probability = (
            float(self.cohort["policy_acquisition_probability"]) if self.cohort else 0.0
        )
        if not 0 <= self.policy_correction_probability < 1 or not (
            0 <= self.policy_acquisition_probability <= 1
        ):
            raise ValueError("Invalid declared policy-correction sampling mixture")
        self.episodes: list[PilotEpisode] = []
        self.cases: list[str] = []
        self.pools: list[dict[str, list[int]]] = []
        self.categories: dict[str, list[int]] = defaultdict(list)
        self.families: set[str] = set()
        self.membership: list[dict] = []
        for entry in entries:
            directory = root / "attempts" / Path(entry["path"]).parent.name / "recording"
            episode = PilotEpisode(directory, action_representation=action_representation)
            metadata = episode.metadata
            if metadata["split"] != split:
                raise ValueError("Indexed episode belongs to another split")
            if metadata.get("collection_condition") != "marker_free_continuous_site_v1":
                raise ValueError("Only the admitted marker-free corpus is accepted")
            category = metadata["collection_case"]["category"]
            if metadata.get("correction_source"):
                if (
                    split != "train"
                    or not self.cohort
                    or not self.policy_correction_probability
                    or metadata["supervision_role"] != "verified_correction"
                    or not metadata["correction_source"].get("exact_native_parity")
                ):
                    raise ValueError(
                        "Policy-error data require a declared training correction cohort"
                    )
                category = "policy_correction"
            pools: dict[str, list[int]] = defaultdict(list)
            corrective_until = -1
            for sample, raw in enumerate(episode.eligible_frames):
                row, annotation = episode.frames[raw], episode.annotations[raw]
                pools[phase(annotation["stage"])].append(sample)
                if category == "policy_correction":
                    acquisition = policy_acquisition_phase(annotation["stage"])
                    if acquisition:
                        pools[acquisition].append(sample)
                if row["task_decision"] != "continue":
                    pools["terminal"].append(sample)
                if raw and not episode.annotations[raw - 1]["positive_imitation"]:
                    corrective_until = raw + 100
                if raw < corrective_until:
                    pools["correction"].append(sample)
            if not pools["terminal"] or (category == "recovery" and not pools["correction"]):
                raise ValueError("Missing terminal/corrective supervision in an admitted episode")
            if category == "policy_correction" and not all(
                pools[name]
                for name in ("policy_start", "policy_approach", "policy_grasp", "policy_lift")
            ):
                raise ValueError(
                    "Policy correction must cover initial state through actual first lift"
                )
            self.categories[category].append(len(self.episodes))
            self.episodes.append(episode)
            self.cases.append(entry["case"])
            self.pools.append(dict(pools))
            self.families.add(metadata["generation_family"])
            self.membership.append(
                dict(
                    entry,
                    family=metadata["generation_family"],
                    sampling_category=category,
                    correction_source=metadata.get("correction_source"),
                )
            )
        required = {"nominal", "recovery", "stop"}
        if self.policy_correction_probability:
            required.add("policy_correction")
        if set(self.categories) != required:
            raise ValueError("The pilot requires nominal, recovery and stop coverage")

    @property
    def protocol(self) -> dict:
        """Freeze declared membership and sampling across allocation resumes."""
        return {"cohort": self.cohort, "episodes": self.membership}

    @property
    def sampling_description(self) -> dict:
        result = dict(SAMPLING_DESCRIPTION)
        if self.policy_correction_probability:
            result.update(
                base_probability=1 - self.policy_correction_probability,
                policy_correction_probability=self.policy_correction_probability,
                policy_acquisition_probability=self.policy_acquisition_probability,
                policy_episode_sampling="uniform source families, then uniform continuations",
                policy_acquisition_sampling="equal start/approach/grasp/lift, then uniform frame",
            )
        return result

    def __len__(self) -> int:
        return self.draws

    def select(self, draw: int) -> tuple[int, int, str]:
        rng = random.Random(self.seed + draw * 1_000_003)
        correction = self.policy_correction_probability > 0 and (
            rng.random() < self.policy_correction_probability
        )
        category = (
            "policy_correction"
            if correction
            else rng.choices(("nominal", "recovery", "stop"), weights=(60, 30, 10))[0]
        )
        eligible = self.categories[category]
        if correction:
            family = rng.choice(
                sorted({self.episodes[i].metadata["generation_family"] for i in eligible})
            )
            eligible = [
                i for i in eligible if self.episodes[i].metadata["generation_family"] == family
            ]
        number = rng.choice(eligible)
        pools = self.pools[number]
        if correction and rng.random() < self.policy_acquisition_probability:
            bucket = rng.choice(["policy_start", "policy_approach", "policy_grasp", "policy_lift"])
            return number, rng.choice(pools[bucket]), bucket
        value = rng.random()
        if category == "stop":
            bucket = (
                "terminal"
                if value < 0.5
                else rng.choice([name for name in pools if name not in ("terminal", "correction")])
            )
        elif value < 0.10:
            bucket = "terminal"
        elif value < 0.15 and "park" in pools:
            bucket = "park"
        elif category == "recovery" and value < 0.50:
            bucket = "correction"
        else:
            bucket = rng.choice(
                [
                    name
                    for name in pools
                    if name not in ("park", "terminal", "correction")
                    and not name.startswith("policy_")
                ]
            )
        return number, rng.choice(pools[bucket]), bucket

    def __getitem__(self, draw: int) -> dict:
        episode, sample, _ = self.select(draw)
        return self.episodes[episode][sample]

    def diagnostic_samples(self) -> list[tuple[int, int, str]]:
        """Cover startup, first-block transitions and later operational phases.

        Teacher stages stratify offline evaluation only; they never enter inputs.
        Keep these samples fixed across checkpoints within a new campaign.
        """
        result: list[tuple[int, int, str]] = []
        for number, pools in enumerate(self.pools):
            episode = self.episodes[number]
            selected: dict[int, str] = {0: "startup"}
            first_block: dict[str, list[int]] = defaultdict(list)
            for sample, raw in enumerate(episode.eligible_frames):
                stage = episode.annotations[raw]["stage"]
                if stage.startswith("level0_"):
                    first_block[stage].append(sample)
            for stage, values in first_block.items():
                selected.setdefault(values[0], stage + "/start")
                selected.setdefault(values[len(values) // 2], stage + "/middle")
            for name in sorted(pools):
                if name != "park":
                    values = pools[name]
                    selected.setdefault(values[len(values) // 2], name)
            result.extend((number, sample, label) for sample, label in sorted(selected.items()))
        return result

    def statistics(self) -> dict:
        if self.split != "train":
            raise ValueError("Normalization may only be fitted on training families")
        totals = {"observation.state": np.zeros(32), "action": np.zeros(18)}
        squares = {name: np.zeros_like(value) for name, value in totals.items()}
        count = 0
        for episode in self.episodes:
            for raw in episode.eligible_frames:
                row = episode.frames[raw]
                values = {
                    "observation.state": np.asarray(
                        [*row["joint_position"], *row["joint_velocity"][:14]], dtype=np.float64
                    ),
                    "action": np.asarray([*row["action"], 0.0, 0.0], dtype=np.float64),
                }
                for name, value in values.items():
                    totals[name] += value
                    squares[name] += value * value
                count += 1
        result = {}
        for name, total in totals.items():
            mean = total / count
            std = np.sqrt(np.maximum(squares[name] / count - mean * mean, 0))
            std[std < 1e-4] = 1.0
            if name == "action":
                # Binary decisions retain their [0,1] meaning; rare class frequency
                # must not turn their z-scores into enormous motor-like values.
                mean[16:], std[16:] = 0.0, 1.0
            result[name] = {"mean": mean.tolist(), "std": std.tolist()}
        if self.action_representation == "observation_arm_offset":
            result["action"] = self._offset_statistics()
        return result

    def _offset_statistics(self) -> dict:
        """Fit the actual anchored future targets, over valid training windows."""
        total, squares, count = np.zeros(16), np.zeros(16), 0
        for episode in self.episodes:
            positions = np.asarray(
                [r["joint_position"][:14] for r in episode.frames], dtype=np.float64
            )
            actions = np.asarray([r["action"] for r in episode.frames], dtype=np.float64)
            invalid = np.r_[
                0, np.cumsum([not a["positive_imitation"] for a in episode.annotations])
            ]
            for offset in range(min(episode.horizon, len(actions))):
                valid = invalid[offset + 1 :] == invalid[: len(actions) - offset]
                values = actions[offset:][valid].copy()
                values[:, :14] -= positions[: len(actions) - offset][valid]
                total += values.sum(0)
                squares += (values * values).sum(0)
                count += len(values)
        mean = total / count
        std = np.sqrt(np.maximum(squares / count - mean * mean, 0))
        std[std < 1e-4] = 1.0
        return {"mean": [*mean.tolist(), 0.0, 0.0], "std": [*std.tolist(), 1.0, 1.0]}


SAMPLING_DESCRIPTION = {
    "category_weights": {"nominal": 0.6, "recovery": 0.3, "stop": 0.1},
    "within_category": "uniform episodes, then declared phase mixture",
    "assembly_terminal_probability": 0.10,
    "assembly_parking_probability": 0.05,
    "recovery_correction_probability": 0.35,
    "correction_context_seconds": 5,
    "remaining_assembly": "uniform available operational phases, then uniform frame",
    "stop_terminal_probability": 0.50,
    "decision_normalization": "identity",
    "loss_reduction": "valid targets within each sample, then equal sample mean",
}
