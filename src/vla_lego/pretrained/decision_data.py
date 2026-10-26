"""Past-only decision sequences and instruction negatives from admitted recordings."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from vla_lego.isaac.campaign import CollectionCase
from vla_lego.isaac.pilot import piece
from vla_lego.pretrained.task_decisions import CLASSES


def observation_grid(frames: list[dict], stride: float = 0.5) -> list[int]:
    """A lower-rate decision stream; full-rate source observations remain untouched."""
    if not frames or stride <= 0:
        raise ValueError("Decision sequences need observations and a positive stride")
    if any(
        not math.isfinite(row["observation_time_s"])
        or not math.isfinite(row["image_time_s"])
        or row["image_time_s"] > row["observation_time_s"]
        for row in frames
    ):
        raise ValueError("Non-finite or future images are not decision observations")
    indices = [0]
    for i, row in enumerate(frames[1:], 1):
        if row["observation_time_s"] <= frames[i - 1]["observation_time_s"]:
            raise ValueError("Decision observations must advance the causal clock")
        if row["observation_time_s"] - frames[indices[-1]]["observation_time_s"] >= stride - 1e-8:
            indices.append(i)
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)
    return indices


def counterfactual_instruction(case: CollectionCase) -> str:
    """A feasible changed goal whose completed source scene does not satisfy it."""
    declaration = case.declaration
    targets = declaration["targets"]

    def name(code: str) -> str:
        return piece(code).description.replace("prism", "rectangular prism")

    if len(targets) == 1:
        target = piece(targets[0])
        alternatives = [p["piece"] for p in declaration["placements"] if p["piece"] != targets[0]]
        alternative = min(
            alternatives,
            key=lambda code: sum(
                getattr(target, key) != getattr(piece(code), key)
                for key in ("shape", "scale", "color")
            ),
        )
        negative = str(declaration["instruction"]).replace(name(targets[0]), name(alternative))
    else:
        a, b = map(name, targets[:2])
        negative = str(declaration["instruction"]).replace(a, "<first-piece>").replace(b, a)
        negative = negative.replace("<first-piece>", b)
    if negative == declaration["instruction"]:
        raise ValueError("Counterfactual instruction did not change the observed goal")
    return negative


@dataclass(frozen=True)
class DecisionSequence:
    case: str
    family: str
    partition: str
    directory: Path
    instruction: str
    frames: list[dict]
    indices: list[int]
    labels: list[int]
    kind: str = "recorded_instruction"

    @property
    def times(self):
        return [self.frames[index]["observation_time_s"] for index in self.indices]


def episode_sequences(root: Path, entry: dict, case: CollectionCase) -> list[DecisionSequence]:
    """Offline outcome labels stay out of the feature/input construction path."""
    directory = (root / entry["recording"]).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("Decision source escapes its admitted corpus")
    metadata = json.loads((directory / "episode.json").read_text())
    if (
        not entry["admitted"]
        or metadata["collection_case"]["case"] != entry["case"]
        or metadata["generation_family"] != entry["family"]
        or metadata["collection_case"]["partition"] != entry["partition"]
        or not (metadata["verified_success"] or metadata.get("verified_stop"))
        or metadata["privileged_policy_inputs"]
    ):
        raise ValueError("Decision source differs from reviewed membership")
    with (directory / "frames.jsonl").open() as stream:
        frames = [json.loads(line) for line in stream]
    if len(frames) != metadata["frames"]:
        raise ValueError("Decision source frame count differs")
    indices = observation_grid(frames)
    sequence = DecisionSequence(
        case=entry["case"],
        family=entry["family"],
        partition=entry["partition"],
        directory=directory,
        instruction=metadata["instruction"],
        frames=frames,
        indices=indices,
        labels=[CLASSES.index(frames[index]["task_decision"]) for index in indices],
    )
    if sequence.labels[-1] != (2 if metadata.get("verified_stop") else 1):
        raise ValueError("Verified decision sequence lacks its actual terminal label")
    result = [sequence]
    if metadata["verified_success"]:
        # This is decision-only supervision on existing images, never new actions,
        # a new episode, or an independent family. Retain the three-second history.
        negative_indices = [
            i for i in indices if frames[i]["observation_time_s"] >= sequence.times[-1] - 3.5
        ]
        result.append(
            replace(
                sequence,
                instruction=counterfactual_instruction(case),
                indices=negative_indices,
                labels=[0] * len(negative_indices),
                kind="counterfactual_instruction",
            )
        )
    return result
