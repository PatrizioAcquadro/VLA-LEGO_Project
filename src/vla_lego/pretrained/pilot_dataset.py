"""Lazy episode access to admitted pilot data, with explicit future-action masks."""

from __future__ import annotations

import json
from pathlib import Path

from vla_lego.pretrained.decisions import DECISION_TARGETS


class PilotEpisode:
    """Prepare SmolVLA-compatible samples without loading privileged evidence.

    One instance remains one physical episode. Sampling frames or action chunks
    never changes its generation family or creates independent demonstrations.
    """

    def __init__(
        self,
        directory: Path,
        *,
        horizon: int = 50,
        include_arm_velocity: bool = True,
        action_representation: str = "absolute",
    ):
        if horizon < 1:
            raise ValueError("Action horizon must be positive.")
        self.directory = directory
        self.metadata = json.loads((directory / "episode.json").read_text())
        if (
            not (self.metadata["verified_success"] or self.metadata.get("verified_stop"))
            or self.metadata["supervision_role"] == "context_only"
        ):
            raise ValueError("Only admitted successful/corrective/stop episodes supply targets.")
        with (directory / "frames.jsonl").open() as stream:
            self.frames = [json.loads(line) for line in stream]
        with (directory / "annotations.jsonl").open() as stream:
            self.annotations = [json.loads(line) for line in stream]
        if len(self.frames) != len(self.annotations) or len(self.frames) != self.metadata["frames"]:
            raise ValueError("Episode frame/annotation counts differ.")
        self.eligible_frames = [
            i for i, annotation in enumerate(self.annotations) if annotation["positive_imitation"]
        ]
        self.horizon = horizon
        self.include_arm_velocity = include_arm_velocity
        if action_representation not in ("absolute", "observation_arm_offset"):
            raise ValueError("Unknown pilot action representation")
        self.action_representation = action_representation

    def __len__(self) -> int:
        return len(self.eligible_frames)

    def __getitem__(self, sample: int) -> dict:
        import numpy as np
        from PIL import Image

        index = self.eligible_frames[sample]
        row = self.frames[index]
        state = row["joint_position"]
        if self.include_arm_velocity:
            state = [*state, *row["joint_velocity"][:14]]
        actions = np.zeros((self.horizon, 18), dtype=np.float32)
        action_is_pad = np.ones(self.horizon, dtype=bool)
        for offset, future in enumerate(self.frames[index : index + self.horizon]):
            if not self.annotations[index + offset]["positive_imitation"]:
                # End this target chunk at an error. Corrective samples restart
                # from their own observed error state, with the raw context retained.
                break
            actions[offset] = [*future["action"], *DECISION_TARGETS[future["task_decision"]]]
            action_is_pad[offset] = False
        if self.action_representation == "observation_arm_offset":
            # Every future target uses this observation's measured anchor,
            # never an unobserved future state or a moving runtime anchor.
            actions[~action_is_pad, :14] -= np.asarray(row["joint_position"][:14], np.float32)
        image_path = (self.directory / row["head_rgb"]).resolve()
        if not image_path.is_relative_to(self.directory.resolve()):
            raise ValueError("Image path escapes its recording.")
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
        return {
            "observation.state": np.asarray(state, dtype=np.float32),
            "observation.images.head": rgb,
            "task": self.metadata["instruction"],
            "action": actions,
            "action_is_pad": action_is_pad,
        }
