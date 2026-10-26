"""Lazy episode access to admitted pilot data, with explicit future-action masks."""

from __future__ import annotations

import json
from pathlib import Path

from vla_lego.pretrained.decisions import DECISION_TARGETS


def load_observation(directory: Path, row: dict, *, fps=20, include_arm_velocity=True) -> dict:
    """Construct deployable inputs without actions, labels, stages or simulator truth."""
    import numpy as np
    from PIL import Image

    state = row["joint_position"]
    if include_arm_velocity:
        state = [*state, *row["joint_velocity"][:14]]
    image_path = (directory / row["head_rgb"]).resolve()
    if not image_path.is_relative_to(directory.resolve()):
        raise ValueError("Image path escapes its recording.")
    if "head_frame" in row:
        from vla_lego.pretrained.rgb_video import read_rgb_frame

        pixels = read_rgb_frame(image_path, row["head_frame"], fps=fps)
    else:
        with Image.open(image_path) as source:
            pixels = np.asarray(source.convert("RGB")).copy()
    return {
        "observation.state": np.asarray(state, dtype=np.float32),
        "observation.images.head": np.asarray(pixels, dtype=np.float32).transpose(2, 0, 1) / 255.0,
    }


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
        include_decisions: bool = True,
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
        self.include_decisions = include_decisions

    def __len__(self) -> int:
        return len(self.eligible_frames)

    def __getitem__(self, sample: int) -> dict:
        import numpy as np

        index = self.eligible_frames[sample]
        row = self.frames[index]
        actions = np.zeros((self.horizon, 18 if self.include_decisions else 16), dtype=np.float32)
        action_is_pad = np.ones(self.horizon, dtype=bool)
        for offset, future in enumerate(self.frames[index : index + self.horizon]):
            if not self.annotations[index + offset]["positive_imitation"]:
                # End this target chunk at an error. Corrective samples restart
                # from their own observed error state, with the raw context retained.
                break
            actions[offset] = (
                [*future["action"], *DECISION_TARGETS[future["task_decision"]]]
                if self.include_decisions
                else future["action"]
            )
            action_is_pad[offset] = False
        if self.action_representation == "observation_arm_offset":
            # Every future target uses this observation's measured anchor,
            # never an unobserved future state or a moving runtime anchor.
            actions[~action_is_pad, :14] -= np.asarray(row["joint_position"][:14], np.float32)
        return {
            **load_observation(
                self.directory,
                row,
                fps=self.metadata["fps"] if "head_frame" in row else 20,
                include_arm_velocity=self.include_arm_velocity,
            ),
            "task": self.metadata["instruction"],
            "action": actions,
            "action_is_pad": action_is_pad,
        }
