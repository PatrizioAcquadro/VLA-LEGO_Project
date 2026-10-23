"""Causal sensor/action recording with privileged evidence kept in a separate file."""

from __future__ import annotations

import json
from pathlib import Path


class EpisodeRecorder:
    """Stream images and rows; accept positive labels only after verified completion."""

    def __init__(self, directory: Path, instruction: str, *, family: str, split: str):
        if not family or split not in ("train", "development"):
            raise ValueError(
                "Collection requires a generation family and a train/development split."
            )
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        (directory / "head").mkdir()
        self.frames = (directory / "frames.jsonl").open("w")
        self.evidence = (directory / "privileged.jsonl").open("w")
        self.instruction = instruction
        self.family, self.split = family, split
        self.count = 0
        self.closed = False
        self.state_names: tuple[str, ...] | None = None
        self.action_names: tuple[str, ...] | None = None
        self.start_time: float | None = None

    def record(
        self,
        observation: dict,
        action,
        action_names: tuple[str, ...],
        truth: list[dict],
        *,
        requested_action,
        command_time_s: float,
    ):
        import numpy as np
        from PIL import Image

        if self.closed:
            raise RuntimeError("Cannot append to a closed episode.")
        state_names = tuple(observation["joint_names"])
        if self.state_names is None:
            self.state_names, self.action_names = state_names, action_names
            self.start_time = observation["timestamp_s"]
        if state_names != self.state_names or action_names != self.action_names:
            raise ValueError("Observation/action joint ordering changed inside an episode.")
        if observation["image_timestamp_s"] > observation["timestamp_s"]:
            raise ValueError("Future camera frame is not a causal observation.")
        if (
            command_time_s > observation["timestamp_s"]
            or abs(command_time_s / 0.05 - round(command_time_s / 0.05)) > 1e-8
        ):
            raise ValueError("Action is not aligned to the declared control clock.")
        if not all(
            np.isfinite(value).all()
            for value in (
                action,
                requested_action,
                observation["joint_position"],
                observation["joint_velocity"],
            )
        ):
            raise ValueError("Recorded sensor/action values must be finite.")
        image_path = f"head/{self.count:06d}.png"
        Image.fromarray(observation["head_rgb"]).save(self.directory / image_path, compress_level=1)
        row = {
            "frame": self.count,
            "observation_time_s": observation["timestamp_s"],
            "image_time_s": observation["image_timestamp_s"],
            "command_time_s": command_time_s,
            "head_rgb": image_path,
            "joint_position": observation["joint_position"].tolist(),
            "joint_velocity": observation["joint_velocity"].tolist(),
            "action": np.asarray(action).tolist(),
            "requested_action": np.asarray(requested_action).tolist(),
        }
        self.frames.write(json.dumps(row) + "\n")
        self.evidence.write(json.dumps({"frame": self.count, "blocks": truth}) + "\n")
        self.count += 1
        if self.count % 20 == 0:
            self.frames.flush()
            self.evidence.flush()

    def finish(self, *, verified_success: bool, reason: str) -> None:
        if self.closed:
            return
        self.frames.close()
        self.evidence.close()
        metadata = {
            "instruction": self.instruction,
            "generation_family": self.family,
            "split": self.split,
            "frames": self.count,
            "fps": 20,
            "state_names": self.state_names,
            "action_names": self.action_names,
            "start_time_s": self.start_time,
            "verified_success": verified_success,
            "supervision_role": "verified_nominal" if verified_success else "context_only",
            "reason": reason,
            "action_timing": "observation_t then target_action_t then 10 native physics steps",
            "privileged_policy_inputs": False,
            "control_clock_enforced": True,
        }
        (self.directory / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.closed = True
