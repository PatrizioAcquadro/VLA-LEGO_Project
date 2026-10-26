"""Causal sensor/action recording with privileged evidence kept in a separate file."""

from __future__ import annotations

import json
from pathlib import Path


class EpisodeRecorder:
    """Stream images and rows; accept positive labels only after verified completion."""

    def __init__(
        self,
        directory: Path,
        instruction: str,
        *,
        family: str,
        split: str,
        correction_source: dict | None = None,
        task_decisions: bool = False,
        collection_case: dict | None = None,
        collection_condition: str | None = None,
        png_compress_level: int = 1,
    ):
        if not family or split not in ("train", "development"):
            raise ValueError(
                "Collection requires a generation family and a train/development split."
            )
        if correction_source is not None and not correction_source.get("exact_native_parity"):
            raise ValueError("A corrective continuation requires exact native prefix replay.")
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        (directory / "head").mkdir()
        self.frames = (directory / "frames.jsonl").open("w")
        self.evidence = (directory / "privileged.jsonl").open("w")
        self.annotations = (directory / "annotations.jsonl").open("w")
        self.stage = "initial_context"
        self.positive_imitation = True
        self.instruction = instruction
        self.family, self.split = family, split
        self.correction_source = correction_source
        self.task_decisions = task_decisions
        self.collection_case = collection_case
        self.collection_condition = collection_condition
        if not 0 <= png_compress_level <= 9:
            raise ValueError("PNG compression must be in0..9")
        self.png_compress_level = png_compress_level
        self.fault_verified = False
        self.task_decision = "continue"
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
        Image.fromarray(observation["head_rgb"]).save(
            self.directory / image_path, compress_level=self.png_compress_level
        )
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
        if self.task_decisions:
            from vla_lego.pretrained.decisions import DECISION_TARGETS

            if self.task_decision not in DECISION_TARGETS:
                raise ValueError("Unknown teacher task-decision annotation.")
            row["task_decision"] = self.task_decision
        self.frames.write(json.dumps(row) + "\n")
        self.evidence.write(json.dumps({"frame": self.count, "blocks": truth}) + "\n")
        self.annotations.write(
            json.dumps(
                {
                    "frame": self.count,
                    "stage": self.stage,
                    "positive_imitation": self.positive_imitation,
                }
            )
            + "\n"
        )
        self.count += 1
        if self.count % 20 == 0:
            self.frames.flush()
            self.evidence.flush()
            self.annotations.flush()

    def finish(self, *, verified_success: bool, reason: str, verified_stop: bool = False) -> None:
        if self.closed:
            return
        if verified_stop and (verified_success or not self.task_decisions):
            raise ValueError("A verified stop is distinct from completion and requires labels.")
        if (
            verified_success
            and self.collection_case is not None
            and self.collection_case["category"] == "recovery"
            and not self.fault_verified
        ):
            raise ValueError("A corrective case requires a physically verified fault.")
        self.frames.close()
        self.evidence.close()
        self.annotations.close()
        metadata: dict = {
            "instruction": self.instruction,
            "collection_condition": self.collection_condition,
            "generation_family": self.family,
            "split": self.split,
            "frames": self.count,
            "fps": 20,
            "state_names": self.state_names,
            "action_names": self.action_names,
            "start_time_s": self.start_time,
            "verified_success": verified_success,
            "verified_stop": verified_stop,
            "task_decisions_recorded": self.task_decisions,
            "supervision_role": (
                (
                    "verified_correction"
                    if self.correction_source or self.fault_verified
                    else "verified_nominal"
                )
                if verified_success
                else "verified_stop" if verified_stop else "context_only"
            ),
            "reason": reason,
            "action_timing": "observation_t then target_action_t then 10 native physics steps",
            "privileged_policy_inputs": False,
            "control_clock_enforced": True,
        }
        if self.correction_source is not None:
            metadata["correction_source"] = self.correction_source
        if self.collection_case is not None:
            metadata["collection_case"] = self.collection_case
            metadata["teacher_induced_fault"] = self.fault_verified
        (self.directory / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.closed = True
