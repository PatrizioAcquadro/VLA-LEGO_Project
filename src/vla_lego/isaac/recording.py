"""Causal sensor/action recording with privileged evidence kept in a separate file."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from vla_lego.pretrained.rgb_video import RGBVideoWriter


def verify_policy_correction(directory: Path, metadata: dict) -> None:
    """Actual recorded command/mask boundaries, not a policy-origin declaration."""
    source = metadata.get("correction_source", {})
    if (
        metadata.get("fault_origin") != "policy"
        or metadata.get("teacher_induced_fault") is not False
        or metadata["split"] != "train"
        or source.get("mode") != "in_place_policy_window_v1"
        or not source.get("physically_verified")
        or source.get("object_state_writes", True)
        or source.get("teacher_commands_during_policy_window", True)
        or source.get("autonomous_from_start", True)
        or not source.get("conditional_start")
    ):
        raise ValueError("Policy-origin metadata lacks a real in-place conditional intervention")
    start, end = source["start_frame"], source["end_frame"]
    if not 0 < start < end < metadata["frames"]:
        raise ValueError(
            "A policy correction needs preceding context and subsequent teacher actions"
        )
    rows = [json.loads(line) for line in (directory / "annotations.jsonl").open()]
    window = rows[start:end]
    if (
        len(rows) != metadata["frames"]
        or source["executed_motor_commands"] != len(window)
        or [row["frame"] for row in window] != list(range(start, end))
        or any(row.get("action_source") != "policy" or row["positive_imitation"] for row in window)
        or any(row.get("action_source") == "policy" for row in rows[:start] + rows[end:])
        or not any(row["positive_imitation"] for row in rows[end:])
    ):
        raise ValueError("Policy correction command origins and imitation masks differ")
    commands = [
        json.loads(line)
        for line in (directory.parent / "policy-window.jsonl").open()
        if line.strip()
    ]
    applied = [row for row in commands if row["applied"]]
    if [row["frame"] for row in applied] != list(range(start, end)) or any(
        row["decision"] != "continue" for row in applied
    ):
        raise ValueError("Policy correction lacks the actual applied prediction stream")


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
        min_free_bytes: int = 0,
        image_storage: str = "png",
    ):
        if not family or split not in ("train", "development"):
            raise ValueError(
                "Collection requires a generation family and a train/development split."
            )
        if correction_source is not None and not correction_source.get("exact_native_parity"):
            raise ValueError("A corrective continuation requires exact native prefix replay.")
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        if image_storage not in ("png", "ffv1"):
            raise ValueError("Unknown RGB recording storage")
        if image_storage == "png":
            (directory / "head").mkdir()
        self.image_storage = image_storage
        self.video: RGBVideoWriter | None = None
        self.video_error: str | None = None
        self.frames = (directory / "frames.jsonl").open("w")
        self.evidence = (directory / "privileged.jsonl").open("w")
        self.annotations = (directory / "annotations.jsonl").open("w")
        self.stage = "initial_context"
        self.positive_imitation = True
        self.action_source = "teacher"
        self.policy_frames = 0
        self.instruction = instruction
        self.family, self.split = family, split
        self.correction_source = correction_source
        self.task_decisions = task_decisions
        self.collection_case = collection_case
        self.collection_condition = collection_condition
        if not 0 <= png_compress_level <= 9:
            raise ValueError("PNG compression must be in0..9")
        self.png_compress_level = png_compress_level
        self.min_free_bytes = min_free_bytes
        self.fault_verified = False
        self.fault_origin = "teacher"
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
        if self.action_source not in ("teacher", "policy") or (
            self.action_source == "policy" and self.positive_imitation
        ):
            raise ValueError("Unverified policy-window actions cannot be positive teacher targets")
        if self.count % 20 == 0 and shutil.disk_usage(self.directory).free < self.min_free_bytes:
            raise OSError("Insufficient free space for the campaign's bounded recording staging")
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
        if self.image_storage == "ffv1":
            image_path = "head.mkv"
            if self.video is None:
                height, width = observation["head_rgb"].shape[:2]
                self.video = RGBVideoWriter(self.directory / image_path, width, height)
            self.video.write(observation["head_rgb"])
        else:
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
        if self.image_storage == "ffv1":
            row["head_frame"] = self.count
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
                    "action_source": self.action_source,
                }
            )
            + "\n"
        )
        if self.action_source == "policy":
            self.policy_frames += 1
        self.count += 1
        if self.count % 20 == 0:
            self.frames.flush()
            self.evidence.flush()
            self.annotations.flush()

    def admit_policy_intervention(self, source: dict) -> None:
        """Admit an in-place observed error; it is not an exact replay or teacher fault."""
        start, end = source["start_frame"], source["end_frame"]
        if (
            source.get("mode") != "in_place_policy_window_v1"
            or self.split != "train"
            or self.collection_case is None
            or self.collection_case.get("correction_origin") != "policy"
            or start < 1
            or end != self.count
            or end - start != self.policy_frames
            or self.policy_frames != source["executed_motor_commands"]
            or self.policy_frames < 1
            or source["teacher_prefix_frames"] != start
            or not source["physically_verified"]
            or source["policy"]["execution_horizon"] != 10
            or not source["policy"]["jaw_overlap"]
            or not source["policy"]["checkpoint"]
        ):
            raise ValueError(
                "Policy correction lacks actual masked in-place commands and error evidence"
            )
        self.correction_source = source
        self.fault_verified = True
        self.fault_origin = "policy"

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
        if (
            verified_success
            and self.collection_case is not None
            and self.collection_case.get("correction_origin") == "policy"
            and (
                self.fault_origin != "policy"
                or self.correction_source is None
                or self.policy_frames < 1
            )
        ):
            raise ValueError(
                "A policy-origin case cannot be completed with an induced teacher fault"
            )
        self.frames.close()
        self.evidence.close()
        self.annotations.close()
        if self.video is not None:
            try:
                self.video.close()
            except Exception as error:
                self.video_error = repr(error)
                verified_success, verified_stop = False, False
                reason = "RGB encoding failed: " + self.video_error
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
            metadata["teacher_induced_fault"] = (
                self.fault_verified and self.fault_origin == "teacher"
            )
            metadata["fault_origin"] = self.fault_origin if self.fault_verified else None
        if self.video is not None:
            metadata.update(
                image_storage="ffv1_bgr0_lossless",
                image_file="head.mkv",
                image_frames=self.video.frames,
                rgb_sha256=self.video.sha256.hexdigest(),
                video_error=self.video_error,
            )
        (self.directory / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.closed = True
