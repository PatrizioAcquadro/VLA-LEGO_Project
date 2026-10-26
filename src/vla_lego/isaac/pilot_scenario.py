"""Restore the collected development condition without exposing it to the policy."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vla_lego.isaac.catalog import BlockPlacement, BlockSpec
from vla_lego.isaac.task import TowerTask


def recorded_scenario(
    root: Path, name: str, *, split: str = "development"
) -> tuple[TowerTask, dict, Path]:
    if split not in ("train", "development"):
        raise ValueError("Unknown recorded pilot split")
    entries = json.loads((root / f"{split}-episodes.json").read_text())
    entry = next(row for row in entries if row["case"] == name)
    directory = root / "attempts" / Path(entry["path"]).parent.name / "recording"
    metadata = json.loads((directory / "episode.json").read_text())
    if metadata["split"] != split or metadata.get("collection_condition") != (
        "marker_free_continuous_site_v1"
    ):
        raise ValueError(
            "Evaluation requires an admitted marker-free case from the requested split"
        )
    report = json.loads((directory.parent / "report.json").read_text())
    source = report["task"]
    task = TowerTask(
        placements=tuple(
            BlockPlacement(BlockSpec(**row["spec"]), tuple(row["xy_m"]), row["yaw_deg"])
            for row in source["placements"]
        ),
        targets=tuple(BlockSpec(**row) for row in source["targets"]),
        assembly_xy_m=tuple(source["assembly_xy_m"]),
        instruction_text=metadata["instruction"],
        build_region_xy_m=tuple(source["build_region_xy_m"]),
    )
    if len(task.placements) != 8 or task.build_region_xy_m is None:
        raise ValueError("Pilot evaluation requires eight objects and no destination marker")
    return task, report, directory


def recovery_prefix(directory: Path) -> tuple[list[dict], dict, list[dict]]:
    frames = [json.loads(line) for line in (directory / "frames.jsonl").open()]
    annotations = [json.loads(line) for line in (directory / "annotations.jsonl").open()]
    boundary = next(
        i
        for i in range(1, len(annotations))
        if annotations[i]["positive_imitation"] and not annotations[i - 1]["positive_imitation"]
    )
    with (directory / "privileged.jsonl").open() as stream:
        expected = next(
            json.loads(line)["blocks"] for i, line in enumerate(stream) if i == boundary
        )
    return frames[:boundary], frames[boundary], expected


def recovered_fault_is_present(task: TowerTask, report: dict, truth: list[dict]) -> bool:
    from vla_lego.isaac.environment import TABLE_TOP

    fault = report["fault"]
    kind = fault["type"]
    if kind == "slip_drop":
        level = fault["target_level"] - 1
        index = next(i for i, p in enumerate(task.placements) if p.spec == task.targets[level])
        state = truth[index]
        return bool(
            fault["fall_m"] >= 0.07
            and abs(
                state["block_pose_w"][2]
                - TABLE_TOP
                - task.placements[index].spec.dimensions_m[2] / 2
            )
            < 0.01
            and state["robot_contact_force_n"] < 1e-6
            and state["nonrobot_contact_force_n"] > 1e-6
        )
    if kind == "interrupted_progress":
        completed = fault["completed_levels"]
        if not 1 <= completed < len(task.targets):
            return False
        bottom = TABLE_TOP
        previous = None
        for target in task.targets[:completed]:
            index = next(i for i, p in enumerate(task.placements) if p.spec == target)
            state = truth[index]
            support = (
                state["nonrobot_contact_force_n"]
                if previous is None
                else state["block_contact_forces_n"][previous][2]
            )
            if (
                abs(state["block_pose_w"][2] - bottom - target.dimensions_m[2] / 2) >= 0.003
                or state["robot_contact_force_n"] >= 1e-6
                or support <= 1e-6
            ):
                return False
            bottom += target.dimensions_m[2]
            previous = index
        next_index = next(
            i for i, p in enumerate(task.placements) if p.spec == task.targets[completed]
        )
        return bool(
            abs(
                truth[next_index]["block_pose_w"][2]
                - TABLE_TOP
                - task.targets[completed].dimensions_m[2] / 2
            )
            < 0.01
        )
    if kind in ("missed grasp", "wrong distractor picked"):
        index = fault["object_index"]
        state = truth[index]
        lift = (
            state["block_pose_w"][2] - TABLE_TOP - task.placements[index].spec.dimensions_m[2] / 2
        )
        if kind == "missed grasp":
            return bool(abs(lift) < 0.005 and state["robot_contact_force_n"] < 0.02)
        return bool(lift > 0.02 and state["robot_contact_force_n"] > 0.02)
    if kind != "physically consequential incomplete placement":
        raise ValueError("Unknown recorded recovery condition")
    level = fault["target_level"] - 1
    index = next(i for i, p in enumerate(task.placements) if p.spec == task.targets[level])
    lower = next(i for i, p in enumerate(task.placements) if p.spec == task.targets[level - 1])
    expected_z = TABLE_TOP + sum(p.dimensions_m[2] for p in task.targets[:level])
    expected_z += task.targets[level].dimensions_m[2] / 2
    return bool(
        truth[index]["block_pose_w"][2] - expected_z > 0.005
        and truth[index]["block_contact_forces_n"][lower][2] < 0.02
        and truth[index]["robot_contact_force_n"] > 0.02
    )


def replay_demonstration_handoff(
    env: Any, directory: Path, stage: str, on_tick: Callable[[int], None]
) -> dict:
    """Initialize a conditional diagnostic through actual recorded robot commands."""
    from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS

    metadata = json.loads((directory / "episode.json").read_text())
    if not metadata.get("verified_success"):
        raise ValueError("A diagnostic handoff requires an admitted physical demonstration")
    frames = [json.loads(line) for line in (directory / "frames.jsonl").open()]
    annotations = [json.loads(line) for line in (directory / "annotations.jsonl").open()]
    boundary = next(
        i
        for i, row in enumerate(annotations)
        if row["positive_imitation"] and row["stage"].startswith(stage)
    )
    if boundary < 1:
        raise ValueError("A handoff must follow at least one recorded robot command")
    for tick, row in enumerate(frames[:boundary]):
        env.capture()
        env.set_joint_targets(dict(zip((*ARM_JOINTS, *JAW_JOINTS), row["action"], strict=True)))
        env.step(10)
        on_tick(tick)
    with (directory / "privileged.jsonl").open() as stream:
        expected = next(
            json.loads(line)["blocks"] for i, line in enumerate(stream) if i == boundary
        )
    position_error = max(
        sum(
            (a - b) ** 2
            for a, b in zip(actual["block_pose_w"][:3], prior["block_pose_w"][:3], strict=True)
        )
        ** 0.5
        for actual, prior in zip(
            [env.task_truth(i) for i in range(len(env.blocks))], expected, strict=True
        )
    )
    env.capture()
    joint_error = float(
        max(
            abs(a - b)
            for a, b in zip(
                env.observe()["joint_position"], frames[boundary]["joint_position"], strict=True
            )
        )
    )
    if position_error > 0.01 or joint_error > 0.005:
        raise RuntimeError("Demonstration handoff does not reproduce its recorded physical state")
    return {
        "teacher_prefix_ticks": boundary,
        "teacher_handoff_stage": stage,
        "teacher_commands_in_prefix": True,
        "conditional_diagnostic": True,
        "autonomous_from_start": False,
        "prefix_object_position_error_m": position_error,
        "prefix_joint_error_rad": joint_error,
    }


def replay_recovery(
    env: Any, task: TowerTask, report: dict, directory: Path, on_tick: Callable[[int], None]
) -> dict:
    """Execute the recorded physical prefix, then verify the actual error state."""
    import numpy as np

    from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS

    frames, expected_observation, expected_truth = recovery_prefix(directory)
    for tick, row in enumerate(frames):
        env.set_joint_targets(dict(zip((*ARM_JOINTS, *JAW_JOINTS), row["action"], strict=True)))
        env.step(10)
        on_tick(tick)
    truth = [env.task_truth(i) for i in range(len(env.blocks))]
    position_error = max(
        float(np.linalg.norm(np.asarray(a["block_pose_w"][:3]) - b["block_pose_w"][:3]))
        for a, b in zip(truth, expected_truth, strict=True)
    )
    fault_present = recovered_fault_is_present(task, report, truth)
    if not fault_present or position_error > 0.01:
        raise RuntimeError("Recorded prefix did not establish the intended physical recovery state")
    env.capture()
    return {
        "teacher_prefix_ticks": len(frames),
        "recovery_initialization": True,
        "prefix_object_position_error_m": position_error,
        "prefix_joint_error_rad": float(
            np.max(
                np.abs(
                    env.observe()["joint_position"][:16]
                    - expected_observation["joint_position"][:16]
                )
            )
        ),
        "recovery_fault_present": fault_present,
        "teacher_commands_in_prefix": True,
        "conditional_diagnostic": True,
        "autonomous_from_start": False,
    }
