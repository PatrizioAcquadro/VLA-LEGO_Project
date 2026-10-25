"""Pilot leakage, visible inventory and causal annotation contracts."""

import json

import numpy as np
import pytest

from vla_lego.isaac.pilot import PilotCase, piece
from vla_lego.isaac.recording import EpisodeRecorder


def case_row():
    return {
        "case": "T1-N1",
        "split": "train",
        "family": "T1",
        "category": "nominal",
        "order": "red CS > blue CL > green PS > yellow PL > purple CS",
        "distractors": "red CL; yellow CL; orange PS",
        "stabilizer": "left",
        "cooperative_placement": "4",
        "variation": "nominal spread",
        "error_target_level": "",
    }


def test_missing_piece_is_replaced_without_an_inventory_count_shortcut():
    row = dict(
        case_row(), case="T1-S", category="stop", missing_piece="red CS", replacement="red PS"
    )
    task = PilotCase(row, 9).task()
    assert len(task.placements) == 8
    assert not task.requested_pieces_available
    assert "small red cube" in task.instruction
    assert "absent" not in task.instruction


@pytest.mark.parametrize("reserved", ["blue PS", "purple PL", "yellow CS", "green CL"])
def test_reserved_combinations_are_rejected_in_any_role(reserved):
    with pytest.raises(ValueError, match="Reserved"):
        piece(reserved)


def test_stage_and_error_mask_are_separate_from_policy_signals(tmp_path):
    recorder = EpisodeRecorder(
        tmp_path / "episode", "instruction", family="T1", split="train", task_decisions=True
    )
    recorder.stage = "level3_wrong_pick"
    recorder.positive_imitation = False
    observation = {
        "joint_names": ("joint",),
        "timestamp_s": 0.0,
        "image_timestamp_s": 0.0,
        "joint_position": np.zeros(1),
        "joint_velocity": np.zeros(1),
        "head_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    recorder.record(
        observation, [0.1], ("joint",), [{"hidden": 1}], requested_action=[0.1], command_time_s=0.0
    )
    recorder.finish(verified_success=False, reason="preserved error")
    frame = json.loads((tmp_path / "episode/frames.jsonl").read_text())
    annotation = json.loads((tmp_path / "episode/annotations.jsonl").read_text())
    assert not {"stage", "positive_imitation", "hidden"} & frame.keys()
    assert frame["action"] == [0.1]
    assert annotation == {"frame": 0, "stage": "level3_wrong_pick", "positive_imitation": False}


def test_recovery_cannot_be_admitted_without_a_physical_fault(tmp_path):
    recorder = EpisodeRecorder(
        tmp_path / "episode",
        "instruction",
        family="T1",
        split="train",
        collection_case={"category": "recovery"},
    )
    with pytest.raises(ValueError, match="physically verified"):
        recorder.finish(verified_success=True, reason="a nominal result is insufficient")
    recorder.finish(verified_success=False, reason="no fault")


@pytest.mark.parametrize("ordinal", [0, 1, 2, 23, 25, 45])
def test_marker_free_instruction_and_site_are_free_of_destination_cues(ordinal):
    task = PilotCase(case_row(), ordinal).task()
    assert not any(word in task.instruction for word in ("square", "marked", "left", "right"))
    assert task.build_region_xy_m is not None
    x, y = task.assembly_xy_m
    assert 0.305 <= x <= 0.445 and -0.125 <= y <= 0.125
    assert all(
        np.linalg.norm(np.array(task.assembly_xy_m) - p.xy_m) > 0.09 for p in task.placements
    )


def test_marked_corpus_cannot_be_admitted_as_marker_free(tmp_path):
    from vla_lego.isaac.pilot_audit import audit_episode

    (tmp_path / "recording").mkdir()
    (tmp_path / "report.json").write_text('{"passed": true}')
    (tmp_path / "recording/episode.json").write_text("{}")
    with pytest.raises(ValueError, match="Previous marked"):
        audit_episode(tmp_path, PilotCase(case_row(), 0))


def test_free_region_default_instruction_does_not_reference_a_marker():
    from dataclasses import replace

    from vla_lego.isaac.task import initial_tower_task

    task = replace(initial_tower_task(), build_region_xy_m=(0.30, 0.45, -0.13, 0.13))
    assert "clear, suitable place" in task.instruction
    assert "square" not in task.instruction


@pytest.mark.parametrize("split", ["train", "development"])
def test_recorded_case_retains_its_split_when_used_for_diagnosis(tmp_path, split):
    from dataclasses import asdict

    from vla_lego.isaac.pilot_scenario import recorded_scenario

    task = PilotCase(case_row(), 0).task()
    directory = tmp_path / "attempts" / "case-001" / "recording"
    directory.mkdir(parents=True)
    (tmp_path / f"{split}-episodes.json").write_text(
        json.dumps([{"case": "case", "path": "attempts/case-001/report.json"}])
    )
    metadata = {
        "split": split,
        "collection_condition": "marker_free_continuous_site_v1",
        "instruction": task.instruction,
    }
    (directory / "episode.json").write_text(json.dumps(metadata))
    (directory.parent / "report.json").write_text(json.dumps({"task": asdict(task)}))
    restored, _, _ = recorded_scenario(tmp_path, "case", split=split)
    assert restored.instruction == task.instruction
    (directory / "episode.json").write_text(json.dumps(dict(metadata, split="other")))
    with pytest.raises(ValueError, match="requested split"):
        recorded_scenario(tmp_path, "case", split=split)


def test_tower_policy_replay_uses_applied_commands_and_rejects_teacher_initialization(tmp_path):
    from dataclasses import asdict
    from types import SimpleNamespace

    import torch

    from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS
    from vla_lego.isaac.policy_replay import replay_prefix

    task = PilotCase(case_row(), 0).task()
    names = [*ARM_JOINTS, *JAW_JOINTS]
    data = SimpleNamespace(
        joint_names=names,
        joint_pos=SimpleNamespace(torch=torch.zeros(1, 16)),
        joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-3.0, 3.0]] * 16])),
    )
    truth = {"block_pose_w": [0.4, 0.0, 0.82, 0.0, 0.0, 0.0, 1.0]}

    class Environment:
        robot = SimpleNamespace(data=data)
        placements = task.placements
        blocks = list(range(len(task.placements)))
        time = 2.0

        def capture(self):
            pass

        def set_joint_targets(self, targets):
            self.targets = [targets[name] for name in names]

        def step(self, count):
            assert count == 10
            self.robot.data.joint_pos.torch[:] = torch.tensor(self.targets)
            self.time += 0.05

        def task_truth(self, index):
            return truth

    metadata = {
        "kind": "autonomous_tower_diagnostic",
        "learned": True,
        "teacher_commands": False,
        "autonomy_started": True,
        "robot_joint_names": names,
        "executed_ticks": 2,
        "head_capture_hz": 20,
        "teacher_prefix_ticks": 0,
        "task": asdict(task),
    }
    (tmp_path / "report.json").write_text(json.dumps(metadata))
    rows = []
    for tick in range(2):
        command = np.asarray([0.1 * (tick + 1)] * 14 + [0.03, 0.03], dtype=np.float32).tolist()
        rows.append(
            {
                "tick": tick,
                "time_s": 2.05 + tick * 0.05,
                "decision": "continue",
                "learned_action": [999] * 18,
                "applied_joint_target": command,
                "robot": {"joint_position": command},
                "blocks": [truth] * len(task.placements),
            }
        )
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = replay_prefix(Environment(), tmp_path, 2)
    assert result["exact_native_parity"] and not result["object_state_writes"]
    assert result["prefix_ticks"] == 2
    metadata["teacher_prefix_ticks"] = 1
    (tmp_path / "report.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="autonomous-from-start"):
        replay_prefix(Environment(), tmp_path, 1)
