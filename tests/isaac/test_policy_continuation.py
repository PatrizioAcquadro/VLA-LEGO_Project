"""Correction provenance and physical handoff fixtures, not learned-policy evidence."""

import json

import numpy as np
import pytest

from vla_lego.isaac.environment import TABLE_TOP
from vla_lego.isaac.policy_continuation import released_prefix
from vla_lego.isaac.recording import EpisodeRecorder, verify_policy_correction
from vla_lego.isaac.task import initial_tower_task


def test_current_handoff_cannot_hide_a_held_piece_or_lost_support():
    task = initial_tower_task()
    states = [
        {
            "block_pose_w": [*p.xy_m, TABLE_TOP + p.spec.dimensions_m[2] / 2, 0, 0, 0, 1],
            "block_velocity_w": [0.0] * 6,
            "robot_contact_force_n": 0.0,
            "nonrobot_contact_force_n": 0.3,
            "block_contact_forces_n": [[0.0] * 3 for _ in task.placements],
        }
        for p in task.placements
    ]
    # This fixture uses the original marked evaluator region.
    states[0]["block_pose_w"][:2] = list(task.assembly_xy_m)
    states[1]["block_pose_w"][:2] = list(task.assembly_xy_m)
    states[1]["block_pose_w"][2] += task.targets[0].dimensions_m[2]
    states[1]["block_contact_forces_n"][0][2] = 0.3
    assert released_prefix(task, states) == 2
    states[1]["block_contact_forces_n"][0][2] = 0
    assert released_prefix(task, states) == 1
    states[1]["robot_contact_force_n"] = 0.5
    with pytest.raises(ValueError, match="still holds"):
        released_prefix(task, states)


def test_policy_window_actions_stay_negative_and_cannot_be_declared_teacher_faults(tmp_path):
    directory = tmp_path / "recording"
    recorder = EpisodeRecorder(
        directory,
        "instruction",
        family="TRAIN",
        split="train",
        task_decisions=True,
        collection_case={"category": "recovery", "correction_origin": "policy"},
    )

    def record(time_s):
        observation = {
            "joint_names": ("joint",),
            "timestamp_s": time_s,
            "image_timestamp_s": time_s,
            "joint_position": np.zeros(1),
            "joint_velocity": np.zeros(1),
            "head_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        recorder.record(
            observation, [0.1], ("joint",), [], requested_action=[0.1], command_time_s=time_s
        )

    record(0)
    recorder.action_source = "policy"
    with pytest.raises(ValueError, match="positive teacher targets"):
        record(0.05)
    recorder.positive_imitation = False
    record(0.05)
    source = {
        "mode": "in_place_policy_window_v1",
        "start_frame": 1,
        "end_frame": 2,
        "teacher_prefix_frames": 1,
        "executed_motor_commands": 1,
        "physically_verified": True,
        "conditional_start": True,
        "autonomous_from_start": False,
        "object_state_writes": False,
        "teacher_commands_during_policy_window": False,
        "policy": {"checkpoint": "actual-component", "execution_horizon": 10, "jaw_overlap": True},
    }
    recorder.admit_policy_intervention(source)
    recorder.action_source, recorder.positive_imitation = "teacher", True
    record(0.1)
    recorder.finish(verified_success=True, reason="isolated recording contract fixture")
    metadata = json.loads((directory / "episode.json").read_text())
    assert metadata["fault_origin"] == "policy" and not metadata["teacher_induced_fault"]
    assert "exact_native_parity" not in metadata["correction_source"]
    (tmp_path / "policy-window.jsonl").write_text(
        json.dumps({"frame": 1, "applied": True, "decision": "continue"}) + "\n"
    )
    verify_policy_correction(directory, metadata)
    metadata["teacher_induced_fault"] = True
    with pytest.raises(ValueError, match="real in-place"):
        verify_policy_correction(directory, metadata)
    metadata["teacher_induced_fault"] = False
    (tmp_path / "policy-window.jsonl").write_text("")
    with pytest.raises(ValueError, match="actual applied"):
        verify_policy_correction(directory, metadata)


def test_a_correct_placement_or_held_piece_cannot_be_labeled_a_policy_drop():
    from copy import deepcopy
    from types import SimpleNamespace

    from vla_lego.isaac.policy_continuation import PolicyIntervention

    task = initial_tower_task()
    after = [
        {
            "block_pose_w": [*p.xy_m, TABLE_TOP + p.spec.dimensions_m[2] / 2, 0, 0, 0, 1],
            "block_velocity_w": [0.0] * 6,
            "robot_contact_force_n": 0.0,
            "nonrobot_contact_force_n": 0.3,
            "block_contact_forces_n": [[0.0] * 3 for _ in task.placements],
        }
        for p in task.placements
    ]
    after[0]["block_pose_w"][:2] = list(task.assembly_xy_m)
    before = deepcopy(after)
    before[1]["block_pose_w"][2] += 0.1
    before[1]["robot_contact_force_n"] = 1
    admitted = []
    recorder = SimpleNamespace(admit_policy_intervention=admitted.append)
    env = SimpleNamespace(task_truth=lambda i: before[i], time=10, recorder=recorder)
    reference = SimpleNamespace(env=env)
    case = SimpleNamespace(
        row={"correction_origin": "policy", "mechanism": "slip_drop", "partition": "train"},
        error_level=1,
        cooperative_level=3,
    )
    intervention = PolicyIntervention.__new__(PolicyIntervention)
    intervention._window = lambda *a, **k: ({"executed_motor_commands": 1}, before, after, False)
    event = intervention.slip_drop(reference, case, task, 1)
    assert not event["teacher_induced"] and event["physically_verified"] and len(admitted) == 1
    assert event["fall_m"] == pytest.approx(0.1)
    after[1]["block_pose_w"][:2] = list(task.assembly_xy_m)
    after[1]["block_pose_w"][2] += task.targets[0].dimensions_m[2]
    after[1]["nonrobot_contact_force_n"] = 0
    after[1]["block_contact_forces_n"][0][2] = 0.3
    with pytest.raises(RuntimeError, match="did not produce"):
        intervention.slip_drop(reference, case, task, 1)
    after[1]["robot_contact_force_n"] = 1
    with pytest.raises(RuntimeError, match="did not produce"):
        intervention.slip_drop(reference, case, task, 1)
    assert len(admitted) == 1
