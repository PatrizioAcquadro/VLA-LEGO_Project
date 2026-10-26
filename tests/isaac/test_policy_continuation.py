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


@pytest.mark.parametrize("behavior", ["waiting", "successful_lift", "empty_lift"])
def test_only_a_physical_empty_close_and_lift_qualifies_as_a_miss(behavior):
    from types import SimpleNamespace

    from vla_lego.isaac.policy_continuation import PolicyIntervention

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
    states[0]["block_pose_w"][:2] = list(task.assembly_xy_m)
    admitted = []
    env = SimpleNamespace(
        blocks=states,
        task_truth=lambda i: states[i],
        time=10,
        recorder=SimpleNamespace(admit_policy_intervention=admitted.append),
        robot=SimpleNamespace(
            data=SimpleNamespace(
                joint_names=["left_WSG32_JAW_OPENING", "right_WSG32_JAW_OPENING"],
                joint_pos=SimpleNamespace(torch=np.array([[0.005, 0.034]])),
            )
        ),
    )
    tool = np.array([[*states[1]["block_pose_w"][:2], states[1]["block_pose_w"][2] + 0.06]])
    reference = SimpleNamespace(env=env, ctrl=SimpleNamespace(pose=lambda side: tool))
    case = SimpleNamespace(
        row={"correction_origin": "policy", "mechanism": "missed_grasp", "partition": "train"},
        error_level=1,
        cooperative_level=3,
    )

    def window(*args, stop_when, **kwargs):
        assert stop_when(states) is None
        if behavior != "waiting":
            tool[0, 2] += 0.08
        if behavior == "successful_lift":
            states[1]["block_pose_w"][2] += 0.08
            states[1]["robot_contact_force_n"] = 1
        event = stop_when(states)
        return {"executed_motor_commands": 2, "observed_error": event}, states, states, False

    intervention = PolicyIntervention.__new__(PolicyIntervention)
    intervention._window = window
    if behavior == "empty_lift":
        event = intervention.missed_grasp(reference, case, task, 1)
        assert not event["teacher_induced"] and event["physically_verified"]
        assert len(admitted) == 1
    else:
        with pytest.raises(RuntimeError, match="did not produce"):
            intervention.missed_grasp(reference, case, task, 1)
        assert not admitted


@pytest.mark.parametrize("behavior", ["correct_piece", "wrong_unilateral", "wrong_bilateral"])
def test_wrong_piece_intervention_requires_the_actual_incorrect_bilateral_lift(behavior):
    from copy import deepcopy
    from types import SimpleNamespace

    from vla_lego.isaac.policy_continuation import PolicyIntervention

    task = initial_tower_task()
    states = [
        {
            "block_pose_w": [*p.xy_m, TABLE_TOP + p.spec.dimensions_m[2] / 2, 0, 0, 0, 1],
            "block_velocity_w": [0.0] * 6,
            "robot_contact_force_n": 0.0,
            "robot_contact_forces_n": [[0.0] * 3 for _ in range(4)],
            "nonrobot_contact_force_n": 0.3,
            "block_contact_forces_n": [[0.0] * 3 for _ in task.placements],
        }
        for p in task.placements
    ]
    states[0]["block_pose_w"][:2] = list(task.assembly_xy_m)
    admitted = []
    env = SimpleNamespace(
        blocks=states,
        time=10,
        task_truth=lambda i: deepcopy(states[i]),
        recorder=SimpleNamespace(admit_policy_intervention=admitted.append),
        contact_robot_paths=[
            f"/{side}_WSG32_{jaw}_JAW_LINK"
            for side in ("left", "right")
            for jaw in ("NEGATIVE", "POSITIVE")
        ],
    )
    case = SimpleNamespace(
        row={"correction_origin": "policy", "mechanism": "wrong_piece", "partition": "train"},
        error_level=1,
        cooperative_level=3,
    )
    lifted = 1 if behavior == "correct_piece" else 2

    def window(*args, stop_when, **kwargs):
        states[lifted]["block_pose_w"][2] += 0.08
        states[lifted]["robot_contact_force_n"] = 1
        states[lifted]["robot_contact_forces_n"][0][0] = 0.6
        if behavior != "wrong_unilateral":
            states[lifted]["robot_contact_forces_n"][1][0] = 0.6
        states[lifted]["nonrobot_contact_force_n"] = 0
        event = stop_when(states)
        return {"executed_motor_commands": 2, "observed_error": event}, states, states, False

    intervention = PolicyIntervention.__new__(PolicyIntervention)
    intervention._window = window
    if behavior == "wrong_bilateral":
        event = intervention.wrong_piece(SimpleNamespace(env=env), case, task, 1)
        assert event["object_index"] == lifted and event["side"] == "left"
        assert not event["teacher_induced"] and len(admitted) == 1
    else:
        with pytest.raises(RuntimeError, match="did not produce"):
            intervention.wrong_piece(SimpleNamespace(env=env), case, task, 1)
        assert not admitted


@pytest.mark.parametrize(
    "behavior", ["waiting", "supported", "still_descending", "failed_transfer"]
)
def test_placement_failure_excludes_transport_pauses_and_supported_placements(behavior):
    from copy import deepcopy
    from types import SimpleNamespace

    from vla_lego.isaac.policy_continuation import PolicyIntervention

    task = initial_tower_task()
    states = [
        {
            "block_pose_w": [*p.xy_m, TABLE_TOP + p.spec.dimensions_m[2] / 2, 0, 0, 0, 1],
            "block_velocity_w": [0.0] * 6,
            "robot_contact_force_n": 0.0,
            "robot_contact_forces_n": [[0.0] * 3 for _ in range(4)],
            "nonrobot_contact_force_n": 0.3,
            "block_contact_forces_n": [[0.0] * 3 for _ in task.placements],
        }
        for p in task.placements
    ]
    for state in states[:2]:
        state["block_pose_w"][:2] = list(task.assembly_xy_m)
    states[1]["block_pose_w"][2] += task.targets[0].dimensions_m[2] + 0.08
    states[1]["robot_contact_force_n"] = 1
    states[1]["nonrobot_contact_force_n"] = 0
    states[1]["robot_contact_forces_n"][0][0] = 0.6
    states[1]["robot_contact_forces_n"][1][0] = 0.6
    admitted = []
    env = SimpleNamespace(
        blocks=states,
        time=10,
        task_truth=lambda i: deepcopy(states[i]),
        recorder=SimpleNamespace(admit_policy_intervention=admitted.append),
        contact_robot_paths=[
            f"/{side}_WSG32_{jaw}_JAW_LINK"
            for side in ("left", "right")
            for jaw in ("NEGATIVE", "POSITIVE")
        ],
    )
    case = SimpleNamespace(
        row={"correction_origin": "policy", "mechanism": "placement", "partition": "train"},
        error_level=1,
    )

    def window(*args, stop_when, **kwargs):
        stop_when(states)
        if behavior != "waiting":
            states[1]["block_pose_w"][2] -= 0.03
        if behavior == "supported":
            states[1]["block_pose_w"][2] -= 0.05
            states[1]["block_contact_forces_n"][0][2] = 0.3
        env.time = 75
        stop_when(states)
        if behavior == "still_descending":
            states[1]["block_pose_w"][2] -= 0.02
        env.time = 80
        stop_when(states)
        return (
            {
                "executed_motor_commands": 1400,
                "source_terminal_decision": "budget",
                "start_time_s": 10,
            },
            states,
            states,
            False,
        )

    intervention = PolicyIntervention.__new__(PolicyIntervention)
    intervention._window = window
    if behavior == "failed_transfer":
        event = intervention.placement(SimpleNamespace(env=env), case, task, 1, "left")
        assert not event["teacher_induced"] and event["height_gap_m"] > 0.01
        assert len(admitted) == 1
    else:
        with pytest.raises(RuntimeError, match="did not establish"):
            intervention.placement(SimpleNamespace(env=env), case, task, 1, "left")
        assert not admitted
