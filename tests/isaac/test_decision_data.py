"""Decision negatives change the goal, never source action labels or family membership."""

import json

import pytest
from PIL import Image

from vla_lego.isaac.campaign import load_campaign
from vla_lego.isaac.campaign_design import build_declaration
from vla_lego.isaac.campaign_faults import stop_evidence
from vla_lego.pretrained.decision_data import (
    counterfactual_instruction,
    episode_sequences,
    observation_grid,
)
from vla_lego.pretrained.decision_features import history_samples
from vla_lego.pretrained.pilot_dataset import load_observation


def test_decision_stream_preserves_terminal_frame_and_refuses_noncausal_clocks():
    frames = [{"observation_time_s": i / 20, "image_time_s": i / 20} for i in range(24)]
    assert observation_grid(frames) == [0, 10, 20, 23]
    frames[0]["image_time_s"] = 0.1
    with pytest.raises(ValueError, match="future"):
        observation_grid(frames)
    frames[0]["image_time_s"] = 0
    frames[1]["observation_time_s"] = 0
    frames[1]["image_time_s"] = 0
    with pytest.raises(ValueError, match="advance"):
        observation_grid(frames)


def test_instruction_negatives_work_for_every_feasible_declared_case(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(build_declaration()))
    for case in load_campaign(path):
        if case.category == "stop":
            continue
        negative = counterfactual_instruction(case)
        assert negative != case.declaration["instruction"]
        if case.category != "single_block":
            first = case.task().targets[0].description.replace("prism", "rectangular prism")
            second = case.task().targets[1].description.replace("prism", "rectangular prism")
            assert negative.index(second) < negative.index(first)


def test_derived_sequences_keep_source_family_and_do_not_create_motor_demonstrations(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(build_declaration()))
    case = load_campaign(path)[0]
    directory = tmp_path / "recording"
    directory.mkdir()
    frames = [
        {
            "frame": i,
            "observation_time_s": i / 20,
            "image_time_s": i / 20,
            "task_decision": "continue" if i < 80 else "complete",
        }
        for i in range(101)
    ]
    metadata = {
        "frames": 101,
        "collection_case": case.row,
        "generation_family": case.row["family"],
        "verified_success": True,
        "privileged_policy_inputs": False,
        "instruction": case.task().instruction,
    }
    (directory / "episode.json").write_text(json.dumps(metadata))
    raw = "".join(json.dumps(row) + "\n" for row in frames)
    (directory / "frames.jsonl").write_text(raw)
    entry = {
        "recording": "recording",
        "case": case.name,
        "family": case.row["family"],
        "partition": case.row["partition"],
        "admitted": True,
    }
    original, negative = episode_sequences(tmp_path, entry, case)
    assert original.labels[-1] == 1 and set(negative.labels) == {0}
    assert negative.family == original.family == entry["family"]
    assert negative.partition == original.partition == "train"
    assert original.kind != negative.kind and original.case == negative.case
    assert negative.indices[-1] == 100 and negative.times[0] <= negative.times[-1] - 3
    assert (directory / "frames.jsonl").read_text() == raw


def test_decision_observation_constructor_excludes_labels_actions_and_hidden_fields(tmp_path):
    Image.new("RGB", (2, 2), (20, 30, 40)).save(tmp_path / "head.png")
    row = {
        "joint_position": [0.1] * 18,
        "joint_velocity": [0.2] * 18,
        "head_rgb": "head.png",
        "task_decision": "complete",
        "action": [float("nan")] * 16,
        "teacher_stage": "verified_complete",
        "privileged_progress": 5,
    }
    observed = load_observation(tmp_path, row)
    assert set(observed) == {"observation.state", "observation.images.head"}
    assert observed["observation.state"].shape == (32,)
    assert observed["observation.images.head"].shape == (3, 2, 2)


def test_counterfactual_context_is_not_padded_as_an_artificial_episode_start():
    times = [100 + i / 2 for i in range(9)]
    samples = history_samples(times, "counterfactual_instruction")
    assert [i for i, _ in samples] == [6, 7, 8]
    assert samples[0][1] == list(range(7))
    assert history_samples(times, "recorded_instruction")[0] == (0, [0] * 7)


def test_safe_impossibility_depends_on_present_support_not_teacher_clock(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(build_declaration()))
    case = next(c for c in load_campaign(path) if c.name == "TR-S01-00-0")
    states = [
        {
            "block_pose_w": [0.4, 0.1, 0.82, 0, 0, 0, 1],
            "robot_contact_force_n": 0.0,
            "nonrobot_contact_force_n": 0.2,
            "block_contact_forces_n": [[0, 0, 0]] * 8,
        }
        for _ in range(8)
    ]
    assert stop_evidence(case.task(), case, states)["passed"]
    states[0]["robot_contact_force_n"] = 1.0
    assert not stop_evidence(case.task(), case, states)["passed"]
