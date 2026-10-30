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


def short_single_recording(tmp_path):
    from vla_lego.isaac.environment import TABLE_TOP

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
            "task_decision": "complete" if i == 60 else "continue",
        }
        for i in range(61)
    ]
    metadata = {
        "frames": len(frames),
        "collection_case": case.row,
        "generation_family": case.row["family"],
        "verified_success": True,
        "privileged_policy_inputs": False,
        "instruction": case.task().instruction,
    }
    (directory / "episode.json").write_text(json.dumps(metadata))
    (directory / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
    task = case.task()
    target = next(i for i, p in enumerate(task.placements) if p.spec == task.targets[0])
    traces = []
    for tick in range(len(frames)):
        held = tick in (1, 2) or tick == 60
        blocks = []
        for i, placement in enumerate(task.placements):
            xy = task.assembly_xy_m if i == target and tick >= 2 else placement.xy_m
            z = (
                TABLE_TOP
                + placement.spec.dimensions_m[2] / 2
                + (0.1 if i == target and held else 0)
            )
            blocks.append(
                {
                    "block_pose_w": [*xy, z, 0, 0, 0, 1],
                    "block_velocity_w": [0.0] * 6,
                    "robot_contact_forces_n": [
                        [0, 0, 0.6 if i == target and held and j < 2 else 0] for j in range(4)
                    ],
                    "robot_contact_force_n": 1.2 if i == target and held else 0.0,
                    "nonrobot_contact_force_n": 0.0 if i == target and held else 0.2,
                }
            )
        traces.append(
            {
                "time_s": (tick + 1) / 20,
                "blocks": blocks,
                "tools": {"left": [0.1, 0.6, 1.1], "right": [0.1, -0.6, 1.1]},
                "robot": {
                    "robot_contact_paths": [
                        f"/{side}_WSG32_{jaw}_JAW_LINK"
                        for side in ("left", "right")
                        for jaw in ("NEGATIVE", "POSITIVE")
                    ]
                },
            }
        )
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in traces))
    entry = {
        "recording": "recording",
        "case": case.name,
        "family": case.row["family"],
        "partition": "train",
        "admitted": True,
    }
    return case, directory, frames, trace, traces, entry


def test_short_single_labels_use_current_physical_dwell_without_future_or_raw_rewrite(tmp_path):
    case, directory, frames, trace, traces, entry = short_single_recording(tmp_path)
    raw = (directory / "frames.jsonl").read_bytes()
    original, negative = episode_sequences(tmp_path, entry, case)
    positive_times = [
        t for t, label in zip(original.times, original.labels, strict=True) if label == 1
    ]
    assert positive_times and positive_times[0] >= 2.2
    assert positive_times[-1] - positive_times[0] >= 0.5
    assert original.labels[-1] == 1  # The unobserved next command holds the piece again.
    assert negative.family == original.family and set(negative.labels) == {0}
    assert (directory / "frames.jsonl").read_bytes() == raw
    traces[0]["time_s"] += 0.05
    trace.write_text("".join(json.dumps(row) + "\n" for row in traces))
    with pytest.raises(ValueError, match="observation time"):
        episode_sequences(tmp_path, entry, case)


def test_short_single_final_held_state_cannot_become_a_completion_label(tmp_path):
    from vla_lego.pretrained.decision_data import single_block_completion_labels

    case, directory, frames, trace, traces, _entry = short_single_recording(tmp_path)
    traces[-2]["blocks"] = traces[-1]["blocks"]
    trace.write_text("".join(json.dumps(row) + "\n" for row in traces))
    with pytest.raises(ValueError, match="do not establish"):
        single_block_completion_labels(directory, case, frames)
