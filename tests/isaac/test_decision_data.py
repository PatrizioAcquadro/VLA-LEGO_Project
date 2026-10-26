"""Decision negatives change the goal, never source action labels or family membership."""

import json

import pytest

from vla_lego.isaac.campaign import load_campaign
from vla_lego.isaac.campaign_design import build_declaration
from vla_lego.pretrained.decision_data import (
    counterfactual_instruction,
    episode_sequences,
    observation_grid,
)


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
