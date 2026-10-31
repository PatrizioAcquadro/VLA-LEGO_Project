"""Essential masked-window and privileged-input boundaries for later adaptation."""

import json

import numpy as np
import pytest
from PIL import Image

from vla_lego.pretrained.pilot_dataset import PilotEpisode


def recording(tmp_path, *, admitted=True):
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "head.png")
    metadata = {
        "verified_success": admitted,
        "supervision_role": "verified_correction" if admitted else "context_only",
        "frames": 3,
        "instruction": "Build the instructed tower.",
    }
    (tmp_path / "episode.json").write_text(json.dumps(metadata))
    rows = [
        {
            "head_rgb": "head.png",
            "joint_position": [0] * 18,
            "joint_velocity": [0] * 18,
            "action": [i + 1] * 16,
            "task_decision": "complete" if i == 2 else "continue",
        }
        for i in range(3)
    ]
    (tmp_path / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "annotations.jsonl").write_text(
        "".join(
            json.dumps({"positive_imitation": i != 1, "stage": "hidden teacher stage"}) + "\n"
            for i in range(3)
        )
    )


def test_future_fault_ends_positive_chunk_and_correction_uses_its_own_observation(tmp_path):
    recording(tmp_path)
    episode = PilotEpisode(tmp_path, horizon=4)
    assert len(episode) == 2
    before, correction = episode[0], episode[1]
    assert before["action_is_pad"].tolist() == [False, True, True, True]
    assert before["action"][0, 0] == 1
    assert correction["action"][0, 0] == 3
    assert correction["action"][0, -2:].tolist() == [1, 0]
    assert correction["observation.state"].shape == (32,)
    assert correction["observation.images.head"].shape == (3, 8, 8)
    assert not {"stage", "task_decision", "blocks", "positive_imitation"} & before.keys()


def test_failed_attempt_cannot_be_loaded_as_positive_data(tmp_path):
    recording(tmp_path, admitted=False)
    with pytest.raises(ValueError, match="Only admitted"):
        PilotEpisode(tmp_path)


def test_motor_only_samples_exclude_terminal_targets_but_keep_error_masks(tmp_path):
    recording(tmp_path)
    episode = PilotEpisode(tmp_path, horizon=4, include_decisions=False)
    correction = episode[1]
    assert correction["action"].shape == (4, 16)
    assert correction["action"][0].tolist() == [3.0] * 16
    assert episode[0]["action_is_pad"].tolist() == [False, True, True, True]


def test_future_offsets_share_the_current_observation_anchor_and_stop_at_errors(tmp_path):
    recording(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    rows[0]["joint_position"] = [0.25] * 18
    rows[2]["joint_position"] = [0.5] * 18
    (tmp_path / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    episode = PilotEpisode(tmp_path, horizon=3, action_representation="observation_arm_offset")
    before, after = episode[0], episode[1]
    assert before["action"][0, 0] == 0.75
    assert before["action"][0, 14] == 1.0
    assert after["action"][0, 0] == 2.5
    assert before["action_is_pad"].tolist() == [False, True, True]


def test_offset_chunk_uses_one_anchor_rather_than_each_future_measured_position(tmp_path):
    recording(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    for i, row in enumerate(rows):
        row["joint_position"] = [i * 0.5] * 18
    (tmp_path / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "annotations.jsonl").write_text(
        "".join(json.dumps({"positive_imitation": True, "stage": "move"}) + "\n" for _ in rows)
    )
    episode = PilotEpisode(tmp_path, horizon=2, action_representation="observation_arm_offset")
    # At t=1, q=0.5: targets 2 and3 are both relative to0.5, not future q=1.
    assert episode[1]["action"][:, 0].tolist() == [1.5, 2.5]
    assert episode[1]["action"][:, 14].tolist() == [2.0, 3.0]


def test_conditional_handoff_report_serializes_numpy_measurements(tmp_path):
    from vla_lego.isaac.pilot_scenario import replay_demonstration_handoff

    recording(tmp_path)
    (tmp_path / "annotations.jsonl").write_text(
        "".join(
            json.dumps({"positive_imitation": True, "stage": "move" if i < 2 else "grip"}) + "\n"
            for i in range(3)
        )
    )
    truth = {"block_pose_w": [0.0] * 7}
    (tmp_path / "privileged.jsonl").write_text((json.dumps({"blocks": [truth]}) + "\n") * 3)

    class Environment:
        blocks = [0]

        def capture(self):
            pass

        def set_joint_targets(self, targets):
            assert len(targets) == 16

        def step(self, count):
            assert count == 10

        def observe(self):
            return {"joint_position": np.zeros(18, dtype=np.float32)}

        def task_truth(self, index):
            return truth

    report = replay_demonstration_handoff(Environment(), tmp_path, "grip", lambda _: None)
    assert report["teacher_prefix_ticks"] == 2
    assert not report["autonomous_from_start"]
    assert json.loads(json.dumps(report))["prefix_joint_error_rad"] == 0.0
