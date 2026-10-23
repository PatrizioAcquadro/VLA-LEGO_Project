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
