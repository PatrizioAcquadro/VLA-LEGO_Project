"""Expanded learning must not start from a declaration or an unfinished corpus."""

import json

import pytest

from vla_lego.pretrained.campaign_data import CampaignData
from vla_lego.pretrained.campaign_learning import require_control_gate


def test_declarations_and_partial_admission_do_not_unlock_training(tmp_path):
    (tmp_path / "admitted-episodes.json").write_text(json.dumps({"episodes": []}))
    with pytest.raises(ValueError, match="complete admitted"):
        CampaignData(tmp_path, "train", seed=1)


def test_unreviewed_success_is_not_admitted_learning_data(tmp_path):
    (tmp_path / "admitted-episodes.json").write_text(
        json.dumps({"episodes": [{"partition": "train", "category": "tower", "admitted": False}]})
    )
    with pytest.raises(ValueError, match="reviewed, admitted"):
        CampaignData(tmp_path, "train", seed=1)


def test_full_task_training_requires_actual_unassisted_control_evidence(tmp_path):
    checkpoint = tmp_path / "020000"
    checkpoint.mkdir()
    gate = {
        "checkpoint": "020000",
        "completed_control_updates": 20_000,
        "teacher_assisted": False,
        "cases": [
            {
                "case": f"control-{i}",
                "start": "familiar" if i < 5 else "held_out",
                "active_arm": "left" if i % 2 else "right",
                "physical_success": i < 8,
            }
            for i in range(10)
        ],
    }
    path = tmp_path / "single-block-gate.json"
    path.write_text(json.dumps(gate))
    assert require_control_gate(tmp_path, checkpoint)["completed_control_updates"] == 20_000
    gate["cases"][0]["physical_success"] = False
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="not passed"):
        require_control_gate(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="resume"):
        require_control_gate(tmp_path, None)


def test_complete_episode_counts_cannot_replace_actual_policy_state_corrections(tmp_path):
    from vla_lego.isaac.campaign import EXPECTED
    from vla_lego.pretrained.campaign_data import admitted_entries

    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "episode.json").write_text(
        json.dumps({"collection_case": {"correction_origin": "teacher"}})
    )
    entries = [
        {
            "case": f"{partition}-{category}-{i}",
            "family": f"{partition}-{category}-{i}",
            "partition": partition,
            "category": category,
            "admitted": True,
            "recording": "recording",
        }
        for (partition, category), count in EXPECTED.items()
        for i in range(count)
    ]
    (tmp_path / "admitted-episodes.json").write_text(json.dumps({"episodes": entries}))
    with pytest.raises(ValueError, match="30 actual"):
        admitted_entries(tmp_path)
