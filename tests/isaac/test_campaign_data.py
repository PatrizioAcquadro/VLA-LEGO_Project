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
