"""Expanded learning must not start from a declaration or an unfinished corpus."""

import json

import pytest

from vla_lego.pretrained.campaign_data import CampaignData


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
