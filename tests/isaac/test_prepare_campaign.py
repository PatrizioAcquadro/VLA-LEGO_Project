"""Corpus preparation must preserve the learning gate and causal signal order."""

import json

import pytest

from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS
from vla_lego.pretrained.prepare_campaign import measured_interface, prepare


def test_incomplete_corpus_cannot_create_learning_inputs(tmp_path):
    (tmp_path / "admitted-episodes.json").write_text(json.dumps({"episodes": []}))
    statistics = tmp_path / "learning/statistics.json"
    with pytest.raises(ValueError, match="complete admitted"):
        prepare(tmp_path, statistics)
    assert not statistics.exists()
    assert not (tmp_path / "alex_interface.json").exists()


def test_measured_interface_keeps_velocity_inputs_and_decisions_out_of_motors():
    metadata = {
        "state_names": [*ARM_JOINTS, "NECK_Z", "NECK_Y", *JAW_JOINTS],
        "action_names": [*ARM_JOINTS, *JAW_JOINTS],
        "fps": 20,
        "privileged_policy_inputs": False,
    }
    interface = measured_interface(metadata)
    assert interface["position_names"] == metadata["state_names"]
    assert len(interface["state_names"]) == 32
    assert interface["state_names"][18:] == [name + "_velocity" for name in ARM_JOINTS]
    assert len(interface["action_joint_names"]) == 16
    assert interface["include_arm_velocity"]
    metadata["action_names"] += ["complete", "abstain"]
    with pytest.raises(ValueError, match="measured Alex interface"):
        measured_interface(metadata)
