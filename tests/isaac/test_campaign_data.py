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


def test_recovery_coverage_counts_observed_interruption_instead_of_requested_placement(tmp_path):
    from vla_lego.pretrained.campaign_data import recovery_coverage

    recording = tmp_path / "episode" / "recording"
    recording.mkdir(parents=True)
    entry = {
        "category": "recovery",
        "partition": "train",
        "recording": "episode/recording",
        "declared_mechanism": "placement",
    }
    report = {
        "passed": True,
        "fault": {
            "type": "interrupted_progress",
            "physically_verified": True,
            "policy_source": {"observed_mechanism": "interrupted_progress"},
        },
    }
    (recording.parent / "report.json").write_text(json.dumps(report))
    assert recovery_coverage(tmp_path, [entry]) == {("train", "interrupted_progress"): 1}
    entry["observed_mechanism"] = "placement"
    with pytest.raises(ValueError, match="Admission labels"):
        recovery_coverage(tmp_path, [entry])
    entry.pop("observed_mechanism")
    report["fault"]["physically_verified"] = False
    (recording.parent / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="verified physical"):
        recovery_coverage(tmp_path, [entry])


def test_impossible_task_prefix_is_not_sampled_or_used_for_motor_statistics(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from vla_lego.isaac.campaign import CONDITION
    from vla_lego.pretrained import campaign_data

    annotations = [{"stage": "level0_transport", "positive_imitation": True} for _ in range(3)] + [
        {"stage": "verified_justified_abstention", "positive_imitation": True} for _ in range(3)
    ]
    frames = [
        {
            "joint_position": [1000.0 if i < 3 else 2.0] * 18,
            "joint_velocity": [0.0] * 18,
            "action": [1000.0 if i < 3 else 3.0] * 16,
        }
        for i in range(6)
    ]
    entry = {
        "case": "stop",
        "family": "stop-family",
        "category": "stop",
        "partition": "train",
        "recording": "recording",
    }
    episode = SimpleNamespace(
        metadata={
            "collection_condition": CONDITION,
            "collection_case": entry,
            "generation_family": "stop-family",
        },
        annotations=annotations,
        frames=frames,
        eligible_frames=list(range(6)),
        horizon=2,
    )
    monkeypatch.setattr(campaign_data, "admitted_entries", lambda root: [entry])
    monkeypatch.setattr(campaign_data, "PilotEpisode", lambda *a, **k: episode)
    data = CampaignData(tmp_path, "train", seed=1)
    assert {sample for values in data.pools[0].values() for sample in values} == {3, 4, 5}
    assert data.diagnostic_samples() == [(0, 3, "released_stop")]
    statistics = data.statistics()
    assert np.allclose(statistics["observation.state"]["mean"][:18], 2.0)
    assert np.allclose(statistics["action"]["mean"][:14], 1.0)
    assert np.allclose(statistics["action"]["mean"][14:], 3.0)
    assert len(episode.frames) == 6 and all(a["positive_imitation"] for a in annotations)
