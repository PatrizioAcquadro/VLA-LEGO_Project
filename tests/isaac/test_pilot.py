"""Pilot leakage, visible inventory and causal annotation contracts."""

import json

import numpy as np
import pytest

from vla_lego.isaac.pilot import PilotCase, piece
from vla_lego.isaac.recording import EpisodeRecorder


def case_row():
    return {
        "case": "T1-N1",
        "split": "train",
        "family": "T1",
        "category": "nominal",
        "order": "red CS > blue CL > green PS > yellow PL > purple CS",
        "distractors": "red CL; yellow CL; orange PS",
        "stabilizer": "left",
        "cooperative_placement": "4",
        "variation": "nominal spread",
        "error_target_level": "",
    }


def test_missing_piece_is_replaced_without_an_inventory_count_shortcut():
    row = dict(
        case_row(), case="T1-S", category="stop", missing_piece="red CS", replacement="red PS"
    )
    task = PilotCase(row, 9).task()
    assert len(task.placements) == 8
    assert not task.requested_pieces_available
    assert "small red cube" in task.instruction
    assert "absent" not in task.instruction


@pytest.mark.parametrize("reserved", ["blue PS", "purple PL", "yellow CS", "green CL"])
def test_reserved_combinations_are_rejected_in_any_role(reserved):
    with pytest.raises(ValueError, match="Reserved"):
        piece(reserved)


def test_stage_and_error_mask_are_separate_from_policy_signals(tmp_path):
    recorder = EpisodeRecorder(
        tmp_path / "episode", "instruction", family="T1", split="train", task_decisions=True
    )
    recorder.stage = "level3_wrong_pick"
    recorder.positive_imitation = False
    observation = {
        "joint_names": ("joint",),
        "timestamp_s": 0.0,
        "image_timestamp_s": 0.0,
        "joint_position": np.zeros(1),
        "joint_velocity": np.zeros(1),
        "head_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    recorder.record(
        observation, [0.1], ("joint",), [{"hidden": 1}], requested_action=[0.1], command_time_s=0.0
    )
    recorder.finish(verified_success=False, reason="preserved error")
    frame = json.loads((tmp_path / "episode/frames.jsonl").read_text())
    annotation = json.loads((tmp_path / "episode/annotations.jsonl").read_text())
    assert not {"stage", "positive_imitation", "hidden"} & frame.keys()
    assert frame["action"] == [0.1]
    assert annotation == {"frame": 0, "stage": "level3_wrong_pick", "positive_imitation": False}


def test_recovery_cannot_be_admitted_without_a_physical_fault(tmp_path):
    recorder = EpisodeRecorder(
        tmp_path / "episode",
        "instruction",
        family="T1",
        split="train",
        collection_case={"category": "recovery"},
    )
    with pytest.raises(ValueError, match="physically verified"):
        recorder.finish(verified_success=True, reason="a nominal result is insufficient")
    recorder.finish(verified_success=False, reason="no fault")


@pytest.mark.parametrize("ordinal", [0, 1, 2, 23, 45])
def test_marker_free_instruction_and_site_are_free_of_destination_cues(ordinal):
    task = PilotCase(case_row(), ordinal).task()
    assert not any(word in task.instruction for word in ("square", "marked", "left", "right"))
    assert task.build_region_xy_m is not None
    assert all(
        np.linalg.norm(np.array(task.assembly_xy_m) - p.xy_m) > 0.09 for p in task.placements
    )


def test_marked_corpus_cannot_be_admitted_as_marker_free(tmp_path):
    from vla_lego.isaac.pilot_audit import audit_episode

    (tmp_path / "recording").mkdir()
    (tmp_path / "report.json").write_text('{"passed": true}')
    (tmp_path / "recording/episode.json").write_text("{}")
    with pytest.raises(ValueError, match="Previous marked"):
        audit_episode(tmp_path, PilotCase(case_row(), 0))
