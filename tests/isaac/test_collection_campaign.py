"""Essential allocation, split, instruction-grounding and physical success contracts."""

import json
from collections import Counter

import pytest

from vla_lego.isaac.campaign import load_campaign
from vla_lego.isaac.campaign_design import build_declaration
from vla_lego.isaac.pick_evaluation import PickPlaceEvaluator


def test_campaign_preserves_quota_pairs_and_balanced_single_block_targets(tmp_path):
    data = build_declaration()
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    cases = load_campaign(path)
    assert len(cases) == 620
    singles = [c for c in cases if c.row["partition"] == "train" and c.category == "single_block"]
    targets = Counter(c.task().targets[0] for c in singles)
    assert len(targets) == 20 and set(targets.values()) == {8}
    for target in targets:
        sides = Counter(
            next(p.xy_m[1] > 0 for p in c.task().placements if p.spec == target)
            for c in singles
            if c.task().targets[0] == target
        )
        assert sides == {True: 4, False: 4}
    nominal = [
        c
        for c in cases
        if c.row["partition"] == "train" and c.category in ("single_block", "tower")
    ]
    assert Counter(c.row["variation"] for c in nominal) == {
        "baseline": 240,
        "layout": 80,
        "physics_appearance": 80,
    }
    initial = [c for c in cases if c.name in data["initial_cases"]]
    assert Counter(c.row["split"] for c in initial) == {"train": 24, "development": 8}


def test_campaign_rejects_changed_scene_within_instruction_pair(tmp_path):
    data = build_declaration()
    data["cases"][1]["placements"] = [dict(p) for p in data["cases"][1]["placements"]]
    data["cases"][1]["placements"][0]["xy_m"] = [0.45, 0.22]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="preserve the physical scene"):
        load_campaign(path)


def test_single_block_success_requires_actual_grasp_transport_release_and_dwell(tmp_path):
    from vla_lego.isaac.environment import TABLE_TOP

    data = build_declaration()
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    task = load_campaign(path)[0].task()
    paths = [
        f"/{side}_WSG32_{jaw}_JAW_LINK"
        for side in ("left", "right")
        for jaw in ("NEGATIVE", "POSITIVE")
    ]
    blocks = [
        {
            "block_pose_w": [*p.xy_m, TABLE_TOP + p.spec.dimensions_m[2] / 2, 0, 0, 0, 1],
            "block_velocity_w": [0] * 6,
            "robot_contact_force_n": 0.0,
            "robot_contact_forces_n": [[0, 0, 0] for _ in paths],
            "nonrobot_contact_force_n": 0.25,
        }
        for p in task.placements
    ]
    tools = {"left": [0, 0.5, 1.3], "right": [0, -0.5, 1.3]}
    evaluator = PickPlaceEvaluator(task, paths)
    evaluator.update(0, blocks, tool_positions=tools)
    state = blocks[evaluator.index]
    state["block_pose_w"][2] += 0.10
    state["robot_contact_forces_n"][:2] = [[0.4, 0, 0], [-0.4, 0, 0]]
    state["robot_contact_force_n"] = 0.8
    evaluator.update(0.05, blocks, tool_positions=tools)
    state["block_pose_w"][:3] = [
        *task.assembly_xy_m,
        TABLE_TOP + task.targets[0].dimensions_m[2] / 2,
    ]
    for tick in range(42):
        evaluator.update(0.1 + tick * 0.05, blocks, tool_positions=tools)
    assert not evaluator.result()["passed"]
    state["robot_contact_force_n"] = 0
    state["robot_contact_forces_n"] = [[0, 0, 0] for _ in paths]
    for tick in range(41):
        evaluator.update(2.2 + tick * 0.05, blocks, tool_positions=tools)
    assert evaluator.result()["passed"]
    prearranged = PickPlaceEvaluator(task, paths)
    for tick in range(41):
        prearranged.update(tick * 0.05, blocks, tool_positions=tools)
    assert not prearranged.result()["passed"]
