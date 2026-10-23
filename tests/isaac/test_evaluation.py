"""Analytical scorer fixtures only; these are not physical demonstrations."""

import pytest

from vla_lego.isaac.catalog import BlockPlacement
from vla_lego.isaac.environment import TABLE_TOP
from vla_lego.isaac.evaluation import TowerEvaluator
from vla_lego.isaac.task import TowerTask, initial_tower_task

PATHS = [
    f"/Robot/{side}_WSG32_{jaw}_JAW_LINK"
    for side in ("left", "right")
    for jaw in ("NEGATIVE", "POSITIVE")
]


def states(task, order=(0, 1, 2, 3, 4), levels=0):
    count = len(task.placements)
    rows = [
        {
            "block_pose_w": [*p.xy_m, TABLE_TOP + p.spec.dimensions_m[2] / 2, 0, 0, 0, 1],
            "block_velocity_w": [0.0] * 6,
            "robot_contact_force_n": 0.0,
            "robot_contact_forces_n": [[0.0] * 3 for _ in PATHS],
            "block_contact_forces_n": [[0.0] * 3 for _ in range(count)],
            "nonrobot_contact_force_n": p.spec.mass_kg * 9.81,
        }
        for p in task.placements
    ]
    bottom = TABLE_TOP
    for level, index in enumerate(order[:levels]):
        spec = task.placements[index].spec
        rows[index]["block_pose_w"][:3] = [*task.assembly_xy_m, bottom + spec.dimensions_m[2] / 2]
        if level:
            load = sum(task.placements[i].spec.mass_kg for i in order[level:levels]) * 9.81
            rows[index]["nonrobot_contact_force_n"] = 0.0
            rows[index]["block_contact_forces_n"][order[level - 1]][2] = load
            rows[order[level - 1]]["block_contact_forces_n"][index][2] = -load
        bottom += spec.dimensions_m[2]
    return rows


def grip(rows, index, side):
    ids = (0, 1) if side == "left" else (2, 3)
    rows[index]["robot_contact_forces_n"][ids[0]] = [0.4, 0, 0]
    rows[index]["robot_contact_forces_n"][ids[1]] = [-0.4, 0, 0]
    rows[index]["robot_contact_force_n"] = 0.8


def construction(task, order=(0, 1, 2, 3, 4), *, release_samples=41, release_period=0.05):
    scorer = TowerEvaluator(task, PATHS)
    scorer.update(0.0, states(task))
    approach = states(task, order, 3)
    approach[order[3]]["block_pose_w"][:3] = [
        *task.assembly_xy_m,
        TABLE_TOP
        + sum(p.dimensions_m[2] for p in task.targets[:3])
        + task.targets[3].dimensions_m[2] / 2
        + 0.025,
    ]
    approach[order[3]]["nonrobot_contact_force_n"] = 0.0
    grip(approach, order[2], "left")
    grip(approach, order[3], "right")
    scorer.update(0.05, approach)
    contact = states(task, order, 4)
    grip(contact, order[2], "left")
    grip(contact, order[3], "right")
    for tick in range(6):
        scorer.update(0.1 + tick * 0.05, contact)
    released = states(task, order, 5)
    for tick in range(release_samples):
        scorer.update(0.4 + tick * release_period, released)
    return scorer


def test_complete_scoring_history_requires_real_cooperation_conditions():
    result = construction(initial_tower_task()).result()
    assert result["passed"]
    assert result["cooperation_events"][0]["stabilizer"] == "left"


def test_prearranged_goal_is_not_construction():
    task = initial_tower_task()
    scorer = TowerEvaluator(task, PATHS)
    for tick in range(41):
        scorer.update(tick * 0.05, states(task, levels=5))
    result = scorer.result()
    assert result["checks"]["ordered_supported_released_stable"]
    assert not result["checks"]["initially_loose"]
    assert not result["passed"]


@pytest.mark.parametrize("order", [(5, 1, 2, 3, 4), (6, 1, 2, 3, 4), (0, 2, 1, 3, 4)])
def test_wrong_size_color_or_order_fails(order):
    result = construction(initial_tower_task(), order).result()
    assert not result["checks"]["ordered_supported_released_stable"]
    assert not result["passed"]


def test_visibly_equivalent_instance_can_replace_original_identity():
    original = initial_tower_task()
    task = TowerTask(
        original.placements + (BlockPlacement(original.targets[0], (0.65, -0.3)),),
        original.targets,
        original.assembly_xy_m,
    )
    result = construction(task, (7, 1, 2, 3, 4)).result()
    assert result["passed"]
    assert result["final_assignment_indices"][0] == 7


def test_holding_a_distractor_prevents_full_release():
    task = initial_tower_task()
    scorer = construction(task)
    held = states(task, levels=5)
    grip(held, 6, "right")
    for tick in range(41):
        scorer.update(2.45 + tick * 0.05, held)
    assert not scorer.result()["passed"]
    assert not scorer.result()["last_sample_checks"][0]["released"]


@pytest.mark.parametrize("samples,period", [(40, 0.05), (41, 0.15)])
def test_short_or_gapped_release_evidence_is_insufficient(samples, period):
    result = construction(
        initial_tower_task(), release_samples=samples, release_period=period
    ).result()
    assert not result["passed"]


def test_grabbing_an_already_supported_upper_piece_is_not_a_placement():
    task = initial_tower_task()
    scorer = TowerEvaluator(task, PATHS)
    scorer.update(0, states(task))
    completed = states(task, levels=5)
    grip(completed, 2, "left")
    grip(completed, 3, "right")
    for tick in range(8):
        scorer.update(0.05 + tick * 0.05, completed)
    for tick in range(41):
        scorer.update(0.45 + tick * 0.05, states(task, levels=5))
    result = scorer.result()
    assert result["checks"]["ordered_supported_released_stable"]
    assert not result["checks"]["functional_cooperation"]
    assert not result["passed"]
