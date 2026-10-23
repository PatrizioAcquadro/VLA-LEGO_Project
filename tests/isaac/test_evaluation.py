"""Analytical scorer fixtures only; these are not physical demonstrations."""

import math
from dataclasses import replace

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
            "motion_speed_max": [0.0, 0.0],
            "motion_window_s": 0.05,
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


def construction(
    task,
    order=(0, 1, 2, 3, 4),
    *,
    release_samples=41,
    release_period=0.05,
    motion_measurement="solver_velocity",
    evaluation_task=None,
):
    scorer = TowerEvaluator(evaluation_task or task, PATHS, motion_measurement=motion_measurement)
    tools = {"left": [0, 0.5, 1.2], "right": [0, -0.5, 1.2]}
    scorer.update(0.0, states(task), tool_positions=tools)
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
    scorer.update(0.05, approach, tool_positions=tools)
    contact = states(task, order, 4)
    grip(contact, order[2], "left")
    grip(contact, order[3], "right")
    for tick in range(6):
        scorer.update(0.1 + tick * 0.05, contact, tool_positions=tools)
    released = states(task, order, 5)
    for tick in range(release_samples):
        scorer.update(0.4 + tick * release_period, released, tool_positions=tools)
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


def test_pose_motion_protocol_does_not_use_unbiased_solver_velocity():
    task = initial_tower_task()
    scorer = construction(task, motion_measurement="pose_step_peaks")
    start = scorer.previous_time + 0.05
    for tick in range(41):
        row = states(task, levels=5)
        for state in row:
            state["block_velocity_w"] = [0, 0, 0, 1, 0, 0]
        scorer.update(start + tick * 0.05, row)
    assert scorer.result()["passed"]


def test_measured_motion_peak_rejects_an_interval_with_quiet_endpoint():
    task = initial_tower_task()
    scorer = construction(task, motion_measurement="pose_step_peaks")
    row = states(task, levels=5)
    row[4]["motion_speed_max"] = [0.02, 0]
    scorer.update(scorer.previous_time + 0.05, row)
    assert not scorer.result()["passed"]


def test_motion_dwell_is_forty_intervals_not_forty_one():
    task = initial_tower_task()
    scorer = construction(task, motion_measurement="pose_step_peaks")
    start = scorer.previous_time + 0.05
    for tick in range(41):
        row = states(task, levels=5)
        if tick == 0:
            row[4]["motion_speed_max"] = [0.02, 0]
        scorer.update(start + tick * 0.05, row)
    assert scorer.result()["passed"]


def test_legacy_trace_cannot_silently_acquire_pose_motion_admission():
    task = initial_tower_task()
    scorer = TowerEvaluator(task, PATHS, motion_measurement="pose_step_peaks")
    row = states(task)
    for state in row:
        state.pop("motion_speed_max")
        state.pop("motion_window_s")
    with pytest.raises(ValueError, match="fresh physics-step measurements"):
        scorer.update(0.0, row)


@pytest.mark.parametrize("displacement,passed", [(0.002, True), (0.01, False)])
def test_functional_dwell_accepts_millimeter_motion_and_rejects_centimeter_drift(
    displacement, passed
):
    task = initial_tower_task()
    scorer = construction(task, motion_measurement="pose_dwell")
    start = scorer.previous_time + 0.05
    for tick in range(41):
        row = states(task, levels=5)
        row[4]["block_pose_w"][0] += 0.002 + displacement * tick / 40
        # Numerical velocity chatter is not a displacement or loss of support.
        row[4]["block_velocity_w"] = [0.02, 0, 0, 0.2, 0, 0]
        scorer.update(start + tick * 0.05, row)
    result = scorer.result()
    assert result["passed"] == passed
    assert result["dwell_corner_displacement_m"][4] == pytest.approx(displacement)


def test_functional_dwell_detects_rocking_even_without_center_motion():
    task = initial_tower_task()
    scorer = construction(task, motion_measurement="pose_dwell")
    row = states(task, levels=5)
    row[4]["block_pose_w"][3:] = [math.sin(0.2), 0, 0, math.cos(0.2)]
    scorer.update(scorer.previous_time + 0.05, row)
    result = scorer.result()
    assert not result["passed"]
    assert result["dwell_corner_displacement_m"][4] > 0.003


def test_functional_dwell_still_requires_support_and_release():
    task = initial_tower_task()
    scorer = construction(task, motion_measurement="pose_dwell")
    row = states(task, levels=5)
    grip(row, 4, "right")
    row[4]["block_contact_forces_n"][3][2] = 0
    scorer.update(scorer.previous_time + 0.05, row)
    result = scorer.result()
    assert not result["passed"]
    assert not result["last_sample_checks"][0]["ordered_support"]
    assert not result["last_sample_checks"][0]["released"]


@pytest.mark.parametrize("xy", [(0.31, -0.09), (0.44, 0.08)])
def test_marker_free_success_does_not_depend_on_teacher_point(xy):
    task = replace(
        initial_tower_task(), assembly_xy_m=xy, build_region_xy_m=(0.30, 0.45, -0.13, 0.13)
    )
    evaluator_task = replace(task, assembly_xy_m=(0.40, 0))
    assert construction(task, evaluation_task=evaluator_task).result()["passed"]


def test_marker_free_invalid_site_and_unwithdrawn_hands_are_rejected():
    task = replace(
        initial_tower_task(), assembly_xy_m=(0.29, 0), build_region_xy_m=(0.30, 0.45, -0.13, 0.13)
    )
    assert not construction(task).result()["passed"]
    task = replace(task, assembly_xy_m=(0.35, 0.06))
    scorer = construction(task)
    scorer.update(
        scorer.previous_time + 0.05,
        states(task, levels=5),
        tool_positions={"left": [0.35, 0.06, 0.9], "right": [0, -0.5, 1.2]},
    )
    assert not scorer.result()["passed"]


@pytest.mark.parametrize(
    "final_flat,gap,passed", [(True, 0.0, True), (False, 0.0, False), (True, 0.02, False)]
)
def test_cooperative_contact_uses_actual_surface_and_keeps_final_geometry(final_flat, gap, passed):
    task = initial_tower_task()
    scorer = TowerEvaluator(task, PATHS, motion_measurement="pose_dwell")
    scorer.update(0, states(task))
    approach = states(task, levels=3)
    approach[3]["block_pose_w"][:3] = [*task.assembly_xy_m, scorer.expected_z[3] + 0.03]
    approach[3]["nonrobot_contact_force_n"] = 0
    grip(approach, 2, "left")
    grip(approach, 3, "right")
    scorer.update(0.05, approach)
    contact = states(task, levels=4)
    angle = 0.16
    height, width = task.targets[3].dimensions_m[2], task.targets[3].dimensions_m[1]
    contact[3]["block_pose_w"][2] += (
        height * (math.cos(angle) - 1) + width * math.sin(angle)
    ) / 2 + gap
    contact[3]["block_pose_w"][3:] = [math.sin(angle / 2), 0, 0, math.cos(angle / 2)]
    grip(contact, 2, "left")
    grip(contact, 3, "right")
    for tick in range(6):
        scorer.update(0.1 + tick * 0.05, contact)
    released = states(task, levels=5)
    if not final_flat:
        released[3]["block_pose_w"] = contact[3]["block_pose_w"][:]
    for tick in range(41):
        scorer.update(0.4 + tick * 0.05, released)
    assert scorer.result()["passed"] == passed
    assert bool(scorer.result()["cooperation_events"]) == (gap == 0)
