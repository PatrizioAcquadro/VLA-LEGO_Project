"""A reachable first placement cannot certify the teacher's complete build site."""

import json
from types import SimpleNamespace

import pytest

from vla_lego.isaac.assembly_planning import choose_build_site, lift_route, pick_route, source_route
from vla_lego.isaac.campaign import load_campaign
from vla_lego.isaac.campaign_design import build_declaration
from vla_lego.isaac.catalog import BlockSpec
from vla_lego.isaac.environment import TABLE_TOP
from vla_lego.isaac.stabilizer_planning import stabilized_level, stabilizer_exit_route


def test_stabilizer_braces_wider_support_beneath_a_tall_narrow_prism():
    small = BlockSpec("cube", "small", "red")
    large = BlockSpec("cube", "large", "yellow")
    prism = BlockSpec("prism", "small", "yellow")
    # The 54.4 mm prism can tip when gripped above its wider supporting cube.
    task = SimpleNamespace(targets=(small, large, prism))
    assert stabilized_level(task, 3) == 1
    assert stabilized_level(SimpleNamespace(targets=(small, small, small)), 3) == 2


@pytest.mark.parametrize("side, sign", [("left", 1), ("right", -1)])
def test_stabilizer_rotation_keeps_the_wide_exit_and_cleared_height(side, sign):
    orientation = [0.06, 0.06, -0.70, 0.70]
    direct = stabilizer_exit_route([0.407, 0.107, 1.03], side, 1.096, orientation)
    rotated = stabilizer_exit_route([0.407, 0.107, 1.03], side, 1.096, orientation, rotate=True)
    assert direct[-1][1] == rotated[-1][1] == [0.407, sign * 0.40, 1.096]
    assert all(xyz[2] == 1.096 for _, xyz, _ in rotated)
    assert rotated[0][1] == rotated[1][1] == [0.407, sign * 0.20, 1.096]
    assert rotated[0][2] == orientation
    assert rotated[1][2] == rotated[2][2] == [0, 0, 0, 1]


def test_site_selection_checks_the_last_level_without_changing_scene_or_goal(tmp_path, monkeypatch):
    from vla_lego.isaac import stabilizer_planning

    monkeypatch.setattr(stabilizer_planning, "plan_stabilizer", lambda *a, **k: (True, 0, []))
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(build_declaration()))
    case = next(c for c in load_campaign(path) if c.name == "TR-R-slip_drop-01-00-0")
    task = case.task()
    indices = [next(i for i, p in enumerate(task.placements) if p.spec == t) for t in task.targets]
    roles = [("left" if task.placements[i].xy_m[1] > 0 else "right", 0) for i in indices]
    calls = []

    def truth(index):
        p = task.placements[index]
        return {"block_pose_w": [*p.xy_m, TABLE_TOP + p.spec.dimensions_m[2] / 2]}

    def grasp(_side, _yaw, route):
        calls.append(route)
        if route[-3][2] > 1.12 and abs(route[-3][1]) > 0.10:
            raise RuntimeError("Native head collision on high cross-body reach")

    reference = SimpleNamespace(
        env=SimpleNamespace(task_truth=truth), grasp_yaw=grasp, grasp_height=lambda *a: 0.06
    )
    selected, report = choose_build_site(reference, task, indices, roles, 3)
    assert selected == (0.4, 0.0)
    assert report["rejected"][0]["level"] == 5
    assert len(calls) == 10
    assert task.assembly_xy_m != selected
    assert report["all_level_arm_routes_admitted"] and not report["physical_success"]
    block = truth(indices[-1])["block_pose_w"]
    route = pick_route(task, 4, block, selected, roles[-1][0], 3)
    expected = TABLE_TOP + sum(t.dimensions_m[2] for t in task.targets[:4])
    assert route[-3][2] == pytest.approx(expected + task.targets[4].dimensions_m[2] / 2 + 0.06)


def test_a_placeable_tower_site_still_requires_stabilizer_access(tmp_path, monkeypatch):
    from copy import deepcopy

    from vla_lego.isaac import stabilizer_planning

    path = tmp_path / "cases.json"
    path.write_text(json.dumps(build_declaration()))
    case = next(c for c in load_campaign(path) if c.name == "DC-T01-00-0")
    task = case.task()
    indices = [next(i for i, p in enumerate(task.placements) if p.spec == t) for t in task.targets]
    roles = [("left" if task.placements[i].xy_m[1] > 0 else "right", 0) for i in indices]
    states = [
        {"block_pose_w": [*p.xy_m, TABLE_TOP + p.spec.dimensions_m[2] / 2, 0, 0, 0, 1]}
        for p in task.placements
    ]
    original = deepcopy(states)

    def plan(*args, states, **kwargs):
        xy = states[indices[0]]["block_pose_w"][:2]
        return (
            (True, 0, []) if xy == [0.4, 0.0] else (None, float("inf"), ["inaccessible stabilizer"])
        )

    monkeypatch.setattr(stabilizer_planning, "plan_stabilizer", plan)
    reference = SimpleNamespace(
        env=SimpleNamespace(task_truth=lambda i: states[i]),
        grasp_yaw=lambda *a: 0,
        grasp_height=lambda *a: 0.06,
    )
    selected, report = choose_build_site(reference, task, indices, roles, 3)
    assert selected == (0.4, 0.0) and report["stabilizer_approach_and_withdrawal_admitted"]
    assert report["rejected"][0]["stage"] == "stabilizer"
    assert states == original and task.assembly_xy_m != selected


def test_a_shallower_pinch_preserves_lift_and_placement_geometry():
    target = BlockSpec("cube", "small", "green")
    task = SimpleNamespace(targets=(target,) * 5)
    route = pick_route(task, 3, [0.5, -0.3, 0.82], (0.4, 0), "right", 3, grasp_height=0.075)
    assert route[4][2] - route[3][2] == pytest.approx(0.10)
    assert route[-3][2] == pytest.approx(TABLE_TOP + 3 * 0.04 + 0.02 + 0.075)
    assert all(p[2] >= route[-3][2] + 0.025 for p in route[5:-3])


def test_displaced_pick_plans_the_same_low_escape_before_its_high_carry():
    prism = BlockSpec("prism", "large", "red")
    cube = BlockSpec("cube", "large", "blue")
    task = SimpleNamespace(targets=(prism, prism, cube, cube, cube))
    block, build = [0.4046, 0.1013, 0.825], [0.4078, 0.0498]
    lifting = lift_route(block, build, "right", displaced=True)
    planned = pick_route(task, 3, block, build, "right", 3, displaced=True)
    approach_length = len(source_route(block, build, "right", displaced=True))
    assert planned[approach_length : approach_length + len(lifting)] == [
        point for _, point in lifting
    ]
    # Clear the nearby prefix laterally while the carried cube is still below its top.
    horizontal = lifting[1:3]
    assert all(point[2] < lifting[-1][1][2] for _, point in horizontal)
    assert horizontal[-1][1][0] > block[0] + 0.07
    assert horizontal[-1][1][1] < build[1] - 0.10
    assert lifting[-1][1][2] - (block[2] + 0.06) == pytest.approx(0.10)
    assert planned[approach_length + len(lifting)][:2] == lifting[-1][1][:2]


@pytest.mark.parametrize("side, sign", [("left", 1), ("right", -1)])
def test_displaced_approach_reverses_the_low_escape_before_regrasp(side, sign):
    block, build = [0.4459, -sign * 0.1066, 0.825], [0.4075, -sign * 0.0483]
    approach = source_route(block, build, side, displaced=True)
    lifting = lift_route(block, build, side, displaced=True)
    assert [point for _, point in approach[2:5]] == [point for _, point in reversed(lifting[:3])]
    assert approach[1][1][:2] == approach[2][1][:2]
    assert approach[-1][1] == [block[0], block[1], block[2] + 0.06]
    task = SimpleNamespace(targets=(BlockSpec("cube", "large", "blue"),) * 5)
    planned = pick_route(task, 3, block, build, side, 3, displaced=True)
    assert planned[: len(approach)] == [point for _, point in approach]
