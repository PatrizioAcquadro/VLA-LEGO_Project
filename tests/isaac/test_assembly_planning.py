"""A reachable first placement cannot certify the teacher's complete build site."""

import json
from types import SimpleNamespace

import pytest

from vla_lego.isaac.assembly_planning import choose_build_site, pick_route
from vla_lego.isaac.campaign import load_campaign
from vla_lego.isaac.campaign_design import build_declaration
from vla_lego.isaac.environment import TABLE_TOP


def test_site_selection_checks_the_last_level_without_changing_scene_or_goal(tmp_path):
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

    reference = SimpleNamespace(env=SimpleNamespace(task_truth=truth), grasp_yaw=grasp)
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
