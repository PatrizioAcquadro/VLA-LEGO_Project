"""Planner boundaries only; these fixtures do not prove physical repair."""

import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("condition", ["diagonal", "vertical", "touch", "arm", "solved", "oom"])
def test_released_route_checks_complete_path_before_motion(monkeypatch, side, condition):
    from vla_lego.isaac import clearance as module

    monkeypatch.setitem(
        sys.modules,
        "isaaclab.utils.math",
        SimpleNamespace(quat_from_matrix=lambda _: torch.tensor([0.0, 0.0, 0.0, 1.0])),
    )
    joints = torch.zeros(1, 7)
    pose = torch.tensor([[0.4, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    env = SimpleNamespace(
        target=pose,
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joints))),
    )
    ctrl = SimpleNamespace(frames={side: (None, list(range(7)), None)}, pose=lambda _: pose)
    direction = 1 if side == "left" else -1
    goal = [0.4, direction * 0.15, 1.1, 0.0, 0.0, 0.0, 1.0]
    inspected, solved = [], []

    def clear(target):
        inspected.append(target.clone())
        if condition == "touch":
            return 0.0
        if condition == "solved" and float(target[0]) > 0.6:
            return -1.0
        if condition == "vertical" and abs(float(target[1])) > 0.03 and float(target[2]) < 1.09:
            return -0.01
        return 0.01

    def geometry(*args, **kwargs):
        assert args[3] is None and kwargs == {"other_hand": True}
        return clear

    def solve(target, seed):
        solved.append(target.clone())
        if condition == "oom":
            raise torch.cuda.OutOfMemoryError("fixture")
        if condition == "arm":
            raise RuntimeError("Native arm/head/table/joint bounds refuse")
        return target.clone()

    def forward(seed):
        tf = torch.eye(4)[None]
        tf[0, :3, 3] = seed[0, :3]
        if condition == "solved":
            tf[0, 0, 3] = 0.7
        return tf, None

    solver = SimpleNamespace(solve=solve, forward=forward, configuration_admission=object())
    admission = solver.configuration_admission
    monkeypatch.setattr(module, "source_grasp_clearance", geometry)
    if condition == "oom":
        with pytest.raises(torch.cuda.OutOfMemoryError):
            module.plan_released_retreat(env, ctrl, solver, side, goal)
        assert len(solved) == 1
    elif condition in ("touch", "arm", "solved"):
        with pytest.raises(RuntimeError, match="No admissible released-hand retreat"):
            module.plan_released_retreat(env, ctrl, solver, side, goal)
        if condition == "touch":
            assert not solved
    else:
        route = module.plan_released_retreat(env, ctrl, solver, side, goal)
        assert route[-1] == goal[:3]
        assert len(route) == (2 if condition == "vertical" else 1)
        if condition == "vertical":
            assert route[0] == pytest.approx([0.4, 0.0, 1.1])
        assert len(solved) > 10 and len(inspected) > len(solved)
    assert solver.configuration_admission is admission
    assert torch.equal(joints, torch.zeros_like(joints))
    assert torch.equal(pose, torch.tensor([[0.4, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]))
