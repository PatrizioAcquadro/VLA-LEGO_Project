"""Preflight contracts only; native physical evidence is recorded separately."""

import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "condition", ["clear", "antipodal", "late_piece", "late_arm", "solved", "solve", "oom"]
)
def test_released_rotation_preflights_without_motion(monkeypatch, side, condition):
    from vla_lego.isaac import clearance, stabilizer_planning

    monkeypatch.setitem(
        sys.modules,
        "isaaclab.utils.math",
        SimpleNamespace(quat_from_matrix=lambda _: torch.tensor([0.0, 0.0, 0.0, 1.0])),
    )
    pose = torch.tensor([[0.4, 0.2 if side == "left" else -0.2, 1.08, 0.0, 0.0, 0.0, 1.0]])
    joints = pose.clone()
    env = SimpleNamespace(
        target=pose.clone(),
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joints))),
    )
    ctrl = SimpleNamespace(frames={side: (None, list(range(7)), None)}, pose=lambda _: pose)
    solved, inspected, batches = [], [], []

    def solve(target, seed):
        solved.append(target.clone())
        if condition == "oom":
            raise torch.cuda.OutOfMemoryError("native query fixture")
        if condition == "solve":
            raise RuntimeError("Native IK refused")
        return target.clone()

    def forward(configurations):
        transforms = torch.eye(4).repeat(len(configurations), 1, 1)
        transforms[:, :3, 3] = configurations[:, :3]
        if condition == "solved":
            transforms[:, 0, 3] = 0.7
        return transforms, None

    def native(configurations):
        batches.append(configurations.clone())
        admitted = torch.ones(len(configurations), dtype=torch.bool)
        if condition == "late_arm" and len(batches) == 3:
            admitted[len(admitted) // 2] = False
        return admitted

    def clear(actual):
        inspected.append(actual.clone())
        if condition == "late_piece" and len(solved) == 3:
            return 0.0
        if condition == "solved" and float(actual[0]) > 0.6:
            return -0.01
        return 0.01

    def geometry(*args, **kwargs):
        assert args[3] is None and kwargs == {"other_hand": True}
        return clear

    admission = object()
    solver = SimpleNamespace(solve=solve, forward=forward, configuration_admission=admission)
    reference = SimpleNamespace(env=env, ctrl=ctrl, solvers={side: solver})
    monkeypatch.setattr(clearance, "arm_clearance", lambda *_: native)
    monkeypatch.setattr(clearance, "source_grasp_clearance", geometry)
    endpoint = torch.tensor([0.0, 0.0, 0.70710678, 0.70710678])
    if condition == "antipodal":
        endpoint = -endpoint
    before_joints, before_pose, before_target = joints.clone(), pose.clone(), env.target.clone()
    if condition in ("clear", "antipodal"):
        result = stabilizer_planning.plan_released_rotation(reference, side, endpoint.tolist())
        assert len(result) == len(solved) == len(batches) == 8
        for target in result:
            assert target[:3] == before_pose[0, :3].tolist()
            assert torch.tensor(target[3:]).norm().item() == pytest.approx(1.0)
        assert result[-1][3:] == pytest.approx([0.0, 0.0, 0.70710678, 0.70710678])
        assert all(float(target[6]) > 0 for target in solved)
        assert inspected and all(len(batch) >= 2 for batch in batches)
    else:
        expected = torch.cuda.OutOfMemoryError if condition == "oom" else RuntimeError
        with pytest.raises(expected):
            stabilizer_planning.plan_released_rotation(reference, side, endpoint.tolist())
        assert len(solved) == (3 if condition in ("late_piece", "late_arm") else 1)
    assert solver.configuration_admission is admission
    assert torch.equal(joints, before_joints)
    assert torch.equal(pose, before_pose)
    assert torch.equal(env.target, before_target)
