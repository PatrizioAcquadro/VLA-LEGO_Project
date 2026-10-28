"""Isolated redundant-control geometry, not native robot qualification."""

import pytest
import torch

from vla_lego.isaac.control import _posture_step
from vla_lego.isaac.kinematics import _bounded_dls_step
from vla_lego.isaac.reference import _linear_profile


def test_joint_at_limit_keeps_cartesian_motion_through_other_joints():
    jacobian = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]])
    error = torch.tensor([[1.0, 1.0]])
    lower = torch.full((1, 3), -2.0)
    upper = torch.tensor([[0.0, 2.0, 2.0]])
    step = _bounded_dls_step(jacobian, error, 1e-4, lower, upper)
    assert step[0, 0] == 0
    assert ((jacobian @ step[..., None])[..., 0] - error).abs().max() < 1e-5
    assert ((step >= lower) & (step <= upper)).all()


def test_fully_bounded_motion_remains_finite_and_respects_motor_rate():
    jacobian = torch.ones((1, 1, 2))
    lower = torch.zeros((1, 2))
    upper = torch.full((1, 2), 0.025)
    step = _bounded_dls_step(jacobian, torch.ones((1, 1)), 0.03, lower, upper)
    assert torch.isfinite(step).all()
    assert torch.equal(step, upper)


def test_posture_correction_uses_redundancy_without_moving_the_tool():
    jacobian = torch.cat((torch.eye(6), torch.full((6, 1), 0.2)), dim=1)[None]
    delta = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0]])
    correction = _posture_step(jacobian, delta)
    assert correction.norm() > 0.1
    assert torch.isfinite(correction).all()
    assert (jacobian @ correction[..., None]).abs().max() < 1e-6


def test_linear_profile_preserves_line_and_bounds_speed_with_zero_end_velocity():
    start = torch.tensor([0.1, -0.2, 0.9])
    goal = torch.tensor([0.4, 0.1, 1.1])
    rows = list(_linear_profile(start.tolist(), goal.tolist()))
    points = torch.tensor([p for p, _ in rows])
    velocity = torch.tensor([v for _, v in rows])
    assert torch.allclose(points[-1], goal)
    assert velocity[-1].abs().max() == 0
    assert velocity.norm(dim=1).max() <= 0.080001
    assert torch.linalg.cross(points - start, (goal - start).expand_as(points)).abs().max() < 1e-6


@pytest.mark.parametrize("admission", ["nominal", "measured", "neither"])
def test_loaded_orientation_replanning_requires_admission_before_any_motion(admission):
    from types import SimpleNamespace

    from vla_lego.isaac.reference import PhysicalReference

    nominal = torch.tensor([0.0, -0.70710678, 0.0, -0.70710678])
    pose = torch.tensor([0.52, 0.4, 0.89, -0.0035, -0.6916, -0.0042, -0.7223])
    attempts = []

    def solve(goal, seed):
        attempts.append(goal.clone())
        if admission == "neither" or (admission == "measured" and len(attempts) == 1):
            raise RuntimeError("Pose admission refused")
        return seed

    reference = PhysicalReference.__new__(PhysicalReference)
    reference.env = SimpleNamespace(
        target=torch.zeros((1, 7)),
        robot=SimpleNamespace(
            data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.zeros(1, 7)))
        ),
    )
    reference.ctrl = SimpleNamespace(
        pose=lambda side: pose[None], frames={"left": (None, list(range(7)), None)}
    )
    reference.solvers = {"left": SimpleNamespace(solve=solve)}
    reference.orientation = lambda side, yaw: nominal
    if admission == "neither":
        with pytest.raises(RuntimeError, match="admission refused"):
            reference.lift_orientation("left", [0.52, 0.4, 0.99], 0)
        assert len(attempts) == 2
    else:
        plan = reference.lift_orientation("left", [0.52, 0.4, 0.99], 0)
        if admission == "nominal":
            assert plan is None and len(attempts) == 1
        else:
            assert plan["quaternion_xyzw"] == pose[3:].tolist()
            assert plan["nominal_first_waypoint_refusal"] == "Pose admission refused"
            assert torch.equal(attempts[0][:3], attempts[1][:3])
            assert torch.equal(attempts[1][3:], pose[3:])


@pytest.mark.parametrize("safe,waypoints", [(True, 1), (True, 2), (False, 2)])
def test_checked_park_refuses_before_motion_when_native_path_is_obstructed(
    monkeypatch, safe, waypoints
):
    from types import SimpleNamespace

    from vla_lego.isaac import clearance
    from vla_lego.isaac.reference import PhysicalReference

    reference = PhysicalReference.__new__(PhysicalReference)
    reference.env = SimpleNamespace(target=torch.zeros((1, 7)))
    reference.ctrl = object()
    reference.solvers = {"right": object()}
    calls = []
    moved = []

    def plan(env, controller, solver, side, goal):
        assert (env, controller, solver, side) == (
            reference.env,
            reference.ctrl,
            reference.solvers["right"],
            "right",
        )
        calls.append("preflight")
        if not safe:
            raise RuntimeError("Other jaw obstructs park")
        return ([goal / 2, goal] if waypoints == 2 else [goal]), 0.012

    monkeypatch.setattr(clearance, "plan_empty_hand_path", plan)

    def move(stage, side, goal):
        calls.append("motion")
        moved.append(goal)

    reference.joint_move = move
    if safe:
        assert reference.park("right", check_clearance=True) == 0.012
        assert calls == ["preflight"] + ["motion"] * waypoints
        assert len(moved) == waypoints
        if waypoints == 2:
            assert torch.equal(moved[0], moved[1] / 2)
    else:
        with pytest.raises(RuntimeError, match="Other jaw obstructs park"):
            reference.park("right", check_clearance=True)
        assert calls == ["preflight"]
