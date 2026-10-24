"""Isolated redundant-control geometry, not native robot qualification."""

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
