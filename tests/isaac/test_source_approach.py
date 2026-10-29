"""A reachable endpoint must not admit an unreachable source-approach interior."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vla_lego.isaac import reference as module


@pytest.mark.parametrize("blocked", [False, True])
def test_source_approach_checks_reachable_interior_before_admission(blocked):
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    assert Path(module.__file__).resolve().is_relative_to(root / "src")
    goals = []

    def solve(goal, seed):
        goals.append(goal.tolist())
        if blocked and 0.47 < float(goal[0]) < 0.50 and float(goal[2]) > 0.94:
            raise RuntimeError("Interior source-approach posture is unreachable")
        return seed

    ref = module.PhysicalReference.__new__(module.PhysicalReference)
    ref.env = SimpleNamespace(
        target=torch.zeros((1, 7)),
        robot=SimpleNamespace(
            data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.zeros(1, 7)))
        ),
    )
    ref.ctrl = SimpleNamespace(frames={"right": (None, list(range(7)), None)})
    ref.admit_stabilizer_clearance = lambda side: None
    ref.solvers = {"right": SimpleNamespace(solve=solve)}
    ref.orientation = lambda side, yaw: torch.tensor([0.0, 0.0, 0.0, 1.0])
    points = [
        [0.40, -0.40, 1.05],
        [0.46, -0.41872, 1.02],
        [0.51865, -0.41872, 0.9225],
        [0.51865, -0.41872, 0.8825],
        [0.51865, -0.41872, 0.9825],
    ]
    if blocked:
        with pytest.raises(RuntimeError, match="No reachable source grasp approach"):
            ref.grasp_yaw("right", 0, points)
    else:
        assert ref.grasp_yaw("right", 0, points) == 0
        assert goals[-1][:3] == pytest.approx(points[-1])
    assert any(0.47 < p[0] < 0.50 and p[2] > 0.94 for p in goals)


def test_source_approach_checks_zero_length_segment_without_losing_next_descent():
    points = list(module._linear_profile([0.52, -0.42, 0.92], [0.52, -0.42, 0.92]))
    assert points[-1][0] == pytest.approx([0.52, -0.42, 0.92])
    assert all(v == 0 for v in points[-1][1])
