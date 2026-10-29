"""An endpoint-safe grasp must not hide a blocked source-approach interior."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vla_lego.isaac import clearance
from vla_lego.isaac import reference as module


@pytest.mark.parametrize("diagonal", [False, True])
def test_full_open_jaw_approach_rejects_obstruction_before_motion(diagonal, monkeypatch):
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    assert Path(module.__file__).resolve().is_relative_to(root / "src")
    checked = []

    def clear(pose):
        xyz = pose[:3].tolist()
        checked.append(xyz)
        return -0.004 if 0.49 < xyz[0] < 0.515 and 0.93 < xyz[2] < 0.99 else 0.02

    monkeypatch.setattr(clearance, "source_grasp_clearance", lambda *args: clear)
    monkeypatch.setattr(clearance, "source_grasp_span", lambda *args: 0.04)
    ref = module.PhysicalReference.__new__(module.PhysicalReference)
    ref.env = SimpleNamespace(
        target=torch.zeros((1, 7)),
        robot=SimpleNamespace(
            data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.zeros(1, 7)))
        ),
    )
    ref.ctrl = SimpleNamespace(frames={"right": (None, list(range(7)), None)})
    ref.admit_stabilizer_clearance = lambda side: None
    ref.orientation = lambda side, yaw: torch.tensor([0.0, 0.0, 0.0, 1.0])
    ref.solvers = {"right": SimpleNamespace(solve=lambda goal, seed: seed)}
    points = [
        [0.40, -0.40, 1.05],
        [0.46 if diagonal else 0.51865, -0.41872, 1.02],
        [0.51865, -0.41872, 0.9225],
        [0.51865, -0.41872, 0.8825],
        [0.51865, -0.41872, 0.9825],
    ]
    assert clear(torch.tensor([*points[3], 0, 0, 0, 1])) > 0
    if diagonal:
        with pytest.raises(RuntimeError, match="Open jaws intersect along"):
            ref.grasp_yaw("right", 0, points, object_index=0)
        assert any(0.49 < xyz[0] < 0.515 and 0.93 < xyz[2] < 0.99 for xyz in checked)
    else:
        assert ref.grasp_yaw("right", 0, points, object_index=0) == 0
        assert len(checked) > 20
