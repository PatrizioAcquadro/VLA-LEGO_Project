"""Pose-step measurement: known rates, hidden interval peaks and quaternion sign."""

import math

import pytest
import torch

from vla_lego.isaac.motion import PoseMotionWindow


def pose(x=0.0, angle=0.0):
    return torch.tensor(
        [[x, 0, 0, 0, math.sin(angle / 2), 0, math.cos(angle / 2)]], dtype=torch.float64
    )


def test_known_translation_and_rotation_rates():
    window = PoseMotionWindow(pose())
    for step in range(1, 11):
        window.update(pose(0.003 * step * 0.005, 0.02 * step * 0.005), 0.005)
    assert window.duration_s == pytest.approx(0.05)
    assert window.peaks[0].tolist() == pytest.approx([0.003, 0.02])


def test_fast_motion_cannot_hide_between_identical_control_endpoints():
    window = PoseMotionWindow(pose())
    for step in range(1, 11):
        window.update(pose(0.002 if step % 2 else 0), 0.005)
    assert window.previous[0, 0] == 0
    assert window.peaks[0, 0] == pytest.approx(0.4)
    window.reset()
    window.update(pose(), 0.005)
    assert window.peaks[0, 0] == 0


def test_quaternion_double_cover_is_not_motion():
    start = pose(angle=0.4)
    window = PoseMotionWindow(start)
    for step in range(10):
        current = start.clone()
        current[:, 3:] *= -1 if step % 2 else 1
        window.update(current, 0.005)
    assert window.peaks[0, 1] == 0
