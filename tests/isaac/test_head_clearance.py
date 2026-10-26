"""Analytical sphere projection fixtures, not native robot demonstrations."""

import itertools

import pytest
import torch

from vla_lego.isaac.clearance import _sphere_separated


@pytest.mark.parametrize(
    "center,expected", [([2.2, 0, 0], True), ([1.8, 0, 0], False), ([2, 0, 0], False)]
)
def test_exact_sphere_projection_distinguishes_clearance_overlap_and_contact(center, expected):
    points = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=3)))[None]
    axes = torch.eye(3)[None] * 2
    assert bool(_sphere_separated(points, axes, torch.tensor(center), 1.0)[0]) is expected
