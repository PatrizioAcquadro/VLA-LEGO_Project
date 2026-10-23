"""Analytical convex-clearance fixtures, not physical demonstrations."""

import itertools

import torch

from vla_lego.isaac.clearance import _separated


def test_face_plane_separates_shapes_whose_axis_aligned_bounds_overlap():
    cube = torch.tensor(list(itertools.product((-0.05, 0.05), repeat=3))) + 1
    tetrahedron = torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]])
    axes = torch.eye(3)
    tetra_axes = torch.tensor([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0], [1.0, 1.0, 1.0]])
    assert _separated(cube[None], axes[None], tetrahedron, tetra_axes).item()
    assert not _separated((cube - 0.8)[None], axes[None], tetrahedron, tetra_axes).item()


def test_batch_clearance_rejects_overlap_and_touching():
    cube = torch.tensor(list(itertools.product((-0.1, 0.1), repeat=3)))
    moving = cube[None] + torch.tensor([[0.0, 0, 0], [0.3, 0, 0], [0.2, 0, 0]])[:, None]
    axes = torch.eye(3)
    assert _separated(moving, axes.expand(3, -1, -1), cube, axes).tolist() == [False, True, False]
