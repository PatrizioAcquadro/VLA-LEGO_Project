"""Analytical convex-clearance fixtures, not physical demonstrations."""

import itertools

import pytest
import torch

from vla_lego.isaac.clearance import _mesh_axes, _separated, _separation_distance


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
    assert torch.allclose(
        _separation_distance(moving, axes.expand(3, -1, -1), cube, axes),
        torch.tensor([-0.2, 0.1, 0.0]),
        atol=1e-7,
    )


def test_mesh_face_planes_clear_a_box_beside_a_sloped_finger():
    finger = torch.tensor(
        [[0.0, 0, -0.1], [1.0, 0, -0.1], [0, 1.0, -0.1], [0, 0, 0.1], [1.0, 0, 0.1], [0, 1.0, 0.1]]
    )
    normals, _ = _mesh_axes(
        finger,
        [3, 3, 4, 4, 4],
        [0, 2, 1, 3, 4, 5, 0, 1, 4, 3, 1, 2, 5, 4, 2, 0, 3, 5],
    )
    box = torch.tensor(list(itertools.product((-0.04, 0.04), repeat=3)))
    axes = torch.eye(3)
    clear = box + torch.tensor([0.6, 0.6, 0.0])
    assert not _separated(finger[None], axes[None], clear, axes).item()
    assert _separation_distance(finger[None], normals[None], clear, axes).item() > 0.08
    assert not _separated(
        finger[None], normals[None], box + torch.tensor([0.4, 0.4, 0.0]), axes
    ).item()


def test_source_span_accounts_for_actual_piece_yaw_and_oblique_grasp():
    import math
    from types import SimpleNamespace

    from vla_lego.isaac.clearance import source_grasp_span

    env = SimpleNamespace(
        placements=[SimpleNamespace(spec=SimpleNamespace(dimensions_m=(0.047, 0.047, 0.068)))],
        task_truth=lambda _: {"block_pose_w": [0.5, 0.4, 0.834, 0, 0, 0, 1]},
    )
    assert source_grasp_span(env, 0, -90) == pytest.approx(0.047)
    assert source_grasp_span(env, 0, -100) + 0.006 < 0.070
    assert source_grasp_span(env, 0, -135) + 0.006 > 0.070

    env.task_truth = lambda _: {
        "block_pose_w": [0.5, 0.4, 0.834, 0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8)]
    }
    assert source_grasp_span(env, 0, 45) == pytest.approx(0.047)
    assert source_grasp_span(env, 0, 0) + 0.006 > 0.070
