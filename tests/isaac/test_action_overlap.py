import pytest
import torch

from vla_lego.pretrained.worker import blend_jaw_overlap


def test_jaw_overlap_aligns_time_and_preserves_fresh_arms_and_decisions():
    current = torch.arange(1 * 50 * 18, dtype=torch.float32).reshape(1, 50, 18)
    previous = current + 1000
    unchanged = current.clone()
    served = blend_jaw_overlap(current, previous, 10)
    assert torch.equal(served[:, :, :14], current[:, :10, :14])
    assert torch.equal(served[:, :, 16:], current[:, :10, 16:])
    assert torch.equal(
        served[:, :, 14:16], (current[:, :10, 14:16] + previous[:, 10:20, 14:16]) / 2
    )
    assert torch.equal(current, unchanged)
    assert torch.equal(blend_jaw_overlap(current, None, 10), current[:, :10])


def test_jaw_overlap_leaves_uncovered_tail_fresh():
    current = torch.ones(1, 50, 18)
    previous = torch.zeros(1, 50, 18)
    served = blend_jaw_overlap(current, previous, 30)
    assert torch.equal(served[:, :20, 14:16], torch.full((1, 20, 2), 0.5))
    assert torch.equal(served[:, 20:], current[:, 20:30])
    assert torch.equal(blend_jaw_overlap(current, previous, 50), current)


def test_jaw_overlap_rejects_incompatible_shapes_and_horizons():
    current = torch.zeros(1, 50, 18)
    with pytest.raises(ValueError):
        blend_jaw_overlap(current, torch.zeros(1, 50, 16), 10)
    with pytest.raises(ValueError):
        blend_jaw_overlap(current, None, 0)
