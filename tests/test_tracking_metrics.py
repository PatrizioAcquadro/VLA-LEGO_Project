"""Contracts between model losses and experiment metrics."""

from tracking.metrics import extract_loss_components


def test_vla_losses_preserve_total_without_double_counting():
    assert extract_loss_components({"total_loss": 3.0, "text_loss": 1.0, "action_loss": 2.0}) == {
        "loss/total": 3.0,
        "loss/ar": 1.0,
        "loss/fm": 2.0,
    }
