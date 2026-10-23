"""Actual rigid-transform motion peaks, independent of solver velocity fields."""

from __future__ import annotations

import math


class PoseMotionWindow:
    """Keep every physics-step speed peak until the next control-window reset."""

    def __init__(self, poses):
        self.previous = poses.detach().clone()
        self.peaks = poses.new_zeros((poses.shape[0], 2))
        self.duration_s = 0.0

    def reset(self) -> None:
        self.peaks.zero_()
        self.duration_s = 0.0

    def update(self, poses, dt_s: float) -> None:
        import torch

        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Motion samples require a positive finite interval.")
        poses = poses.detach()
        linear = (poses[:, :3] - self.previous[:, :3]).norm(dim=-1) / dt_s
        current = poses[:, 3:]
        previous = self.previous[:, 3:]
        current = current / current.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        previous = previous / previous.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # Unit-quaternion chord length = 2*sin(rotation_angle/4). Taking
        # the shorter chord makes q and -q represent identical orientations.
        chord = torch.minimum((current - previous).norm(dim=-1), (current + previous).norm(dim=-1))
        angular = 4 * torch.asin((chord / 2).clamp(0, 1)) / dt_s
        self.peaks = torch.maximum(self.peaks, torch.stack((linear, angular), dim=-1))
        self.previous.copy_(poses)
        self.duration_s += dt_s
