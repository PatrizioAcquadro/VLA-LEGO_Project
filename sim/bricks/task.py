"""Tower/planar-triangle goals and independent physical outcome evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, isfinite, pi, sin

import mujoco
import numpy as np

from sim.bricks.episode import EpisodeManager
from sim.bricks.scene import TABLE_CENTER, TABLE_TOP
from sim.bricks.specs import BrickSpec


@dataclass(frozen=True)
class PlacementTarget:
    brick_index: int
    position: tuple[float, float, float]
    yaw: float = 0.0
    support_index: int | None = None


@dataclass(frozen=True)
class AssemblyGoal:
    shape: str
    targets: tuple[PlacementTarget, ...]


@dataclass(frozen=True)
class PlacementResult:
    brick_index: int
    position_error_m: float
    orientation_error_rad: float
    supported: bool
    stable: bool
    success: bool


@dataclass(frozen=True)
class AssemblyResult:
    placements: tuple[PlacementResult, ...]

    @property
    def success(self) -> bool:
        return bool(self.placements) and all(p.success for p in self.placements)


def make_goal(bricks: Sequence[BrickSpec], shape: str = "tower") -> AssemblyGoal:
    """Build a centered tower or three separated placements forming a triangle."""
    if not bricks:
        raise ValueError("A goal needs at least one brick")
    targets = []
    if shape == "tower":
        bottom = TABLE_TOP
        support = None
        order = sorted(range(len(bricks)), key=lambda i: np.prod(bricks[i].size[:2]), reverse=True)
        for i in order:
            height = bricks[i].size[2]
            targets.append(
                PlacementTarget(i, (*TABLE_CENTER, bottom + height / 2), support_index=support)
            )
            bottom += height
            support = i
    elif shape == "triangle":
        if len(bricks) != 3:
            raise ValueError("The planar triangle needs exactly three bricks")
        radius = max(np.hypot(*b.size[:2]) for b in bricks)
        for i, spec in enumerate(bricks):
            angle = pi / 2 + 2 * pi * i / 3
            position = (
                TABLE_CENTER[0] + radius * cos(angle),
                TABLE_CENTER[1] + radius * sin(angle),
                TABLE_TOP + spec.size[2] / 2,
            )
            targets.append(PlacementTarget(i, position, angle + pi / 2))
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return AssemblyGoal(shape, tuple(targets))


def _placement(episode, target, position_tolerance, orientation_tolerance):
    model, data = episode.model, episode.data
    i = target.brick_index
    body = episode.body_ids[i]
    position_error = float(np.linalg.norm(data.xpos[body] - target.position))
    spec = episode.bricks[i]
    symmetry = pi / 2 if np.isclose(spec.size[0], spec.size[1]) else pi
    errors = []
    for angle in np.arange(target.yaw, target.yaw + 2 * pi, symmetry):
        q = (cos(angle / 2), 0, 0, sin(angle / 2))
        errors.append(2 * np.arccos(np.clip(abs(np.dot(data.xquat[body], q)), 0, 1)))
    orientation_error = float(min(errors))
    support = (
        model.geom("table_surface").id
        if target.support_index is None
        else episode.geom_ids[target.support_index]
    )
    supported = False
    robot_contact = False
    force = np.zeros(6)
    brick_geom = episode.geom_ids[i]
    for j, contact in enumerate(data.contact):
        geoms = (contact.geom1, contact.geom2)
        if brick_geom not in geoms:
            continue
        other = contact.geom2 if contact.geom1 == brick_geom else contact.geom1
        other_body = model.geom_bodyid[other]
        if other_body != 0 and other_body not in episode.body_ids:
            robot_contact = True
        if other == support and abs(contact.frame[2]) > 0.5:
            mujoco.mj_contactForce(model, data, j, force)
            supported |= force[0] > 0
    joint = model.joint(f"brick_{i}_joint")
    velocity = data.qvel[joint.dofadr[0] : joint.dofadr[0] + 6]
    stable = bool(np.linalg.norm(velocity[:3]) <= 0.01 and np.linalg.norm(velocity[3:]) <= 0.1)
    success = (
        position_error <= position_tolerance
        and orientation_error <= orientation_tolerance
        and supported
        and stable
        and not robot_contact
    )
    return PlacementResult(
        i, position_error, orientation_error, bool(supported), stable, bool(success)
    )


def evaluate_goal(
    episode: EpisodeManager,
    goal: AssemblyGoal,
    *,
    position_tolerance: float | None = None,
    orientation_tolerance: float = pi / 36,
    stability_seconds: float = 0.25,
) -> AssemblyResult:
    """Sample pose/support/motion over a window; advances physics without assisting bricks.

    Defaults are diagnostic calibration bounds, not physical-transfer qualification.
    Every brick must pass at every sample. Simulator truth stays in this evaluator.
    """
    if sorted(t.brick_index for t in goal.targets) != list(range(len(episode.bricks))):
        raise ValueError("A goal must evaluate each brick exactly once")
    if position_tolerance is None:
        position_tolerance = 0.1 * min(min(b.size) for b in episode.bricks)
    values = (position_tolerance, orientation_tolerance, stability_seconds)
    if any(not isfinite(v) or v <= 0 for v in values):
        raise ValueError("Evaluation bounds and stability window must be finite and positive")
    history = []
    for sample in range(6):
        if sample:
            episode.step(stability_seconds / 5)
        history.append(
            tuple(
                _placement(episode, t, position_tolerance, orientation_tolerance)
                for t in goal.targets
            )
        )
    placements = []
    for j, target in enumerate(goal.targets):
        samples = [frame[j] for frame in history]
        placements.append(
            PlacementResult(
                target.brick_index,
                max(p.position_error_m for p in samples),
                max(p.orientation_error_rad for p in samples),
                all(p.supported for p in samples),
                all(p.stable for p in samples),
                all(p.success for p in samples),
            )
        )
    return AssemblyResult(tuple(placements))
