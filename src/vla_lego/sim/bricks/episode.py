"""Reusable seeded reset for free lightweight bricks and an optional Alex robot."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, cos, isfinite, sin
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from vla_lego.sim.bricks.scene import TABLE_CENTER, TABLE_HALF_SIZE, TABLE_TOP, scene_xml
from vla_lego.sim.bricks.specs import BrickSpec, load_bricks

if TYPE_CHECKING:
    from vla_lego.sim.bricks.task import AssemblyGoal


class EpisodeManager:
    """Compile once; reset the scene or initialize an explicitly diagnostic layout."""

    def __init__(self, bricks: Sequence[BrickSpec] | None = None, *, include_robot=True):
        self.bricks = tuple(load_bricks() if bricks is None else bricks)
        self.model = mujoco.MjModel.from_xml_string(
            scene_xml(self.bricks, include_robot=include_robot)
        )
        self.data = mujoco.MjData(self.model)
        self.body_ids = tuple(self.model.body(f"brick_{i}").id for i in range(len(self.bricks)))
        self.geom_ids = tuple(
            self.model.geom(f"brick_{i}_geom").id for i in range(len(self.bricks))
        )
        self.qpos_addresses = tuple(
            self.model.joint(f"brick_{i}_joint").qposadr[0] for i in range(len(self.bricks))
        )
        self.reset(seed=0)

    def reset(self, seed: int = 0, *, layout: AssemblyGoal | None = None) -> None:
        """Spawn separated bricks. A supplied layout is scene setup, not robot action."""
        mujoco.mj_resetData(self.model, self.data)
        rest = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "rest")
        if rest >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, rest)
        if layout is not None:
            if sorted(t.brick_index for t in layout.targets) != list(range(len(self.bricks))):
                raise ValueError("A diagnostic layout must contain each brick exactly once")
            for target in layout.targets:
                self._set_pose(target.brick_index, target.position, target.yaw)
        else:
            rng = np.random.default_rng(seed)
            occupied: list[tuple[np.ndarray, np.ndarray]] = []
            for i, spec in enumerate(self.bricks):
                yaw = float(rng.uniform(-np.pi, np.pi))
                sx, sy = spec.size[:2]
                half = (
                    np.array(
                        (
                            abs(cos(yaw)) * sx + abs(sin(yaw)) * sy,
                            abs(sin(yaw)) * sx + abs(cos(yaw)) * sy,
                        )
                    )
                    / 2
                )
                bound = np.array(TABLE_HALF_SIZE) - half - 0.01
                if np.any(bound <= 0):
                    raise ValueError(f"Brick {spec.name} does not fit the workspace")
                for _ in range(200):
                    xy = np.array(TABLE_CENTER) + rng.uniform(-bound, bound)
                    if all(np.any(np.abs(xy - p) > half + h + 0.005) for p, h in occupied):
                        occupied.append((xy, half))
                        self._set_pose(i, (*xy, TABLE_TOP + spec.size[2] / 2 + 0.002), yaw)
                        break
                else:
                    raise ValueError("Cannot spawn separated bricks in the workspace")
        mujoco.mj_forward(self.model, self.data)
        self.step(0.4)

    def _set_pose(self, index, position, yaw):
        address = self.qpos_addresses[index]
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = (cos(yaw / 2), 0, 0, sin(yaw / 2))

    def step(self, seconds: float) -> None:
        """Advance native MuJoCo physics while holding current actuator commands."""
        if not isfinite(seconds) or seconds < 0:
            raise ValueError("Duration must be finite and nonnegative")
        for _ in range(ceil(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Non-finite simulation state")
