"""Physical single-block scoring without weakening five-block qualification."""

from __future__ import annotations

import math
from collections import deque

from vla_lego.isaac.environment import TABLE_TOP
from vla_lego.isaac.evaluation import (
    CONTACT_ZERO_N,
    DWELL_DISPLACEMENT_M,
    DWELL_S,
    PERIOD_S,
    POSITION_TOLERANCE_M,
    _corners,
    _norm,
)
from vla_lego.isaac.task import SingleBlockTask


class PickPlaceEvaluator:
    def __init__(self, task: SingleBlockTask, robot_contact_paths: list[str]):
        self.task = task
        self.index = next(i for i, p in enumerate(task.placements) if p.spec == task.targets[0])
        self.jaws = [
            [
                i
                for i, path in enumerate(robot_contact_paths)
                if path.rsplit("/", 1)[-1]
                in (f"{side}_WSG32_NEGATIVE_JAW_LINK", f"{side}_WSG32_POSITIVE_JAW_LINK")
            ]
            for side in ("left", "right")
        ]
        if any(len(ids) != 2 for ids in self.jaws):
            raise ValueError("Single-block scoring requires both named jaw pairs")
        self.history: deque = deque(maxlen=round(DWELL_S / PERIOD_S) + 1)
        self.start: tuple[float, ...] | None = None
        if task.build_region_xy_m is None:
            raise ValueError("A single-block task requires the declared clear build region")
        self.region = task.build_region_xy_m
        self.grasped_lift = False
        self.transported = False
        self.samples = 0

    def update(self, time_s: float, blocks: list[dict], *, tool_positions: dict | None = None):
        state = blocks[self.index]
        pose = state["block_pose_w"]
        if self.start is None:
            self.start = tuple(pose[:3])
        if self.history and time_s - self.history[-1][0] > PERIOD_S + 1e-8:
            self.history.clear()
        bilateral = any(
            min(_norm(state["robot_contact_forces_n"][i]) for i in ids) > CONTACT_ZERO_N
            and sum(_norm(state["robot_contact_forces_n"][i]) for i in ids) > 0.5
            for ids in self.jaws
        )
        self.grasped_lift |= bilateral and pose[2] - self.start[2] >= 0.07
        self.transported |= self.grasped_lift and math.dist(pose[:2], self.start[:2]) >= 0.08
        x0, x1, y0, y1 = self.region
        finite = all(math.isfinite(v) for v in (*pose, *state["block_velocity_w"]))
        corners = _corners(self.task.targets[0].dimensions_m, pose) if finite else ()
        placed = bool(corners) and (
            x0 <= pose[0] <= x1
            and y0 <= pose[1] <= y1
            and abs(min(p[2] for p in corners) - TABLE_TOP) < POSITION_TOLERANCE_M
            and state["nonrobot_contact_force_n"] > CONTACT_ZERO_N
            and all(b["robot_contact_force_n"] < CONTACT_ZERO_N for b in blocks)
            and tool_positions is not None
            and all(
                min(math.dist(tool[:3], b["block_pose_w"][:3]) for b in blocks) >= 0.10
                for tool in tool_positions.values()
            )
        )
        self.history.append((time_s, placed, corners))
        self.samples += 1

    def result(self) -> dict:
        complete = (
            len(self.history) == self.history.maxlen
            and self.history[-1][0] - self.history[0][0] >= DWELL_S - 1e-8
        )
        displacement = math.inf
        if complete and all(row[1] for row in self.history):
            displacement = max(
                math.dist(a, b)
                for row in self.history
                for a, b in zip(self.history[0][2], row[2], strict=True)
            )
        checks = {
            "correct_bilateral_grasp_lift": self.grasped_lift,
            "transported": self.transported,
            "complete_dwell": complete,
            "supported_released_withdrawn": complete and all(r[1] for r in self.history),
            "stable": displacement <= DWELL_DISPLACEMENT_M,
        }
        return {
            "passed": all(checks.values()),
            "task_kind": "single_block",
            "checks": checks,
            "dwell_corner_displacement_m": displacement if math.isfinite(displacement) else None,
            "cooperation_events": [],
            "samples": self.samples,
            "max_ordered_supported_levels": int(checks["supported_released_withdrawn"]),
            "qualification": False,
        }
