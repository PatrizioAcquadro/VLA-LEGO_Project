"""Task-external geometric reference tools for physically executed demonstrations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from vla_lego.isaac.control import ToolController
from vla_lego.isaac.kinematics import ArmKinematics


class PhysicalReference:
    """Move only robot joints; callers own task choices and truth-based planning."""

    def __init__(self, env: Any, trace, output_dir: Path, on_sample=None, *, event_settling=False):
        self.env, self.trace, self.output_dir, self.on_sample = env, trace, output_dir, on_sample
        self.ctrl = ToolController(env)
        self.initial_orientations = {
            side: self.ctrl.pose(side)[0, 3:].clone() for side in ("left", "right")
        }
        self.solvers = {
            side: ArmKinematics(
                Path(env.robot.cfg.spawn.asset_path),
                side,
                env.robot.data.root_pose_w.torch[0],
                "cuda:0",
            )
            for side in ("left", "right")
        }
        self.stages: list[dict] = []
        self.max_limit_projection = 0.0
        self.event_settling = event_settling

    def grasp_yaw(self, side: str, preferred: float, waypoints: list[list[float]]) -> float:
        """Choose a reachable equivalent jaw approach for square-section blocks."""
        _, ids, _ = self.ctrl.frames[side]
        failures = []
        for offset in (0, 90, -90, 180):
            yaw = preferred + offset
            seed = self.env.robot.data.joint_pos.torch[0, ids]
            try:
                for xyz in waypoints:
                    seed = self.solvers[side].solve(
                        self.env.target.new_tensor([*xyz, *self.orientation(side, yaw).tolist()]),
                        seed,
                    )
                return yaw
            except RuntimeError as error:
                failures.append(str(error))
        raise RuntimeError(f"No reachable square-face grasp approach: {failures}")

    def orientation(self, side: str, yaw_deg: float):
        from isaaclab.utils.math import quat_mul

        half = math.radians(yaw_deg) / 2
        yaw = self.env.target.new_tensor([0, 0, math.sin(half), math.cos(half)])
        return quat_mul(yaw, self.initial_orientations[side])

    def sample(self, stage: str) -> None:
        truth = [self.env.task_truth(i) for i in range(len(self.env.blocks))]
        self.trace.write(
            json.dumps(
                {
                    "stage": stage,
                    "time_s": self.env.time,
                    "tools": {side: self.ctrl.pose(side)[0].tolist() for side in self.solvers},
                    "robot": self.env.robot_diagnostics(),
                    "applied_joint_target": self.env.target[0].tolist(),
                    "requested_joint_target": self.env.requested_target[0].tolist(),
                    "blocks": truth,
                }
            )
            + "\n"
        )
        if self.on_sample is not None:
            self.on_sample(stage, truth)

    def wait(self, ticks: int, stage: str) -> None:
        if self.env.recorder is not None:
            self.env.recorder.stage = stage
        for _ in range(ticks):
            self.env.step(10)
            self.sample(stage)

    def settle(self, ids, speed: float, max_ticks: int, stage: str) -> None:
        quiet = 0
        for _ in range(max_ticks):
            self.wait(1, stage)
            velocity = self.env.robot.data.joint_vel.torch[0, ids]
            quiet = quiet + 1 if float(velocity.abs().max()) < speed else 0
            if quiet >= 3:
                break

    def joint_move(self, stage: str, side: str, goal) -> None:
        import torch

        if self.env.recorder is not None:
            self.env.recorder.stage = stage
        _, ids, _ = self.ctrl.frames[side]
        limits = self.env.robot.data.joint_pos_limits.torch[0, ids]
        start = self.env.robot.data.joint_pos.torch[0, ids].clamp(limits[:, 0], limits[:, 1])
        bounded_goal = goal.clamp(limits[:, 0], limits[:, 1])
        self.max_limit_projection = max(
            self.max_limit_projection, float((bounded_goal - goal).abs().max())
        )
        goal = bounded_goal
        ticks = max(1, math.ceil(float((goal - start).abs().max()) / 0.025))
        for tick in range(ticks):
            q = torch.lerp(start, goal, (tick + 1) / ticks).clamp(limits[:, 0], limits[:, 1])
            self.env.set_joint_targets(dict(zip(self.solvers[side].names, q.tolist(), strict=True)))
            self.env.step(10)
            self.sample(stage)
        if self.event_settling:
            self.settle(ids, 0.02, 40, stage + "_settle")
        else:
            self.wait(40, stage + "_settle")

    def park(self, side: str) -> None:
        self.joint_move(
            side + "_park",
            side,
            self.env.target.new_tensor(
                [0.8, 0.35 if side == "left" else -0.35, 0.0, -1.8, 0.0, 0.0, 0.0]
            ),
        )

    def move(
        self,
        stage: str,
        side: str,
        xyz: list[float],
        yaw_deg: float = 0.0,
        *,
        quaternion_xyzw: list[float] | None = None,
        linear: bool = False,
        linear_step_m: float = 0.004,
    ) -> None:
        from PIL import Image

        orientation = (
            self.orientation(side, yaw_deg).tolist() if quaternion_xyzw is None else quaternion_xyzw
        )
        desired = [*xyz, *orientation]
        _, ids, _ = self.ctrl.frames[side]
        if linear:
            if not 0 < linear_step_m <= 0.02:
                raise ValueError("Cartesian waypoint spacing must be in (0, 0.02] meters.")
            start = self.ctrl.pose(side)[0, :3].tolist()
            count = max(1, math.ceil(math.dist(start, xyz) / linear_step_m))
            for waypoint in range(1, count):
                intermediate = [
                    a + (b - a) * waypoint / count for a, b in zip(start, xyz, strict=True)
                ]
                solution = self.solvers[side].solve(
                    self.env.target.new_tensor([*intermediate, *orientation]),
                    self.env.robot.data.joint_pos.torch[0, ids],
                )
                self.joint_move(stage, side, solution)
        solution = self.solvers[side].solve(
            self.env.target.new_tensor(desired), self.env.robot.data.joint_pos.torch[0, ids]
        )
        self.joint_move(stage, side, solution)
        for _ in range(200):
            feedback = self.ctrl.command({side: desired}, integrate_target=True)[side]
            self.sample(stage + "_refine")
            if feedback["position_error_m"] < 0.003 and feedback["rotation_error_rad"] < 0.04:
                break
        actual = self.ctrl.pose(side)[0]
        position_error = float((actual[:3] - actual.new_tensor(xyz)).norm())
        result = {
            "stage": stage,
            "side": side,
            "desired": desired,
            "actual": actual.tolist(),
            "position_error_m": position_error,
            "rotation_error_rad": feedback["rotation_error_rad"],
        }
        self.stages.append(result)
        self.trace.flush()
        self.env.capture()
        Image.fromarray(self.env.observe()["head_rgb"]).save(self.output_dir / f"{stage}.png")
        print(json.dumps(result), flush=True)
        if position_error >= 0.005 or feedback["rotation_error_rad"] >= 0.05:
            raise RuntimeError(f"Native reference pose did not converge at {stage}")

    def grip(self, stage: str, side: str, width_m: float | None) -> None:
        # Measured inner gap = 2*(leader+.001); retain the validated 1 mm preload.
        target = 0.034 if width_m is None else width_m / 2 - 0.002
        self.env.set_joint_targets({f"{side}_WSG32_JAW_OPENING": target})
        if self.event_settling:
            self.settle(self.env.jaw_ids, 0.001, 20, stage)
        else:
            self.wait(20, stage)
