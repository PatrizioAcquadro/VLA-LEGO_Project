"""Screen the initial tower's fixed source/release poses before physical execution."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--yaw-deg", type=float, nargs="+", default=[0.0])
    parser.add_argument("--levels", type=int, nargs="+", choices=range(5), default=list(range(5)))
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("source_approach", "source_grasp", "stack_release", "stack_clearance"),
        default=["source_approach", "source_grasp", "stack_release"],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {"kind": "tower_ik_screen", "physical_execution": False, "poses": []}
    app = None
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        from isaaclab.utils.math import quat_mul

        from vla_lego.isaac.control import ToolController
        from vla_lego.isaac.environment import TABLE_TOP, AssemblyEnvironment
        from vla_lego.isaac.kinematics import ArmKinematics
        from vla_lego.isaac.task import initial_tower_task

        task = initial_tower_task()
        env = AssemblyEnvironment(
            args.cache_dir.resolve(), placements=task.placements, assembly_xy_m=task.assembly_xy_m
        )
        env.step(400)
        ctrl = ToolController(env)
        solvers = {
            side: ArmKinematics(
                Path(env.robot.cfg.spawn.asset_path),
                side,
                env.robot.data.root_pose_w.torch[0],
                "cuda:0",
            )
            for side in ("left", "right")
        }
        poses = []
        bottom = TABLE_TOP
        for level, target in enumerate(task.targets):
            index = next(i for i, p in enumerate(task.placements) if p.spec == target)
            xyz = env.task_truth(index)["block_pose_w"][:3]
            side = "left" if xyz[1] >= 0 else "right"
            for phase, offset in (("source_approach", 0.10), ("source_grasp", 0.06)):
                poses.append((level, phase, side, [xyz[0], xyz[1], xyz[2] + offset]))
            for side in ("left", "right"):
                for phase, clearance in (("stack_release", 0.0), ("stack_clearance", 0.04)):
                    poses.append(
                        (
                            level,
                            phase,
                            side,
                            [
                                *task.assembly_xy_m,
                                bottom + target.dimensions_m[2] / 2 + 0.06 + clearance,
                            ],
                        )
                    )
            bottom += target.dimensions_m[2]
        for level, phase, side, xyz in poses:
            if level not in args.levels or phase not in args.phases:
                continue
            for yaw_deg in args.yaw_deg:
                half_yaw = math.radians(yaw_deg) / 2
                yaw = env.target.new_tensor([0, 0, math.sin(half_yaw), math.cos(half_yaw)])
                orientation = quat_mul(yaw, ctrl.pose(side)[0, 3:])
                desired = [*xyz, *orientation.tolist()]
                seed = env.target.new_tensor(
                    [0.8, 0.35 if side == "left" else -0.35, 0.0, -1.8, 0.0, 0.0, 0.0]
                )
                result = {
                    "level": level,
                    "phase": phase,
                    "side": side,
                    "pose": desired,
                    "yaw_deg": yaw_deg,
                }
                try:
                    solution = solvers[side].solve(env.target.new_tensor(desired), seed)
                    result.update(solved=True, joint_solution=solution.tolist())
                except RuntimeError as error:
                    result.update(solved=False, reason=str(error))
                report["poses"].append(result)
                print(json.dumps(result), flush=True)
        report.update(
            passed=all(p["solved"] for p in report["poses"]),
            solved=sum(p["solved"] for p in report["poses"]),
            attempted=len(report["poses"]),
            component_tolerances={"position_m": 0.003, "orientation_rad": 0.04},
            contact_feasibility=None,
        )
    except Exception:
        report.update(passed=False, error=traceback.format_exc())
        raise
    finally:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if app is not None:
            app.close(exit_code=0 if report.get("passed") else 1)


if __name__ == "__main__":
    main()
