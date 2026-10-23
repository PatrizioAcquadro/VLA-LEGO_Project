"""Contact-only reference diagnostic, retaining every stage and failure."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--family")
    parser.add_argument("--split", choices=("train", "development"))
    args = parser.parse_args()
    if args.record and (not args.family or not args.split):
        parser.error("--record requires --family and --split before collection")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    app = None
    recorder = None
    report: dict = {"kind": "reference_grasp_probe", "learned": False, "stages": []}
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        import torch
        from PIL import Image

        from vla_lego.isaac.control import ToolController
        from vla_lego.isaac.environment import TABLE_TOP, AssemblyEnvironment
        from vla_lego.isaac.kinematics import ArmKinematics
        from vla_lego.isaac.recording import EpisodeRecorder

        env = AssemblyEnvironment(args.cache_dir.resolve())
        ctrl = ToolController(env)
        env.step(400)
        if args.record:
            recorder = EpisodeRecorder(
                args.output_dir / "recording",
                "Pick up the small red cube and place it back on the table in the open space to the left.",
                family=args.family,
                split=args.split,
            )
            env.recorder = recorder
        initial = {side: ctrl.pose(side)[0].tolist() for side in ("left", "right")}
        report["initial_tool_poses_xyzw"] = initial
        report["initial_truth"] = env.task_truth()
        solvers = {
            side: ArmKinematics(
                Path(env.robot.cfg.spawn.asset_path),
                side,
                env.robot.data.root_pose_w.torch[0],
                "cuda:0",
            )
            for side in ("left", "right")
        }
        with (args.output_dir / "trace.jsonl").open("w") as trace:

            def joint_move(name: str, side: str, q_goal) -> None:
                _, ids, _ = ctrl.frames[side]
                q_start = env.robot.data.joint_pos.torch[0, ids].clone()
                ticks = max(1, round(float((q_goal - q_start).abs().max()) / 0.5 / 0.05))
                for tick in range(ticks):
                    q = torch.lerp(q_start, q_goal, (tick + 1) / ticks)
                    env.set_joint_targets(dict(zip(solvers[side].names, q.tolist(), strict=True)))
                    env.step(10)
                    trace.write(
                        json.dumps(
                            {
                                "stage": name,
                                "t": env.time,
                                "joint_targets": q.tolist(),
                                "actual_pose": ctrl.pose(side)[0].tolist(),
                                "truth": env.task_truth(),
                                "robot": env.robot_diagnostics(),
                            }
                        )
                        + "\n"
                    )
                env.step(400)
                trace.flush()

            def move(name: str, side: str, position: list[float], seconds: float) -> bool:
                desired = [*position, *initial[side][3:]]
                _, ids, _ = ctrl.frames[side]
                solution = solvers[side].solve(
                    env.target.new_tensor(desired), env.robot.data.joint_pos.torch[0, ids]
                )
                joint_move(name + "_transit", side, solution)
                for tick in range(round(seconds / 0.05)):
                    feedback = ctrl.command({side: desired})[side]
                    trace.write(
                        json.dumps(
                            {
                                "stage": name,
                                "t": env.time,
                                **feedback,
                                "actual_pose": ctrl.pose(side)[0].tolist(),
                                "truth": env.task_truth(),
                                "robot": env.robot_diagnostics(),
                            }
                        )
                        + "\n"
                    )
                    if tick % 20 == 0:
                        trace.flush()
                    if (
                        feedback["position_error_m"] < 0.003
                        and feedback["rotation_error_rad"] < 0.04
                    ):
                        break
                env.capture()
                Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / f"{name}.png")
                current = ctrl.pose(side)[0]
                position_error = float((current[:3] - current.new_tensor(position)).norm())
                result = {
                    "stage": name,
                    "actual_pose": current.tolist(),
                    "position_error_m": position_error,
                    "rotation_error_rad": feedback["rotation_error_rad"],
                    "truth": env.task_truth(),
                }
                report["stages"].append(result)
                print(json.dumps(result), flush=True)
                return position_error < 0.005 and feedback["rotation_error_rad"] < 0.05

            # Fold behind the table before extending over it; object state is untouched.
            for side in ("left", "right"):
                parked = env.target.new_tensor(
                    [0.8, 0.35 if side == "left" else -0.35, 0.0, -1.8, 0.0, 0.0, 0.0]
                )
                joint_move(f"{side}_park", side, parked)
                env.capture()
                Image.fromarray(env.observe()["head_rgb"]).save(
                    args.output_dir / f"{side}_park.png"
                )
                pose = ctrl.pose(side)[0].tolist()
                report["stages"].append(
                    {"stage": f"{side}_park", "actual_pose": pose, "truth": env.task_truth()}
                )
                if pose[0] > 0.25 or pose[2] < TABLE_TOP + 0.08:
                    raise RuntimeError(f"{side} staging posture lacks table clearance")
            block = env.task_truth()["block_pose_w"][:3]
            for name, position in (
                ("over_edge", [0.40, 0.40, 1.05]),
                ("center_above_table", [0.40, 0.15, 1.05]),
                ("above_block", [block[0], block[1], 1.02]),
            ):
                if not move(name, "left", position, 10):
                    raise RuntimeError(f"Reference staging failed at {name}")
            if not move("approach", "left", [block[0], block[1], block[2] + 0.10], 10):
                raise RuntimeError("Reference pregrasp is unreachable under current servo")
            # UMI contact geometry extends 71.5 mm beyond the shared TCP.
            # For a 40 mm block, this offset leaves 8.5 mm table clearance
            # while overlapping 31.5 mm of the block's vertical side.
            grasp = [block[0], block[1], block[2] + 0.06]
            if not move("descend", "left", grasp, 5):
                raise RuntimeError("Reference descent did not converge")
            env.set_joint_targets({"left_WSG32_JAW_OPENING": 0.018})
            env.step(200)
            grasp_truth = env.task_truth()
            lifted = move("lift", "left", [grasp[0], grasp[1], grasp[2] + 0.10], 6)
            lift_truth = env.task_truth()
            if not lifted or lift_truth["block_pose_w"][2] < block[2] + 0.07:
                raise RuntimeError("No verified physical lift")
            destination = [block[0] + 0.03, block[1] + 0.08, grasp[2] + 0.10]
            if not move("transport", "left", destination, 6):
                raise RuntimeError("Reference transport failed")
            destination[2] = grasp[2]
            if not move("place", "left", destination, 5):
                raise RuntimeError("Reference placement failed")
            env.set_joint_targets({"left_WSG32_JAW_OPENING": 0.034})
            env.step(200)
            destination[2] += 0.12
            if not move("retreat", "left", destination, 5):
                raise RuntimeError("Reference retreat failed")
            dwell = []
            for _ in range(10):
                env.step(40)
                dwell.append(env.task_truth())
            final = dwell[-1]
            stable_release = all(
                abs(sample["block_pose_w"][2] - (TABLE_TOP + 0.02)) < 0.003
                and abs(sample["block_pose_w"][0] - destination[0]) < 0.01
                and abs(sample["block_pose_w"][1] - destination[1]) < 0.01
                and sample["robot_contact_force_n"] < 1e-6
                and sum(v * v for v in sample["block_velocity_w"][:3]) < 0.01**2
                and sum(v * v for v in sample["block_velocity_w"][3:]) < 0.1**2
                for sample in dwell
            )
            report.update(
                grasp_truth=grasp_truth,
                lift_truth=lift_truth,
                final_truth=final,
                release_dwell=dwell,
                passed=stable_release,
            )
    except Exception:
        report.update(passed=False, error=traceback.format_exc())
        raise
    finally:
        if recorder is not None:
            recorder.finish(
                verified_success=bool(report.get("passed")), reason=report.get("error", "complete")
            )
        report["wall_seconds_before_app_close"] = time.monotonic() - started
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if app is not None:
            app.close(exit_code=0 if report.get("passed") else 1)


if __name__ == "__main__":
    main()
