"""Replay physical tower commands, separating reconstruction from task admission."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--solver-iterations", type=int, nargs=2, default=(8, 4))
    parser.add_argument("--baseline-replay", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--motion-measurement", choices=("solver_velocity", "pose_step_peaks"))
    args = parser.parse_args()
    source = json.loads((args.source_run / "report.json").read_text())
    recording = args.source_run / "recording"
    metadata = json.loads((recording / "episode.json").read_text())
    if source["learned"] or not metadata.get("control_clock_enforced"):
        raise ValueError("Tower replay requires causal physical reference commands.")
    # Historical native references explicitly used 8 position / 4 velocity iterations.
    source_iterations = tuple(source.get("physics_solver_iterations", (8, 4)))
    iterations = tuple(args.solver_iterations)
    same_physics = iterations == source_iterations
    source_measurement = source.get("motion_measurement", "solver_velocity")
    motion_measurement = args.motion_measurement or source_measurement
    if not same_physics or motion_measurement != source_measurement:
        if args.baseline_replay is None:
            parser.error("An intervention requires --baseline-replay with exact native parity.")
        baseline = json.loads((args.baseline_replay / "report.json").read_text())
        if (
            not baseline.get("exact_native_parity")
            or not baseline.get("source_physics_unchanged")
            or baseline["source_run"] != str(args.source_run.resolve())
        ):
            raise ValueError("Baseline does not prove exact reconstruction of this source.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {
        "kind": "physical_tower_command_replay",
        "learned": False,
        "passed": False,
        "qualification": False,
        "source_run": str(args.source_run.resolve()),
        "source_passed": source["passed"],
        "source_outcome_preserved": True,
        "source_physics_unchanged": same_physics,
        "source_solver_iterations": list(source_iterations),
        "physics_solver_iterations": list(iterations),
        "task": source["task"],
        "motion_measurement": motion_measurement,
        "source_motion_measurement": source_measurement,
    }
    app = recorder = None
    started = time.monotonic()
    count = 0
    errors = dict.fromkeys(
        ("joint_position", "joint_velocity", "block_pose", "block_velocity"), 0.0
    )
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        import numpy as np
        from PIL import Image

        from vla_lego.isaac.catalog import BlockPlacement, BlockSpec
        from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS, AssemblyEnvironment
        from vla_lego.isaac.evaluation import TowerEvaluator
        from vla_lego.isaac.recording import EpisodeRecorder
        from vla_lego.isaac.task import TowerTask

        raw = source["task"]
        task = TowerTask(
            tuple(
                BlockPlacement(BlockSpec(**p["spec"]), tuple(p["xy_m"])) for p in raw["placements"]
            ),
            tuple(BlockSpec(**p) for p in raw["targets"]),
            tuple(raw["assembly_xy_m"]),
        )
        env = AssemblyEnvironment(
            args.cache_dir.resolve(),
            placements=task.placements,
            assembly_xy_m=task.assembly_xy_m,
            solver_iterations=iterations,
            measure_pose_motion=motion_measurement == "pose_step_peaks",
        )
        if source["robot_joint_names"] != env.robot.data.joint_names:
            raise ValueError("Source robot joint ordering differs from the native asset.")
        report["robot_joint_names"] = env.robot.data.joint_names
        names = (*ARM_JOINTS, *JAW_JOINTS)
        if metadata["action_names"] != list(names):
            raise ValueError("Source uses a different physical action interface.")
        action_ids = [env.robot.data.joint_names.index(n) for n in names]
        state_ids = [env.robot.data.joint_names.index(n) for n in metadata["state_names"]]
        limits = env.robot.data.joint_pos_limits.torch[0, action_ids].cpu().numpy()
        env.step(round(metadata["start_time_s"] / 0.005))
        evaluator = TowerEvaluator(
            task, env.contact_robot_paths, motion_measurement=motion_measurement
        )
        evaluator.update(env.time, [env.task_truth(i) for i in range(len(env.blocks))])
        if args.record:
            # Replays are descendants: inherit the parent's family and split.
            recorder = EpisodeRecorder(
                args.output_dir / "recording",
                metadata["instruction"],
                family=metadata["generation_family"],
                split=metadata["split"],
            )
            env.recorder = recorder
        with (
            (recording / "frames.jsonl").open() as frames,
            (args.source_run / "trace.jsonl").open() as source_trace,
            (args.output_dir / "trace.jsonl").open("w") as trace,
        ):
            for frame_line, trace_line in zip(frames, source_trace, strict=True):
                row, reference = json.loads(frame_line), json.loads(trace_line)
                if row["frame"] != count or abs(row["observation_time_s"] - env.time) > 1e-8:
                    raise ValueError("Source frame/control clock differs from the replay.")
                if abs(reference["time_s"] - env.time - 0.05) > 1e-8:
                    raise ValueError("Reference evidence is not aligned to the source action.")
                velocity = env.robot.data.joint_vel.torch[0, state_ids].cpu().numpy()
                errors["joint_velocity"] = max(
                    errors["joint_velocity"],
                    float(np.max(np.abs(velocity - row["joint_velocity"]))),
                )
                action = np.asarray(row["action"], dtype=np.float32)
                if action.shape != (16,) or not np.isfinite(action).all():
                    raise ValueError("Invalid recorded robot command.")
                if recorder is None:
                    env.capture()
                action = np.clip(action, limits[:, 0], limits[:, 1])
                env.set_joint_targets(dict(zip(names, action.tolist(), strict=True)))
                env.step(10)
                robot = env.robot_diagnostics()
                truth = [env.task_truth(i) for i in range(len(env.blocks))]
                errors["joint_position"] = max(
                    errors["joint_position"],
                    float(
                        np.max(
                            np.abs(
                                np.asarray(robot["joint_position"])
                                - reference["robot"]["joint_position"]
                            )
                        )
                    ),
                )
                for key, field in (
                    ("block_pose", "block_pose_w"),
                    ("block_velocity", "block_velocity_w"),
                ):
                    errors[key] = max(
                        errors[key],
                        float(
                            np.max(
                                np.abs(
                                    np.asarray([b[field] for b in truth])
                                    - np.asarray([b[field] for b in reference["blocks"]])
                                )
                            )
                        ),
                    )
                evaluator.update(env.time, truth)
                trace.write(
                    json.dumps(
                        {
                            "stage": reference["stage"],
                            "time_s": env.time,
                            "robot": robot,
                            "applied_joint_target": env.target[0].tolist(),
                            "requested_joint_target": env.requested_target[0].tolist(),
                            "blocks": truth,
                        }
                    )
                    + "\n"
                )
                count += 1
                if count % 200 == 0:
                    trace.flush()
                    print(json.dumps({"replayed_frames": count, "time_s": env.time}), flush=True)
        if count != metadata["frames"]:
            raise ValueError("Replay did not consume the complete source recording.")
        env.capture()
        Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "final.png")
        evaluation = evaluator.result()
        exact = same_physics and all(v == 0 for v in errors.values())
        report.update(
            completed_reference_sequence=True,
            evaluation=evaluation,
            passed=evaluation["passed"]
            and (motion_measurement == source_measurement or not same_physics or exact),
            exact_native_parity=exact,
        )
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if recorder is not None:
            recorder.finish(
                verified_success=bool(report["passed"]),
                reason=(
                    "complete task admitted" if report["passed"] else "full task evaluator rejected"
                ),
            )
        report.update(
            replayed_frames=count,
            max_source_state_errors=errors,
            wall_seconds_before_app_close=time.monotonic() - started,
        )
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if app is not None:
            app.close(exit_code=0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
