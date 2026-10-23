"""Bounded autonomous single-block diagnostic using a separately loaded VLA."""

from __future__ import annotations

import argparse
import json
import socket
import time
import traceback
from pathlib import Path

from vla_lego.pretrained.transport import receive_message, send_message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=1200)
    args = parser.parse_args()
    if args.ticks < 1:
        parser.error("--ticks must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {
        "kind": "autonomous_single_block_diagnostic",
        "learned": True,
        "teacher_commands": False,
        "simulation_paused_during_inference": True,
        "passed": False,
        "tick_budget": args.ticks,
        "learned_completion_signal": False,
        "autonomy_started": False,
        "head_capture_hz": 20,
    }
    app = None
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(120)
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        import numpy as np
        from PIL import Image

        from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS, AssemblyEnvironment

        connection.connect(str(args.socket))
        env = AssemblyEnvironment(args.cache_dir.resolve())
        env.step(400)
        report["robot_joint_names"] = env.robot.data.joint_names
        action_names = (*ARM_JOINTS, *JAW_JOINTS)
        action_ids = [env.robot.data.joint_names.index(name) for name in action_names]
        limits = env.robot.data.joint_pos_limits.torch[0, action_ids].cpu().numpy()
        instruction = "Pick up the small red cube and place it back on the table in the open space to the left."
        chunk: list = []
        representation: str | None = None
        inference_seconds = []
        clipped_ticks = 0
        clipped_counts = np.zeros(16, dtype=np.int64)
        max_projection = np.zeros(16, dtype=np.float32)
        initial_height = env.task_truth()["block_pose_w"][2]
        max_contact_lift = 0.0
        report["autonomy_started"] = True
        with (args.output_dir / "trace.jsonl").open("w") as trace:
            for tick in range(args.ticks):
                # Match the collection camera stream even between model replans.
                env.capture()
                if not chunk:
                    observation = env.observe()
                    send_message(
                        connection,
                        {
                            "state": observation["joint_position"].tolist(),
                            "state_names": list(observation["joint_names"]),
                            "velocity": observation["joint_velocity"].tolist(),
                            "image_shape": list(observation["head_rgb"].shape),
                            "instruction": instruction,
                        },
                        observation["head_rgb"].tobytes(),
                    )
                    response, _ = receive_message(connection)
                    current_representation = response["action_representation"]
                    if current_representation not in ("absolute", "measured_arm_offset"):
                        raise ValueError("Unknown learned command representation.")
                    if representation is not None and current_representation != representation:
                        raise ValueError("Command representation changed inside an episode.")
                    representation = current_representation
                    chunk = response["actions"]
                    if not chunk:
                        raise ValueError("Empty learned action chunk.")
                    inference_seconds.append(response["inference_seconds"])
                    if tick % 200 == 0:
                        Image.fromarray(observation["head_rgb"]).save(
                            args.output_dir / f"head-{tick:04d}.png"
                        )
                action = np.asarray(chunk.pop(0), dtype=np.float32)
                if action.shape != (16,) or not np.isfinite(action).all():
                    raise ValueError("Expected exactly sixteen finite learned joint targets.")
                requested = action.copy()
                if representation == "measured_arm_offset":
                    requested[:14] += (
                        env.robot.data.joint_pos.torch[0, action_ids[:14]].cpu().numpy()
                    )
                bounded = np.clip(requested, limits[:, 0], limits[:, 1])
                clipped_ticks += int(np.any(requested != bounded))
                clipped_counts += requested != bounded
                max_projection = np.maximum(max_projection, np.abs(requested - bounded))
                trace.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "time_s": env.time,
                            "learned_action": action.tolist(),
                            "action_representation": representation,
                            "absolute_request": requested.tolist(),
                            "bounded_request": bounded.tolist(),
                        }
                    )
                    + "\n"
                )
                # This is the same finite/hard-limit/rate-bounded interface as collection.
                env.set_joint_targets(dict(zip(action_names, bounded.tolist(), strict=True)))
                env.step(10)
                truth = env.task_truth()
                if truth["robot_contact_force_n"] > 1e-6:
                    max_contact_lift = max(
                        max_contact_lift, truth["block_pose_w"][2] - initial_height
                    )
                trace.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "time_s": env.time,
                            "applied_target": env.target[0, action_ids].cpu().tolist(),
                            "robot": env.robot_diagnostics(),
                            "truth": truth,
                        }
                    )
                    + "\n"
                )
                if tick % 100 == 0:
                    trace.flush()
                    print(json.dumps({"tick": tick, "time_s": env.time}), flush=True)
        dwell = []
        for _ in range(10):
            env.step(40)
            dwell.append(env.task_truth())
        env.capture()
        Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "final.png")
        report.update(
            release_dwell=dwell,
            inference_seconds=inference_seconds,
            clipped_request_ticks=clipped_ticks,
            clipped_counts=dict(zip(action_names, clipped_counts.tolist(), strict=True)),
            max_projection=dict(zip(action_names, max_projection.tolist(), strict=True)),
            max_contact_lift_m=max_contact_lift,
            executed_ticks=args.ticks,
            action_representation=representation,
            passed=max_contact_lift > 0.07
            and all(
                abs(sample["block_pose_w"][0] - 0.48) < 0.01
                and abs(sample["block_pose_w"][1] - 0.23) < 0.01
                and abs(sample["block_pose_w"][2] - 0.82) < 0.003
                and sample["robot_contact_force_n"] < 1e-6
                and sum(v * v for v in sample["block_velocity_w"][:3]) < 0.01**2
                and sum(v * v for v in sample["block_velocity_w"][3:]) < 0.1**2
                for sample in dwell
            ),
        )
    except Exception:
        report.update(error=traceback.format_exc())
        raise
    finally:
        try:
            send_message(connection, {"close": True})
        except OSError:
            pass
        connection.close()
        report["wall_seconds_before_app_close"] = time.monotonic() - started
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k
                    not in (
                        "release_dwell",
                        "inference_seconds",
                        "clipped_counts",
                        "max_projection",
                    )
                }
            ),
            flush=True,
        )
        if app is not None:
            app.close(exit_code=0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
