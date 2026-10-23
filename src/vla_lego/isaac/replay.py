"""Replay recorded applied joint commands in native physics, without a teacher."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--capture-every", type=int, default=1)
    parser.add_argument("--compare-images", action="store_true")
    parser.add_argument(
        "--action-representation", choices=("absolute", "measured_arm_offset"), default="absolute"
    )
    args = parser.parse_args()
    if args.capture_every < 1 or 10 % args.capture_every:
        parser.error("--capture-every must be a positive divisor of 10 decision ticks.")
    metadata = json.loads((args.recording / "episode.json").read_text())
    if not metadata.get("control_clock_enforced"):
        raise ValueError("Replay requires a verified causal recording.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    app = None
    started = time.monotonic()
    report: dict = {"kind": "native_joint_replay", "learned": False, "passed": False}
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        import numpy as np
        from PIL import Image

        from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS, AssemblyEnvironment

        env = AssemblyEnvironment(args.cache_dir.resolve())
        if metadata["action_names"] != [*ARM_JOINTS, *JAW_JOINTS]:
            raise ValueError("Recorded commands use a different joint interface.")
        env.step(round(metadata["start_time_s"] / 0.005))
        arm_ids = [env.robot.data.joint_names.index(name) for name in ARM_JOINTS]
        action_ids = [env.robot.data.joint_names.index(name) for name in metadata["action_names"]]
        limits = env.robot.data.joint_pos_limits.torch[0, action_ids].cpu().numpy()
        max_state_error = 0.0
        max_command_error = 0.0
        count = 0
        image_metrics: list[dict] = []
        with (args.recording / "frames.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if abs(row["observation_time_s"] - env.time) > 1e-6:
                    raise ValueError("Replay control clock diverged from the recording.")
                action = np.asarray(row["action"], dtype=np.float32)
                if args.action_representation == "measured_arm_offset":
                    offset = action[:14] - np.asarray(row["joint_position"][:14], dtype=np.float32)
                    action[:14] = env.robot.data.joint_pos.torch[0, arm_ids].cpu().numpy() + offset
                action = np.clip(action, limits[:, 0], limits[:, 1])
                max_command_error = max(
                    max_command_error, float(np.max(np.abs(action - np.asarray(row["action"]))))
                )
                env.set_joint_targets(
                    dict(zip(metadata["action_names"], action.tolist(), strict=True))
                )
                if count % args.capture_every == 0:
                    env.capture()
                observation = env.observe()
                state = observation["joint_position"]
                if args.compare_images and count % 10 == 0:
                    image_path = (args.recording / row["head_rgb"]).resolve()
                    if not image_path.is_relative_to(args.recording.resolve()):
                        raise ValueError("Recorded image escapes its episode directory.")
                    with Image.open(image_path) as original:
                        source_rgb = np.asarray(original.convert("RGB"))
                    actual_rgb = observation["head_rgb"]
                    if source_rgb.shape != actual_rgb.shape:
                        raise ValueError("Replay image dimensions differ from the source.")
                    difference = np.abs(actual_rgb.astype(np.int16) - source_rgb.astype(np.int16))
                    image_metrics.append(
                        {
                            "frame": count,
                            "mean_absolute_channel_error": float(difference.mean()),
                            "p99_absolute_channel_error": float(np.quantile(difference, 0.99)),
                            "max_absolute_channel_error": int(difference.max()),
                            "fraction_channels_error_over_10": float((difference > 10).mean()),
                        }
                    )
                max_state_error = max(
                    max_state_error,
                    float(np.max(np.abs(state - np.asarray(row["joint_position"])))),
                )
                env.step(10)
                count += 1
        env.capture()
        Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "final.png")
        final = env.task_truth()
        # Fixed component replay gates, distinct from full-task qualification.
        passed = (
            count == metadata["frames"]
            and max_state_error < 0.005
            and abs(final["block_pose_w"][0] - 0.48) < 0.01
            and abs(final["block_pose_w"][1] - 0.23) < 0.01
            and abs(final["block_pose_w"][2] - 0.82) < 0.003
            and final["robot_contact_force_n"] < 1e-6
            and sum(v * v for v in final["block_velocity_w"][:3]) < 0.01**2
            and sum(v * v for v in final["block_velocity_w"][3:]) < 0.1**2
        )
        report.update(
            passed=passed,
            recording=str(args.recording),
            frames=count,
            max_joint_state_error=max_state_error,
            max_decoded_command_error=max_command_error,
            final_truth=final,
            privileged_inputs_used_for_commands=False,
            action_representation=args.action_representation,
            capture_every_control_ticks=args.capture_every,
        )
        if args.compare_images:
            (args.output_dir / "image_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in image_metrics)
            )
            report["decision_image_comparison"] = {
                "frames": len(image_metrics),
                "mean_absolute_channel_error": float(
                    np.mean([row["mean_absolute_channel_error"] for row in image_metrics])
                ),
                "max_frame_mean_absolute_channel_error": max(
                    row["mean_absolute_channel_error"] for row in image_metrics
                ),
            }
    except Exception:
        report.update(error=traceback.format_exc())
        raise
    finally:
        report["wall_seconds_before_app_close"] = time.monotonic() - started
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if app is not None:
            app.close(exit_code=0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
