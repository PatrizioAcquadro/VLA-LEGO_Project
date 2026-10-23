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
    args = parser.parse_args()
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
        max_state_error = 0.0
        count = 0
        with (args.recording / "frames.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if abs(row["observation_time_s"] - env.time) > 1e-6:
                    raise ValueError("Replay control clock diverged from the recording.")
                env.set_joint_targets(
                    dict(zip(metadata["action_names"], row["action"], strict=True))
                )
                env.capture()
                state = env.observe()["joint_position"]
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
            final_truth=final,
            privileged_inputs_used_for_commands=False,
        )
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
