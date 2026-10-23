"""Reconstruct an actual autonomous prefix using only its applied robot commands."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any


def replay_prefix(env: Any, source: Path, ticks: int) -> dict:
    """Require exact native state parity before interpreting a failed-policy state."""
    import numpy as np

    from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS

    metadata = json.loads((source / "report.json").read_text())
    if (
        not metadata["learned"]
        or metadata["teacher_commands"]
        or not metadata.get("autonomy_started")
        or metadata.get("robot_joint_names") != env.robot.data.joint_names
    ):
        raise ValueError("Replay needs a named, autonomous native source with no teacher.")
    if ticks < 1 or ticks > metadata["executed_ticks"]:
        raise ValueError("Prefix must fit the actually executed source attempt.")
    capture_hz = metadata.get("head_capture_hz", 2)
    if capture_hz not in (2, 20):
        raise ValueError("Unknown source capture cadence.")
    capture_every = 20 // capture_hz
    names = (*ARM_JOINTS, *JAW_JOINTS)
    ids = [env.robot.data.joint_names.index(name) for name in names]
    limits = env.robot.data.joint_pos_limits.torch[0, ids].cpu().numpy()
    state_error = pose_error = 0.0
    count = 0
    pending = None
    with (source / "trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if "learned_action" in row:
                if pending is not None or row["tick"] != count:
                    raise ValueError("Autonomous source command/state ordering is invalid.")
                pending = row
                continue
            if "applied_target" not in row:
                raise ValueError("Unrecognized autonomous source row.")
            if pending is None or row["tick"] != pending["tick"]:
                raise ValueError("Autonomous state has no matching source command.")
            if abs(pending["time_s"] - env.time) > 1e-8:
                raise ValueError("Replay clock differs from the autonomous source.")
            if count % capture_every == 0:
                env.capture()
            action = np.asarray(row["applied_target"], dtype=np.float32)
            if action.shape != (16,) or not np.isfinite(action).all():
                raise ValueError("Invalid recorded robot command.")
            # Numerical limit projection matches the actual native source interface.
            action = np.clip(action, limits[:, 0], limits[:, 1])
            env.set_joint_targets(dict(zip(names, action.tolist(), strict=True)))
            env.step(10)
            if abs(row["time_s"] - env.time) > 1e-8:
                raise ValueError("Post-command replay clock differs from the source.")
            actual = env.robot.data.joint_pos.torch[0].cpu().numpy()
            state_error = max(
                state_error,
                float(np.max(np.abs(actual - np.asarray(row["robot"]["joint_position"])))),
            )
            pose_error = max(
                pose_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(env.task_truth()["block_pose_w"])
                            - np.asarray(row["truth"]["block_pose_w"])
                        )
                    )
                ),
            )
            count += 1
            pending = None
            if count == ticks:
                break
    if count != ticks:
        raise ValueError("Source trace ended before the requested prefix.")
    result = {
        "source_run": str(source.resolve()),
        "prefix_ticks": count,
        "source_head_capture_hz": capture_hz,
        "max_joint_state_error": state_error,
        "max_block_pose_component_error": pose_error,
        "exact_native_parity": state_error == 0.0 and pose_error == 0.0,
        "object_state_writes": False,
        "teacher_commands_in_prefix": False,
        "autonomous_success_claim": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--prefix-ticks", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {"kind": "native_policy_prefix_replay", "passed": False}
    app = None
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        from PIL import Image

        from vla_lego.isaac.environment import AssemblyEnvironment

        env = AssemblyEnvironment(args.cache_dir.resolve())
        env.step(400)
        result = replay_prefix(env, args.source_run, args.prefix_ticks)
        env.capture()
        Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "prefix-final.png")
        report.update(result, passed=result["exact_native_parity"], final_truth=env.task_truth())
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds_before_app_close"] = time.monotonic() - started
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if app is not None:
            app.close(exit_code=0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
