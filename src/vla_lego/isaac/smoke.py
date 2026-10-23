"""Bounded native foundation probe; this does not score assembly success."""

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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report: dict = {"kind": "native_foundation_probe", "assembly_success": None}
    app = None
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            headless=True,
            device="cuda:0",
            enable_cameras=True,
            width=960,
            height=600,
        ).app
        import carb
        import numpy as np
        import torch
        from PIL import Image

        from vla_lego.isaac.environment import (
            BLOCK_SIZE,
            JAW_JOINTS,
            TABLE_TOP,
            AssemblyEnvironment,
        )

        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", "RayTracedLighting")
        settings.set("/rtx/post/tonemap/op", 4)
        env = AssemblyEnvironment(args.cache_dir.resolve())
        root_initial = env.robot.data.root_pos_w.torch.clone()
        env.step(600)
        open_actual = env.robot.data.joint_pos.torch[0, env.jaw_ids].tolist()
        env.set_joint_targets(dict.fromkeys(JAW_JOINTS, 0.0))
        env.step(200)
        closed_actual = env.robot.data.joint_pos.torch[0, env.jaw_ids].tolist()
        env.set_joint_targets(dict.fromkeys(JAW_JOINTS, 0.034))
        env.step(200)
        for _ in range(8):
            env.step(1, render=True)
        observation = env.observe()
        rgb = observation.pop("head_rgb")
        Image.fromarray(rgb).save(args.output_dir / "head.png")
        np.savez_compressed(args.output_dir / "observation.npz", **observation)
        truth = env.task_truth()
        root_drift = float((env.robot.data.root_pos_w.torch - root_initial).abs().max())
        checks = {
            "cuda": env.robot.data.joint_pos.torch.is_cuda,
            "finite_joints": bool(torch.isfinite(env.robot.data.joint_pos.torch).all()),
            "fixed_root": root_drift < 1e-6,
            "gravity_table_support": abs(truth["block_pose_w"][2] - (TABLE_TOP + BLOCK_SIZE / 2))
            < 0.003,
            "block_settled": float(np.linalg.norm(truth["block_velocity_w"][:3])) < 0.02,
            "jaw_open": all(abs(value - 0.034) < 0.002 for value in open_actual),
            "jaw_closed": all(abs(value) < 0.002 for value in closed_actual),
            "nonblank_rgb": rgb.shape == (600, 960, 3) and float(rgb.std()) > 5,
            "red_block_visible": int(
                ((rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] < 80)).sum()
            )
            > 20,
        }
        env.stage.GetRootLayer().Export(str(args.output_dir.resolve() / "scene.usda"))
        report.update(
            checks=checks,
            passed=all(checks.values()),
            device=torch.cuda.get_device_name(),
            joint_names=list(env.robot.data.joint_names),
            body_names=list(env.robot.data.body_names),
            joint_limits=env.robot.data.joint_pos_limits.torch[0].tolist(),
            joint_position=env.robot.data.joint_pos.torch[0].tolist(),
            body_position_w=env.robot.data.body_pos_w.torch[0].tolist(),
            body_quaternion_w=env.robot.data.body_quat_w.torch[0].tolist(),
            camera_position_w=env.camera.data.pos_w.torch[0].tolist(),
            camera_quaternion_world_xyzw=env.camera.data.quat_w_world.torch[0].tolist(),
            open_actual_m=open_actual,
            closed_actual_m=closed_actual,
            task_truth=truth,
            root_drift_m=root_drift,
            rgb_mean=float(rgb.mean()),
            rgb_std=float(rgb.std()),
            simulated_seconds=env.time,
            assumptions={
                "table_top_m": TABLE_TOP,
                "block_size_m": BLOCK_SIZE,
                "block_mass_kg": 0.025,
                "block_static_dynamic_friction": [0.6, 0.5],
                "physical_calibration": "unmeasured",
            },
        )
    except Exception:
        report.update(passed=False, error=traceback.format_exc())
        raise
    finally:
        report["wall_seconds_before_app_close"] = time.monotonic() - start
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if app is not None:
            app.close()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
