"""Verify a loose five-target/two-distractor scene, never a prearranged tower."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {"kind": "loose_scene_probe", "assembly_success": None}
    app = None
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        from PIL import Image
        from pxr import UsdPhysics

        from vla_lego.isaac.environment import TABLE_TOP, AssemblyEnvironment
        from vla_lego.isaac.task import initial_tower_task

        task = initial_tower_task()
        env = AssemblyEnvironment(
            args.cache_dir.resolve(), placements=task.placements, assembly_xy_m=task.assembly_xy_m
        )
        env.step(600)
        for _ in range(8):
            env.step(1, render=True)
        Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "head.png")
        truth = [env.task_truth(i) for i in range(len(env.blocks))]
        checks = {
            "requested_pieces_available": task.requested_pieces_available,
            "marker_visual_only": all(
                not prim.HasAPI(UsdPhysics.CollisionAPI)
                and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
                for prim in env.stage.Traverse()
                if prim.GetName().startswith("Marker_")
            ),
            "assembly_area_empty": all(
                abs(placement.xy_m[0] - task.assembly_xy_m[0])
                > 0.06 + placement.spec.dimensions_m[0] / 2
                or abs(placement.xy_m[1] - task.assembly_xy_m[1])
                > 0.06 + placement.spec.dimensions_m[1] / 2
                for placement in task.placements
            ),
            "all_loose_supported": all(
                abs(state["block_pose_w"][2] - (TABLE_TOP + placement.spec.dimensions_m[2] / 2))
                < 0.003
                for state, placement in zip(truth, task.placements, strict=True)
            ),
            "all_released": all(state["robot_contact_force_n"] < 1e-6 for state in truth),
            "no_initial_stack": all(
                all(sum(v * v for v in force) < 1e-12 for force in state["block_contact_forces_n"])
                for state in truth
            ),
        }
        report.update(
            passed=all(checks.values()),
            checks=checks,
            task=asdict(task),
            instruction=task.instruction,
            truth=truth,
            state_timestamp_s=env.observe()["timestamp_s"],
            image_timestamp_s=env.observe()["image_timestamp_s"],
        )
        env.stage.GetRootLayer().Export(str(args.output_dir.resolve() / "scene.usda"))
    except Exception:
        report.update(passed=False, error=traceback.format_exc())
        raise
    finally:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if app is not None:
            app.close(exit_code=0 if report.get("passed") else 1)


if __name__ == "__main__":
    main()
