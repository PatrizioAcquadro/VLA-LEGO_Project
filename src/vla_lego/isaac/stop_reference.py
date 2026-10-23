"""Record justified abstention with a genuinely missing requested block and native hold."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--missing-target", type=int, choices=range(5), default=4)
    parser.add_argument("--mirror", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "physical_task_stop_reference",
        "passed": False,
        "verified_stop": False,
        "learned": False,
        "qualification": False,
    }
    app = recorder = None
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        from PIL import Image

        from vla_lego.isaac.environment import (
            ARM_JOINTS,
            JAW_JOINTS,
            TABLE_TOP,
            AssemblyEnvironment,
        )
        from vla_lego.isaac.recording import EpisodeRecorder
        from vla_lego.isaac.task import initial_tower_task

        task = initial_tower_task(mirror=args.mirror).without_target(args.missing_target)
        env = AssemblyEnvironment(
            args.cache_dir.resolve(), placements=task.placements, assembly_xy_m=task.assembly_xy_m
        )
        env.step(400)
        recorder = EpisodeRecorder(
            args.output_dir / "recording",
            task.instruction,
            family=args.family,
            split=args.split,
            task_decisions=True,
        )
        recorder.task_decision = "abstain"
        env.recorder = recorder
        names = (*ARM_JOINTS, *JAW_JOINTS)
        ids = [env.robot.data.joint_names.index(n) for n in names]
        limits = env.robot.data.joint_pos_limits.torch[0, ids]
        hold = env.robot.data.joint_pos.torch[0, ids].clamp(limits[:, 0], limits[:, 1])
        safe_hold = True
        with (args.output_dir / "trace.jsonl").open("w") as stream:
            for _ in range(20):
                env.set_joint_targets(dict(zip(names, hold.tolist(), strict=True)))
                env.step(10)
                truth = [env.task_truth(i) for i in range(len(env.blocks))]
                safe_hold = safe_hold and all(
                    b["robot_contact_force_n"] < 1e-6
                    and b["nonrobot_contact_force_n"] > 1e-6
                    and abs(b["block_pose_w"][2] - TABLE_TOP - p.spec.dimensions_m[2] / 2) < 0.003
                    for b, p in zip(truth, task.placements, strict=True)
                )
                stream.write(
                    json.dumps(
                        {"time_s": env.time, "blocks": truth, "robot": env.robot_diagnostics()}
                    )
                    + "\n"
                )
        env.capture()
        Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "final.png")
        report.update(
            task=asdict(task),
            missing_target_level=args.missing_target,
            requested_piece_missing=not task.requested_pieces_available,
            safe_released_hold=safe_hold,
            verified_stop=not task.requested_pieces_available and safe_hold,
            tower_success=False,
            recorded_frames=recorder.count,
        )
    finally:
        if recorder is not None:
            recorder.finish(
                verified_success=False,
                verified_stop=bool(report["verified_stop"]),
                reason="requested piece absent; held native joint targets",
            )
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if app is not None:
            app.close(exit_code=0 if report["verified_stop"] else 1)


if __name__ == "__main__":
    main()
