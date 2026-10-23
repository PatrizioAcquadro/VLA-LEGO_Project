"""Collect approved eight-object missing-piece examples with native safe holds."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from pathlib import Path

from vla_lego.isaac.pilot import load_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--event-settling", action="store_true")
    args = parser.parse_args()
    case = next(c for c in load_cases(args.cases) if c.name == args.case)
    if case.category != "stop" or (case.row["family"], case.row["split"]) != (
        args.family,
        args.split,
    ):
        parser.error("Case/category/family/split mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {
        "pilot_case": case.row,
        "collection_condition": "marker_free_continuous_site_v1",
        "passed": False,
        "verified_stop": False,
        "kind": "approved_pilot_stop",
        "learned": False,
        "qualification": False,
    }
    app = recorder = None
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        from PIL import Image

        from vla_lego.isaac.environment import TABLE_TOP, AssemblyEnvironment
        from vla_lego.isaac.recording import EpisodeRecorder
        from vla_lego.isaac.reference import PhysicalReference

        task = case.task()
        env = AssemblyEnvironment(
            args.cache_dir,
            placements=task.placements,
            assembly_xy_m=None if task.build_region_xy_m is not None else task.assembly_xy_m,
            appearance=case.appearance,
        )
        env.step(400)
        recorder = EpisodeRecorder(
            args.output_dir / "recording",
            task.instruction,
            family=args.family,
            split=args.split,
            task_decisions=True,
            collection_case=case.row,
            collection_condition="marker_free_continuous_site_v1",
        )
        env.recorder = recorder
        with (args.output_dir / "trace.jsonl").open("w") as trace:
            reference = PhysicalReference(env, trace, args.output_dir, event_settling=True)
            for side in ("left", "right"):
                reference.park(side)
            env.capture()
            Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "initial.png")
            reference.wait(4, "inspect_visible_inventory")
            safe = True
            recorder.task_decision = "abstain"
            for _ in range(4):
                reference.wait(1, "justified_abstention")
                safe &= all(
                    b["robot_contact_force_n"] < 1e-6
                    and b["nonrobot_contact_force_n"] > 1e-6
                    and abs(b["block_pose_w"][2] - TABLE_TOP - p.spec.dimensions_m[2] / 2) < 0.003
                    for b, p in zip(
                        (env.task_truth(i) for i in range(8)), task.placements, strict=True
                    )
                )
        env.capture()
        Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "final.png")
        env.stage.Export(str(args.output_dir / "scene.usda"))
        report.update(
            task=asdict(task),
            safe_released_hold=safe,
            passed=safe and not task.requested_pieces_available,
            verified_stop=safe and not task.requested_pieces_available,
            tower_success=False,
            recorded_frames=recorder.count,
        )
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if recorder is not None:
            recorder.finish(
                verified_success=False,
                verified_stop=bool(report["verified_stop"]),
                reason=report.get("error", "visible missing requested piece; safe native hold"),
            )
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if app is not None:
            app.close(exit_code=0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
