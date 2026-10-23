"""Short robot-motion and prearranged-brick physics demos; no learned policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np

from vla_lego.sim.asset_loader import load_scene
from vla_lego.sim.bricks.episode import EpisodeManager
from vla_lego.sim.bricks.task import evaluate_goal, make_goal
from vla_lego.sim.offscreen import RenderConfig, create_renderer, render_frame, resolve_camera_id
from vla_lego.sim.sim_runner import SimRunner


def _record(model, data, path, camera, advance, frames=60):
    config = RenderConfig(width=640, height=480, camera_name=camera)
    camera_id = resolve_camera_id(model, camera)
    with create_renderer(model, config) as renderer, iio.get_writer(path, fps=20) as video:
        for index in range(frames):
            frame = render_frame(renderer, model, data, index, config, camera_id)
            if frame.rgb.std() < 1:
                raise RuntimeError("Rendered frame is empty or uniform")
            video.append_data(frame.rgb)
            if index in (0, frames - 1):
                iio.imwrite(path.with_name(f"{path.stem}_{index:03d}.png"), frame.rgb)
            advance(index)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("robot", "bricks", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"kind": "simulation_diagnostic", "seed": args.seed, "checks": {}}
    try:
        if args.scenario in ("robot", "all"):
            model = load_scene("alex_upper_body")
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key("rest").id)
            runner = SimRunner(model, data)

            def advance(index):
                action = np.zeros(17)
                action[0] = 0.05 * np.sin(2 * np.pi * index / 60)
                action[-2:] = 0.5 + 0.5 * np.sin(2 * np.pi * index / 60)
                runner.step(action)
                if not np.isfinite(data.qpos).all():
                    raise RuntimeError("Non-finite robot state")

            _record(model, data, args.output_dir / "robot.mp4", "third_person", advance)
            report["checks"]["robot_motion"] = "finite 17-D scripted commands; no task assembly"
        if args.scenario in ("bricks", "all"):
            episode = EpisodeManager()
            for shape in ("tower", "triangle"):
                goal = make_goal(episode.bricks, shape)
                episode.reset(seed=args.seed, layout=goal)
                _record(
                    episode.model,
                    episode.data,
                    args.output_dir / f"{shape}.mp4",
                    "workspace_closeup",
                    lambda _: episode.step(0.05),
                )
                result = evaluate_goal(episode, goal)
                report["checks"][shape] = asdict(result) | {"success": result.success}
                if not result.success:
                    raise RuntimeError(f"Prearranged {shape} did not remain stable")
        report["success"] = True
    except Exception as error:
        report.update(success=False, error=str(error))
        raise
    finally:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Diagnostic artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
