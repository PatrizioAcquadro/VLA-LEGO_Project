#!/usr/bin/env python3
"""Validate multi-view camera contract and synchronization (Phase 1.1.7).

Run:
    python scripts/validate_cameras.py

Artifacts are saved to ``logs/camera_validation/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

# Ensure project root is importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from sim.action_space import ACTION_DIM, ARM_DIM, GRIPPER_DIM  # noqa: E402
from sim.asset_loader import load_scene  # noqa: E402
from sim.camera import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_CAPTURE_HZ,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    NUM_VIEWS,
    MultiViewRenderer,
)
from sim.offscreen import RenderedFrame, resolve_camera_id, save_video  # noqa: E402
from sim.sim_runner import SimRunner  # noqa: E402

ARTIFACT_DIR = _project_root / "logs" / "camera_validation"


def print_contract() -> None:
    """Print the frozen camera contract."""
    print("=" * 70)
    print("MULTI-VIEW CAMERA CONTRACT (Phase 1.1.7)")
    print("=" * 70)
    print(f"  Views:           {list(CAMERA_NAMES)}")
    print(f"  Num views:       {NUM_VIEWS}")
    print(f"  Resolution:      {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
    print(f"  Capture rate:    {DEFAULT_CAPTURE_HZ} Hz")
    print()


def test_render_at_rest(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Render both views at rest and verify non-black RGB."""
    print("-" * 70)
    print("Render at rest (home keyframe)")
    print("-" * 70)

    passed = True
    with MultiViewRenderer(model) as mv:
        frame = mv.capture(data, step_index=0)
        for name in CAMERA_NAMES:
            rgb = frame.views[name].rgb
            shape_ok = rgb.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
            not_black = bool(np.any(rgb > 0))
            mean_val = float(np.mean(rgb))
            print(
                f"  {name:15s}  shape={rgb.shape}  "
                f"not_black={'OK' if not_black else 'FAIL'}  "
                f"mean_px={mean_val:.1f}"
            )
            if not shape_ok or not not_black:
                passed = False

    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_sync(model: mujoco.MjModel) -> bool:
    """Verify both views share the same step_index."""
    print("-" * 70)
    print("Synchronization check")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    with MultiViewRenderer(model) as mv:
        frame = mv.capture(data, step_index=5)
        indices = {name: frame.views[name].step_index for name in CAMERA_NAMES}
        all_same = len(set(indices.values())) == 1
        print(f"  Step indices:   {indices}")
        print(f"  Frame index:    {frame.step_index}")
        print(f"  Synchronized:   {'OK' if all_same else 'FAIL'}")

    passed = all_same and frame.step_index == 5
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_robot_cam_movement(model: mujoco.MjModel) -> bool:
    """Verify robot_cam moves with spine_z while third_person is static."""
    print("-" * 70)
    print("Robot camera movement test")
    print("-" * 70)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    robot_cam_id = resolve_camera_id(model, "robot_cam")
    tp_cam_id = resolve_camera_id(model, "third_person")

    rest_robot_pos = data.cam_xpos[robot_cam_id].copy()
    rest_tp_pos = data.cam_xpos[tp_cam_id].copy()

    # Rotate spine_z
    spine_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spine_z")
    qpos_adr = model.jnt_qposadr[spine_id]
    data.qpos[qpos_adr] = 0.4
    mujoco.mj_forward(model, data)

    moved_robot_pos = data.cam_xpos[robot_cam_id].copy()
    moved_tp_pos = data.cam_xpos[tp_cam_id].copy()

    robot_delta = float(np.linalg.norm(moved_robot_pos - rest_robot_pos))
    tp_delta = float(np.linalg.norm(moved_tp_pos - rest_tp_pos))

    robot_moved = robot_delta > 0.01
    tp_static = tp_delta < 1e-10

    print(f"  robot_cam delta:      {robot_delta:.6f} m ({'OK' if robot_moved else 'FAIL'})")
    print(f"  third_person delta:   {tp_delta:.2e} m ({'OK' if tp_static else 'FAIL'})")
    print(f"  robot_cam rest pos:   {rest_robot_pos}")
    print(f"  robot_cam moved pos:  {moved_robot_pos}")

    passed = robot_moved and tp_static
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def test_metadata(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Verify camera metadata extraction."""
    print("-" * 70)
    print("Camera metadata")
    print("-" * 70)

    passed = True
    with MultiViewRenderer(model) as mv:
        meta = mv.get_metadata(data)
        for name in CAMERA_NAMES:
            m = meta[name]
            fovy_ok = m.fovy > 0
            pos_ok = bool(np.all(np.isfinite(m.pos)))
            mat_ok = bool(np.all(np.isfinite(m.mat)))
            print(f"  {name}:")
            print(f"    fovy:  {m.fovy:.1f} deg")
            print(f"    pos:   {m.pos}")
            print(f"    mat:   {m.mat.flatten()[:6]}...")
            if not (fovy_ok and pos_ok and mat_ok):
                passed = False

    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def save_sample_frames(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Save sample frames from both cameras at rest."""
    import imageio.v3 as iio

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with MultiViewRenderer(model, width=640, height=480) as mv:
        frame = mv.capture(data, step_index=0)
        for name in CAMERA_NAMES:
            path = ARTIFACT_DIR / f"{name}_sample.png"
            iio.imwrite(str(path), frame.views[name].rgb)
            print(f"  Saved: {path}")


def save_dual_view_video(model: mujoco.MjModel, n_steps: int = 100, seed: int = 42) -> None:
    """Save side-by-side dual-view video with random actions."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    runner = SimRunner(model, data)
    rng = np.random.default_rng(seed)

    # Collect side-by-side frames
    combined_frames: list[RenderedFrame] = []
    with MultiViewRenderer(model, width=320, height=240) as mv:
        for step in range(n_steps):
            frame = mv.capture(data, step_index=step)

            # Concatenate horizontally: [robot_cam | third_person]
            left = frame.views["robot_cam"].rgb
            right = frame.views["third_person"].rgb
            combined = np.concatenate([left, right], axis=1)
            combined_frames.append(RenderedFrame(step_index=step, rgb=combined))

            # Step with random action
            action = rng.uniform(-0.3, 0.3, size=ACTION_DIM)
            action[ARM_DIM:] = rng.uniform(0, 1, size=GRIPPER_DIM)
            runner.step(action)

    video_path = save_video(combined_frames, ARTIFACT_DIR / "dual_view_video.mp4", fps=20.0)
    print(f"  Saved: {video_path}")


def save_metadata_json(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Save camera metadata to JSON."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with MultiViewRenderer(model) as mv:
        meta = mv.get_metadata(data)
        meta_dict = {name: m.to_dict() for name, m in meta.items()}
        meta_dict["_contract"] = {
            "num_views": NUM_VIEWS,
            "camera_names": list(CAMERA_NAMES),
            "default_width": DEFAULT_WIDTH,
            "default_height": DEFAULT_HEIGHT,
            "capture_hz": DEFAULT_CAPTURE_HZ,
        }

    json_path = ARTIFACT_DIR / "camera_metadata.json"
    with open(json_path, "w") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"  Saved: {json_path}")


def main() -> None:
    print("Loading Alex upper-body model...")
    model = load_scene("alex_upper_body")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print_contract()

    results: dict[str, bool] = {}
    results["render_at_rest"] = test_render_at_rest(model, data)
    results["sync"] = test_sync(model)
    results["robot_cam_movement"] = test_robot_cam_movement(model)
    results["metadata"] = test_metadata(model, data)

    # Artifacts
    print("-" * 70)
    print("Exporting artifacts")
    print("-" * 70)
    save_sample_frames(model, data)
    save_dual_view_video(model)
    save_metadata_json(model, data)
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} {status}")
    print()

    if all_pass:
        print("All validation checks PASSED.")
        print(f"Artifacts saved to: {ARTIFACT_DIR}")
    else:
        print("Some checks FAILED -- see details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
