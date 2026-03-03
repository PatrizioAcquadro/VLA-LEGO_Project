"""Tests for multi-view camera contract (Phase 1.1.7).

Run:
    pytest tests/test_cameras.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.camera import (  # noqa: E402
    CAMERA_NAMES,
    DEFAULT_CAPTURE_HZ,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    NUM_VIEWS,
    CameraMetadata,
    MultiViewFrame,
    MultiViewRenderer,
    get_camera_metadata,
)
from sim.offscreen import resolve_camera_id  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def alex_model():
    """Load the Alex upper-body scene."""
    from sim.asset_loader import load_scene

    return load_scene("alex_upper_body")


@pytest.fixture(scope="module")
def alex_data(alex_model):
    """Create MjData at rest (shared, read-only)."""
    data = mujoco.MjData(alex_model)
    mujoco.mj_resetData(alex_model, data)
    mujoco.mj_forward(alex_model, data)
    return data


@pytest.fixture()
def fresh_data(alex_model):
    """Fresh MjData for tests that mutate state."""
    data = mujoco.MjData(alex_model)
    mujoco.mj_resetData(alex_model, data)
    mujoco.mj_forward(alex_model, data)
    return data


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestCameraContract:
    """Verify frozen camera constants and MJCF definitions."""

    def test_num_views_is_2(self) -> None:
        assert NUM_VIEWS == 2

    def test_camera_names_frozen(self) -> None:
        assert CAMERA_NAMES == ("robot_cam", "third_person")

    def test_default_resolution(self) -> None:
        assert DEFAULT_WIDTH == 320
        assert DEFAULT_HEIGHT == 240

    def test_default_capture_hz(self) -> None:
        assert DEFAULT_CAPTURE_HZ == 20.0

    def test_cameras_exist_in_model(self, alex_model) -> None:
        for name in CAMERA_NAMES:
            cam_id = resolve_camera_id(alex_model, name)
            assert cam_id >= 0, f"Camera '{name}' not found in model"


# ---------------------------------------------------------------------------
# Multi-view capture tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestMultiViewCapture:
    """Test synchronized multi-view rendering."""

    def test_capture_returns_both_views(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            assert isinstance(frame, MultiViewFrame)
            assert len(frame.views) == NUM_VIEWS
            for name in CAMERA_NAMES:
                assert name in frame.views

    def test_rgb_shapes(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            for name in CAMERA_NAMES:
                rgb = frame.views[name].rgb
                assert rgb.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
                assert rgb.dtype == np.uint8

    def test_rgb_not_black(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            for name in CAMERA_NAMES:
                assert np.any(frame.views[name].rgb > 0), f"{name} rendered all-black"

    def test_step_index_matches(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=42)
            assert frame.step_index == 42
            for name in CAMERA_NAMES:
                assert frame.views[name].step_index == 42

    def test_timestamp_populated(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            assert frame.timestamp >= 0.0

    def test_custom_resolution(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model, width=160, height=120) as mv:
            frame = mv.capture(alex_data, step_index=0)
            for name in CAMERA_NAMES:
                assert frame.views[name].rgb.shape == (120, 160, 3)

    def test_depth_rendering(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model, render_depth=True) as mv:
            frame = mv.capture(alex_data, step_index=0)
            for name in CAMERA_NAMES:
                assert frame.views[name].depth is not None
                assert frame.views[name].depth.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH)


# ---------------------------------------------------------------------------
# Synchronization tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestCameraSync:
    """Verify multi-view synchronization contract."""

    def test_views_share_step_index(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=7)
            indices = {frame.views[n].step_index for n in CAMERA_NAMES}
            assert len(indices) == 1, f"Step indices differ across views: {indices}"

    def test_successive_captures_differ(self, alex_model, fresh_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame1 = mv.capture(fresh_data, step_index=0)
            mujoco.mj_step(alex_model, fresh_data)
            frame2 = mv.capture(fresh_data, step_index=1)
            assert frame1.step_index != frame2.step_index


# ---------------------------------------------------------------------------
# Camera movement tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestRobotCamMovement:
    """Verify robot_cam tracks spine_z rotation while third_person is static."""

    def test_robot_cam_tracks_spine(self, alex_model, fresh_data) -> None:
        mujoco.mj_forward(alex_model, fresh_data)
        robot_cam_id = resolve_camera_id(alex_model, "robot_cam")
        rest_pos = fresh_data.cam_xpos[robot_cam_id].copy()

        # Rotate spine_z to near its limit
        spine_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "spine_z")
        qpos_adr = alex_model.jnt_qposadr[spine_id]
        fresh_data.qpos[qpos_adr] = 0.4  # ~23 degrees
        mujoco.mj_forward(alex_model, fresh_data)

        moved_pos = fresh_data.cam_xpos[robot_cam_id].copy()
        delta = np.linalg.norm(moved_pos - rest_pos)
        assert delta > 0.01, f"robot_cam did not move with spine (delta={delta:.6f})"

    def test_third_person_static(self, alex_model, fresh_data) -> None:
        mujoco.mj_forward(alex_model, fresh_data)
        tp_cam_id = resolve_camera_id(alex_model, "third_person")
        rest_pos = fresh_data.cam_xpos[tp_cam_id].copy()

        # Rotate spine
        spine_id = mujoco.mj_name2id(alex_model, mujoco.mjtObj.mjOBJ_JOINT, "spine_z")
        qpos_adr = alex_model.jnt_qposadr[spine_id]
        fresh_data.qpos[qpos_adr] = 0.4
        mujoco.mj_forward(alex_model, fresh_data)

        moved_pos = fresh_data.cam_xpos[tp_cam_id].copy()
        delta = np.linalg.norm(moved_pos - rest_pos)
        assert delta < 1e-10, f"third_person camera moved (delta={delta:.6f})"


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestCameraMetadata:
    """Test camera intrinsic/extrinsic metadata extraction."""

    def test_metadata_fovy_positive(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            meta = mv.get_metadata(alex_data)
            for name in CAMERA_NAMES:
                assert meta[name].fovy > 0, f"{name} fovy not positive"

    def test_robot_cam_fovy(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            meta = mv.get_metadata(alex_data)
            assert meta["robot_cam"].fovy == pytest.approx(60.0)

    def test_third_person_fovy(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            meta = mv.get_metadata(alex_data)
            assert meta["third_person"].fovy == pytest.approx(50.0)

    def test_metadata_pos_finite(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            meta = mv.get_metadata(alex_data)
            for name in CAMERA_NAMES:
                assert np.all(np.isfinite(meta[name].pos)), f"{name} pos not finite"
                assert np.all(np.isfinite(meta[name].mat)), f"{name} mat not finite"

    def test_metadata_resolution(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            meta = mv.get_metadata(alex_data)
            for name in CAMERA_NAMES:
                assert meta[name].width == DEFAULT_WIDTH
                assert meta[name].height == DEFAULT_HEIGHT

    def test_metadata_to_dict(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            meta = mv.get_metadata(alex_data)
            for name in CAMERA_NAMES:
                d = meta[name].to_dict()
                assert isinstance(d, dict)
                assert d["name"] == name
                assert isinstance(d["pos"], list)
                assert len(d["pos"]) == 3

    def test_get_camera_metadata_standalone(self, alex_model, alex_data) -> None:
        meta = get_camera_metadata(alex_model, alex_data, "robot_cam")
        assert isinstance(meta, CameraMetadata)
        assert meta.name == "robot_cam"
        assert meta.pos.shape == (3,)
        assert meta.mat.shape == (3, 3)


# ---------------------------------------------------------------------------
# Context manager / lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.mujoco
@pytest.mark.assets
class TestRendererLifecycle:
    """Test renderer creation, context manager, and cleanup."""

    def test_context_manager(self, alex_model, alex_data) -> None:
        with MultiViewRenderer(alex_model) as mv:
            frame = mv.capture(alex_data, step_index=0)
            assert len(frame.views) == NUM_VIEWS

    def test_closed_renderer_raises(self, alex_model, alex_data) -> None:
        mv = MultiViewRenderer(alex_model)
        mv.close()
        with pytest.raises(RuntimeError, match="closed"):
            mv.capture(alex_data, step_index=0)

    def test_camera_ids_property(self, alex_model) -> None:
        with MultiViewRenderer(alex_model) as mv:
            ids = mv.camera_ids
            assert len(ids) == NUM_VIEWS
            for name in CAMERA_NAMES:
                assert name in ids
                assert isinstance(ids[name], int)
                assert ids[name] >= 0
