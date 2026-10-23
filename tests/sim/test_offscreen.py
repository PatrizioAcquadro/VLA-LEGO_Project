"""Tests for sim.offscreen (Phase 0.2.3).

All tests run headlessly — no display required.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_lego.sim.asset_loader import ASSETS_DIR

mujoco = pytest.importorskip("mujoco")


TEST_SCENE = ASSETS_DIR / "scenes" / "test_scene.xml"
# Named camera defined in test_scene.xml for reliable offscreen rendering
TEST_CAMERA = "overhead"


@pytest.fixture
def model():
    """Load the test scene model."""
    from vla_lego.sim.mujoco_env import load_model

    return load_model(TEST_SCENE)


@pytest.fixture
def model_and_data(model):
    """Load test scene model and create fresh data."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    return model, data


@pytest.fixture
def render_config():
    """Default RenderConfig using the named test camera."""
    from vla_lego.sim.offscreen import RenderConfig

    return RenderConfig(camera_name=TEST_CAMERA)


@pytest.mark.mujoco
class TestOffscreenImport:
    """Test that the offscreen module imports cleanly."""

    def test_import_offscreen_module(self) -> None:
        """sim.offscreen imports without error."""
        import vla_lego.sim.offscreen  # noqa: F401

    def test_render_config_defaults(self) -> None:
        """RenderConfig has expected defaults."""
        from vla_lego.sim.offscreen import RenderConfig

        cfg = RenderConfig()
        assert cfg.width == 640
        assert cfg.height == 480
        assert cfg.camera_name is None
        assert cfg.camera_id == -1
        assert cfg.render_depth is False
        assert cfg.render_segmentation is False

    def test_rendered_frame_dataclass(self) -> None:
        """RenderedFrame can be constructed with required fields."""
        from vla_lego.sim.offscreen import RenderedFrame

        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = RenderedFrame(step_index=0, rgb=rgb)
        assert frame.step_index == 0
        assert frame.depth is None
        assert frame.segmentation is None


@pytest.mark.mujoco
class TestBackendValidation:
    """Test offscreen backend validation."""

    def test_validate_backend_succeeds(self, model) -> None:
        """validate_backend returns success=True on the lab PC."""
        from vla_lego.sim.offscreen import validate_backend

        result = validate_backend(model)
        assert result["success"] is True

    def test_validate_backend_has_expected_keys(self, model) -> None:
        """Result dict has all documented keys."""
        from vla_lego.sim.offscreen import validate_backend

        result = validate_backend(model)
        expected_keys = {"backend", "rgb_shape", "depth_shape", "seg_shape", "success"}
        assert expected_keys.issubset(result.keys())


@pytest.mark.mujoco
class TestRenderFrame:
    """Test single-frame rendering."""

    def test_rgb_shape_and_dtype(self, model_and_data) -> None:
        """RGB frame has correct shape and dtype."""
        from vla_lego.sim.offscreen import RenderConfig, create_renderer, render_frame

        model, data = model_and_data
        config = RenderConfig()
        renderer = create_renderer(model, config)
        try:
            frame = render_frame(renderer, model, data, step_index=0, config=config)
            assert frame.rgb.shape == (480, 640, 3)
            assert frame.rgb.dtype == np.uint8
        finally:
            renderer.close()

    def test_rgb_not_all_zeros(self, model_and_data, render_config) -> None:
        """Rendered scene is not all black (light + geometry present)."""
        from vla_lego.sim.offscreen import create_renderer, render_frame, resolve_camera_id

        model, data = model_and_data
        renderer = create_renderer(model, render_config)
        cam_id = resolve_camera_id(model, render_config.camera_name)
        try:
            frame = render_frame(
                renderer, model, data, step_index=0, config=render_config, camera_id=cam_id
            )
            assert frame.rgb.sum() > 0
        finally:
            renderer.close()

    def test_rgb_custom_resolution(self, model_and_data) -> None:
        """Custom resolution produces correctly sized frames."""
        from vla_lego.sim.offscreen import RenderConfig, create_renderer, render_frame

        model, data = model_and_data
        config = RenderConfig(width=320, height=240)
        renderer = create_renderer(model, config)
        try:
            frame = render_frame(renderer, model, data, step_index=0, config=config)
            assert frame.rgb.shape == (240, 320, 3)
        finally:
            renderer.close()

    def test_depth_shape_and_dtype(self, model_and_data) -> None:
        """Depth frame has correct shape and dtype."""
        from vla_lego.sim.offscreen import RenderConfig, create_renderer, render_frame

        model, data = model_and_data
        config = RenderConfig(render_depth=True)
        renderer = create_renderer(model, config)
        try:
            frame = render_frame(renderer, model, data, step_index=0, config=config)
            assert frame.depth is not None
            assert frame.depth.shape == (480, 640)
            assert frame.depth.dtype == np.float32
        finally:
            renderer.close()

    def test_depth_has_nonzero_values(self, model_and_data) -> None:
        """Depth map has non-zero values (floor/box visible)."""
        from vla_lego.sim.offscreen import RenderConfig, create_renderer, render_frame

        model, data = model_and_data
        config = RenderConfig(render_depth=True)
        renderer = create_renderer(model, config)
        try:
            frame = render_frame(renderer, model, data, step_index=0, config=config)
            assert frame.depth is not None
            assert frame.depth.max() > 0
        finally:
            renderer.close()

    def test_segmentation_shape_and_dtype(self, model_and_data) -> None:
        """Segmentation mask has correct shape and dtype."""
        from vla_lego.sim.offscreen import RenderConfig, create_renderer, render_frame

        model, data = model_and_data
        config = RenderConfig(render_segmentation=True)
        renderer = create_renderer(model, config)
        try:
            frame = render_frame(renderer, model, data, step_index=0, config=config)
            assert frame.segmentation is not None
            assert frame.segmentation.shape[:2] == (480, 640)
            assert frame.segmentation.dtype == np.int32
        finally:
            renderer.close()

    def test_step_index_stored(self, model_and_data) -> None:
        """step_index is correctly stored in the frame."""
        from vla_lego.sim.offscreen import RenderConfig, create_renderer, render_frame

        model, data = model_and_data
        config = RenderConfig()
        renderer = create_renderer(model, config)
        try:
            frame = render_frame(renderer, model, data, step_index=42, config=config)
            assert frame.step_index == 42
        finally:
            renderer.close()


@pytest.mark.mujoco
class TestRenderTrajectory:
    """Test trajectory rendering with sync contract."""

    def test_trajectory_frame_count(self, model_and_data) -> None:
        """Correct number of frames produced with render_every."""
        from vla_lego.sim.offscreen import RenderConfig, render_trajectory

        model, data = model_and_data
        frames = render_trajectory(model, data, n_steps=100, config=RenderConfig(), render_every=10)
        assert len(frames) == 10

    def test_trajectory_sync_contract(self, model_and_data) -> None:
        """Frame step_index matches the actual simulation step."""
        from vla_lego.sim.offscreen import RenderConfig, render_trajectory

        model, data = model_and_data
        frames = render_trajectory(model, data, n_steps=100, config=RenderConfig(), render_every=10)
        expected_steps = list(range(0, 100, 10))
        actual_steps = [f.step_index for f in frames]
        assert actual_steps == expected_steps

    def test_trajectory_every_step(self, model_and_data) -> None:
        """render_every=1 captures every step."""
        from vla_lego.sim.offscreen import RenderConfig, render_trajectory

        model, data = model_and_data
        frames = render_trajectory(model, data, n_steps=10, config=RenderConfig(), render_every=1)
        assert len(frames) == 10

    def test_trajectory_frames_differ(self, model_and_data, render_config) -> None:
        """First and last frames differ (box falls under gravity)."""
        from vla_lego.sim.offscreen import render_trajectory

        model, data = model_and_data
        frames = render_trajectory(model, data, n_steps=500, config=render_config, render_every=50)
        assert not np.array_equal(frames[0].rgb, frames[-1].rgb)

    def test_trajectory_deterministic(self) -> None:
        """Two runs from same initial state produce identical RGB arrays."""
        from vla_lego.sim.mujoco_env import load_model
        from vla_lego.sim.offscreen import RenderConfig, render_trajectory

        config = RenderConfig(width=160, height=120, camera_name=TEST_CAMERA)
        results = []
        for _ in range(2):
            model = load_model(TEST_SCENE)
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            frames = render_trajectory(model, data, n_steps=50, config=config, render_every=10)
            results.append([f.rgb for f in frames])

        for a, b in zip(results[0], results[1], strict=True):
            # atol=1 allows for GPU-accelerated renderer (EGL) rounding jitter
            assert np.allclose(a, b, atol=1)


@pytest.mark.mujoco
class TestCameraResolution:
    """Test camera name/ID resolution."""

    def test_free_camera_renders_without_error(self, model_and_data) -> None:
        """Default free camera (id=-1) renders without crashing."""
        from vla_lego.sim.offscreen import RenderConfig, create_renderer, render_frame

        model, data = model_and_data
        config = RenderConfig()
        renderer = create_renderer(model, config)
        try:
            frame = render_frame(renderer, model, data, step_index=0, config=config)
            assert frame.rgb.shape == (480, 640, 3)
        finally:
            renderer.close()

    def test_named_camera_renders_visible(self, model_and_data, render_config) -> None:
        """Named camera produces non-empty frame with visible scene."""
        from vla_lego.sim.offscreen import create_renderer, render_frame, resolve_camera_id

        model, data = model_and_data
        renderer = create_renderer(model, render_config)
        cam_id = resolve_camera_id(model, render_config.camera_name)
        try:
            frame = render_frame(
                renderer, model, data, step_index=0, config=render_config, camera_id=cam_id
            )
            assert frame.rgb.sum() > 0
        finally:
            renderer.close()

    def test_invalid_camera_name_raises(self, model) -> None:
        """Non-existent camera name raises ValueError."""
        from vla_lego.sim.offscreen import resolve_camera_id

        with pytest.raises(ValueError, match="not found"):
            resolve_camera_id(model, camera_name="nonexistent")


@pytest.mark.mujoco
class TestVideoExport:
    """Test video file export."""

    def test_save_video_creates_file(self, model_and_data, tmp_path) -> None:
        """MP4 is written and has nonzero size."""
        pytest.importorskip("imageio")
        from vla_lego.sim.offscreen import RenderConfig, render_trajectory, save_video

        model, data = model_and_data
        frames = render_trajectory(model, data, n_steps=30, config=RenderConfig(), render_every=1)
        out = save_video(frames, tmp_path / "test.mp4", fps=10)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_video_empty_frames_raises(self, tmp_path) -> None:
        """Empty frame list raises ValueError."""
        pytest.importorskip("imageio")
        from vla_lego.sim.offscreen import save_video

        with pytest.raises(ValueError, match="empty"):
            save_video([], tmp_path / "empty.mp4")


@pytest.mark.mujoco
class TestSampleFrameExport:
    """Test sample frame PNG export."""

    def test_save_sample_frames_creates_pngs(self, model_and_data, tmp_path) -> None:
        """Correct number of PNG files created."""
        pytest.importorskip("imageio")
        from vla_lego.sim.offscreen import RenderConfig, render_trajectory, save_sample_frames

        model, data = model_and_data
        frames = render_trajectory(model, data, n_steps=100, config=RenderConfig(), render_every=20)
        saved = save_sample_frames(frames, tmp_path / "samples", max_samples=3)
        pngs = list((tmp_path / "samples").glob("*.png"))
        assert len(pngs) == 3
        assert len(saved) == 3

    def test_save_sample_frames_with_depth(self, model_and_data, tmp_path) -> None:
        """Depth PNGs are also saved when depth frames present."""
        pytest.importorskip("imageio")
        from vla_lego.sim.offscreen import RenderConfig, render_trajectory, save_sample_frames

        model, data = model_and_data
        config = RenderConfig(render_depth=True)
        frames = render_trajectory(model, data, n_steps=20, config=config, render_every=10)
        saved = save_sample_frames(frames, tmp_path / "samples", max_samples=2)
        rgb_pngs = list((tmp_path / "samples").glob("*_rgb.png"))
        depth_pngs = list((tmp_path / "samples").glob("*_depth.png"))
        assert len(rgb_pngs) == 2
        assert len(depth_pngs) == 2
        # 2 rgb + 2 depth = 4 total
        assert len(saved) == 4
