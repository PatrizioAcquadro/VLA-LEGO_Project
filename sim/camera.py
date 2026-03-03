"""Frozen 2-view camera contract for the Alex robot (Phase 1.1.7).

Defines the canonical camera configuration, multi-view rendering pipeline,
and camera metadata extraction for synchronized dual-camera capture.

Camera views (frozen):
    - robot_cam: Head-mounted ego-centric (moves with torso via spine_z)
    - third_person: Fixed external observation camera

Usage::

    from sim.camera import MultiViewRenderer, CAMERA_NAMES

    model = load_scene("alex_upper_body")
    data = mujoco.MjData(model)
    renderer = MultiViewRenderer(model)
    frame = renderer.capture(data, step_index=0)
    frame.views["robot_cam"].rgb   # (H, W, 3) uint8
    frame.views["third_person"].rgb
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.offscreen import (
    RenderConfig,
    RenderedFrame,
    create_renderer,
    render_frame,
    resolve_camera_id,
)

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

CAMERA_NAMES: tuple[str, ...] = ("robot_cam", "third_person")
"""Canonical camera view names (frozen for dataset compatibility)."""

NUM_VIEWS: int = 2
"""Number of camera views in the frozen contract."""

DEFAULT_WIDTH: int = 320
"""Default render width in pixels."""

DEFAULT_HEIGHT: int = 240
"""Default render height in pixels."""

DEFAULT_CAPTURE_HZ: float = 20.0
"""Default capture rate aligned with policy control rate (Hz)."""

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraMetadata:
    """Intrinsic and extrinsic metadata for a single camera.

    Attributes:
        name: Camera name in MJCF.
        camera_id: Resolved MuJoCo camera ID.
        fovy: Vertical field of view (degrees).
        pos: Camera position in world frame, shape (3,).
        mat: Camera orientation matrix in world frame, shape (3, 3).
        width: Render width in pixels.
        height: Render height in pixels.
    """

    name: str
    camera_id: int
    fovy: float
    pos: np.ndarray
    mat: np.ndarray
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "name": self.name,
            "camera_id": self.camera_id,
            "fovy": self.fovy,
            "pos": self.pos.tolist(),
            "mat": self.mat.tolist(),
            "width": self.width,
            "height": self.height,
        }


@dataclass
class MultiViewFrame:
    """Synchronized multi-view capture at a single simulation timestep.

    All views are rendered from the exact same simulation state before
    any physics stepping occurs, guaranteeing timestep synchronization.

    Attributes:
        step_index: Simulation step (control tick) when captured.
        timestamp: Simulation time in seconds.
        views: Dict mapping camera name to RenderedFrame.
    """

    step_index: int
    timestamp: float
    views: dict[str, RenderedFrame] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def get_camera_metadata(
    model: mujoco.MjModel,  # type: ignore[name-defined]
    data: mujoco.MjData,  # type: ignore[name-defined]
    camera_name: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> CameraMetadata:
    """Extract intrinsic/extrinsic metadata for a named camera.

    Uses ``data.cam_xpos`` / ``data.cam_xmat`` for world-frame pose,
    which is critical for body-attached cameras like ``robot_cam`` whose
    position changes as the robot moves.

    Args:
        model: Compiled MuJoCo model.
        data: MuJoCo data (must have had ``mj_forward`` called).
        camera_name: MJCF camera name.
        width: Render width.
        height: Render height.

    Returns:
        CameraMetadata with live world-frame pose.
    """
    cam_id = resolve_camera_id(model, camera_name)
    return CameraMetadata(
        name=camera_name,
        camera_id=cam_id,
        fovy=float(model.cam_fovy[cam_id]),
        pos=data.cam_xpos[cam_id].copy(),
        mat=data.cam_xmat[cam_id].reshape(3, 3).copy(),
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# MultiViewRenderer
# ---------------------------------------------------------------------------


class MultiViewRenderer:
    """Synchronized multi-view renderer for the frozen 2-camera setup.

    Creates a single MuJoCo ``Renderer`` (shared resolution) and renders
    all frozen camera views from the same simulation state.

    Args:
        model: Compiled MuJoCo model.
        width: Render width (default 320).
        height: Render height (default 240).
        render_depth: Also capture depth maps.
        render_segmentation: Also capture segmentation masks.
    """

    def __init__(
        self,
        model: mujoco.MjModel,  # type: ignore[name-defined]
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        render_depth: bool = False,
        render_segmentation: bool = False,
    ) -> None:
        self._model = model
        self._width = width
        self._height = height

        self._config = RenderConfig(
            width=width,
            height=height,
            render_depth=render_depth,
            render_segmentation=render_segmentation,
        )

        # Resolve camera IDs for all frozen views
        self._camera_ids: dict[str, int] = {}
        for name in CAMERA_NAMES:
            self._camera_ids[name] = resolve_camera_id(model, name)

        # Create single shared renderer
        self._renderer = create_renderer(model, self._config)
        self._closed = False

    @property
    def camera_ids(self) -> dict[str, int]:
        """Mapping of camera name to resolved MuJoCo camera ID."""
        return dict(self._camera_ids)

    @property
    def width(self) -> int:
        """Render width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Render height in pixels."""
        return self._height

    def capture(
        self,
        data: mujoco.MjData,  # type: ignore[name-defined]
        step_index: int,
    ) -> MultiViewFrame:
        """Render all views from the current simulation state.

        Both cameras are rendered before any physics stepping, guaranteeing
        timestep synchronization. Each ``RenderedFrame`` in the returned
        ``MultiViewFrame.views`` shares the same ``step_index``.

        Args:
            data: MuJoCo data at current state.
            step_index: Current simulation step for sync tracking.

        Returns:
            MultiViewFrame with all camera views.

        Raises:
            RuntimeError: If renderer has been closed.
        """
        if self._closed:
            raise RuntimeError("MultiViewRenderer has been closed")

        views: dict[str, RenderedFrame] = {}
        for name, cam_id in self._camera_ids.items():
            # render_frame calls mj_forward internally (idempotent per timestep)
            frame = render_frame(
                self._renderer, self._model, data, step_index, self._config, cam_id
            )
            views[name] = frame

        return MultiViewFrame(
            step_index=step_index,
            timestamp=float(data.time),
            views=views,
        )

    def get_metadata(
        self,
        data: mujoco.MjData,  # type: ignore[name-defined]
    ) -> dict[str, CameraMetadata]:
        """Return camera metadata with live world-frame poses.

        For body-attached cameras (``robot_cam``), the position updates
        as the robot moves. Call ``mj_forward`` before this method.

        Args:
            data: MuJoCo data (must have had ``mj_forward`` called).

        Returns:
            Dict mapping camera name to CameraMetadata.
        """
        return {
            name: get_camera_metadata(self._model, data, name, self._width, self._height)
            for name in CAMERA_NAMES
        }

    def close(self) -> None:
        """Release the renderer resources."""
        if not self._closed:
            self._renderer.close()
            self._closed = True

    def __enter__(self) -> MultiViewRenderer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
