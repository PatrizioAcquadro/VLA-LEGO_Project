"""Frozen geometry and physics constants for LEGO simulation (Phase 1.2.1).

All dimensions in meters (MuJoCo convention). Values frozen from
press-fit spec (docs/press-fit-spec.md, Sections 3 & 8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Stud geometry (frozen) ---
STUD_PITCH: float = 0.008  # 8.0 mm center-to-center
STUD_COLLISION_RADIUS: float = 0.00235  # 2.35 mm (50 um undersize for clearance)
STUD_VISUAL_RADIUS: float = 0.0024  # 2.4 mm (true LEGO dimension)
STUD_HALF_HEIGHT: float = 0.00085  # half of 1.7 mm
STUD_HEIGHT: float = 0.0017  # 1.7 mm above brick top

# --- Tube geometry (frozen) ---
TUBE_RING_RADIUS: float = 0.0024  # 2.4 mm (capsule centers)
TUBE_CAPSULE_RADIUS: float = 0.0005  # 0.5 mm
TUBE_CAPSULE_HALF_HEIGHT: float = 0.0008  # half of 1.6 mm
TUBE_CAPSULE_COUNT: int = 8
TUBE_CAPSULE_ANGULAR_SPACING: float = 2.0 * math.pi / TUBE_CAPSULE_COUNT  # 45 deg

# --- Brick shell geometry (frozen) ---
BRICK_HEIGHT: float = 0.0096  # 9.6 mm body only
WALL_THICKNESS: float = 0.0012  # 1.2 mm
TOP_THICKNESS: float = 0.001  # 1.0 mm top wall
INTER_BRICK_GAP: float = 0.0002  # 0.2 mm total (0.1 per side)

# --- Material ---
DENSITY_ABS: float = 1050.0  # kg/m^3 (ABS plastic)

# --- Contact isolation (frozen from spec Section 4.4) ---
LEGO_CONTYPE: int = 2
LEGO_CONAFFINITY_INTERNAL: int = 2  # stud/tube: LEGO-to-LEGO only
LEGO_CONAFFINITY_SURFACE: int = 3  # brick surface: LEGO + robot

# --- Geom groups ---
COLLISION_GROUP: int = 3  # matches Alex robot collision geoms
VISUAL_GROUP: int = 0  # visual-only geoms

# --- Default brick color (LEGO red, RGBA) ---
DEFAULT_BRICK_COLOR: tuple[float, float, float, float] = (0.78, 0.09, 0.09, 1.0)


@dataclass(frozen=True)
class BrickType:
    """Parametric definition of a LEGO brick type."""

    name: str  # e.g., "2x2"
    nx: int  # studs along X
    ny: int  # studs along Y
    height: float = BRICK_HEIGHT

    @property
    def n_studs(self) -> int:
        return self.nx * self.ny

    @property
    def n_tubes(self) -> int:
        """For 2-wide bricks: ny - 1 tubes along centerline."""
        return self.ny - 1

    @property
    def shell_half_x(self) -> float:
        return (self.nx * STUD_PITCH - INTER_BRICK_GAP) / 2.0

    @property
    def shell_half_y(self) -> float:
        return (self.ny * STUD_PITCH - INTER_BRICK_GAP) / 2.0

    @property
    def shell_half_z(self) -> float:
        return self.height / 2.0


BRICK_TYPES: dict[str, BrickType] = {
    "2x2": BrickType(name="2x2", nx=2, ny=2),
    "2x4": BrickType(name="2x4", nx=2, ny=4),
    "2x6": BrickType(name="2x6", nx=2, ny=6),
}
