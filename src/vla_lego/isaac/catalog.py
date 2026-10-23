"""Printable seed cuboids and independent visible attributes."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

ASSET_DIRECTORY = Path(__file__).parent / "assets" / "blocks"
COLORS = {
    "red": (0.80, 0.03, 0.03),
    "blue": (0.03, 0.15, 0.80),
    "green": (0.03, 0.65, 0.10),
    "yellow": (0.90, 0.80, 0.03),
    "orange": (0.95, 0.30, 0.03),
    "purple": (0.55, 0.05, 0.75),
}
SCALES = {"small": 0.8, "large": 1.0}
# Effective printed bulk density: 25 g for a 40 mm cube. Unmeasured assumption.
PROVISIONAL_DENSITY_KG_M3 = 0.025 / 0.04**3


def source_dimensions(shape: str) -> tuple[float, float, float]:
    """Read the admitted centered cuboid STLs; keep source units in meters."""
    path = ASSET_DIRECTORY / {"cube": "cube_s0.stl", "prism": "prism_f6.stl"}[shape]
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if count != 12 or len(raw) != 84 + count * 50:
        raise ValueError(f"Expected the admitted 12-triangle source cuboid: {path}")
    vertices = [
        struct.unpack_from("<3f", raw, 84 + triangle * 50 + 12 + vertex * 12)
        for triangle in range(count)
        for vertex in range(3)
    ]
    minima = [min(v[axis] for v in vertices) for axis in range(3)]
    maxima = [max(v[axis] for v in vertices) for axis in range(3)]
    if any(abs(lo + hi) > 1e-7 for lo, hi in zip(minima, maxima, strict=True)):
        raise ValueError("Source geometry must be centered at its grasp/support frame.")
    if any(any(abs(v[axis]) < maxima[axis] - 1e-7 for axis in range(3)) for v in vertices):
        raise ValueError("Source vertices do not lie on the admitted cuboid corners.")
    dimensions = [hi - lo for lo, hi in zip(minima, maxima, strict=True)]
    return dimensions[0], dimensions[1], dimensions[2]


@dataclass(frozen=True)
class BlockSpec:
    shape: str
    scale: str
    color: str

    @property
    def dimensions_m(self) -> tuple[float, float, float]:
        scale = SCALES[self.scale]
        x, y, z = source_dimensions(self.shape)
        return x * scale, y * scale, z * scale

    @property
    def mass_kg(self) -> float:
        x, y, z = self.dimensions_m
        return PROVISIONAL_DENSITY_KG_M3 * x * y * z

    @property
    def rgb(self) -> tuple[float, float, float]:
        return COLORS[self.color]

    @property
    def description(self) -> str:
        return f"{self.scale} {self.color} {self.shape}"


@dataclass(frozen=True)
class BlockPlacement:
    spec: BlockSpec
    xy_m: tuple[float, float]
    yaw_deg: float = 0.0


def catalog() -> tuple[BlockSpec, ...]:
    """Full 2 x 2 x 6 simulation crossing; physical sample measurements stay open."""
    return tuple(
        BlockSpec(shape, scale, color)
        for shape in ("cube", "prism")
        for scale in SCALES
        for color in COLORS
    )
