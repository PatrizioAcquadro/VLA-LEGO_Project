"""LEGO brick generation and metadata (Phase 1.2.1).

Provides procedural MJCF generation for LEGO brick models,
connector metadata, and mass computation.
"""

from __future__ import annotations

from sim.lego.brick_generator import (
    generate_brick_body_xml,
    generate_brick_mjcf,
    write_brick_assets,
)
from sim.lego.connector import BrickConnectors, ConnectorPoint, get_brick_connectors
from sim.lego.constants import (
    BRICK_HEIGHT,
    BRICK_TYPES,
    DEFAULT_BRICK_COLOR,
    DENSITY_ABS,
    INTER_BRICK_GAP,
    STUD_COLLISION_RADIUS,
    STUD_HEIGHT,
    STUD_PITCH,
    STUD_VISUAL_RADIUS,
    TUBE_CAPSULE_COUNT,
    TUBE_CAPSULE_HALF_HEIGHT,
    TUBE_CAPSULE_RADIUS,
    TUBE_RING_RADIUS,
    WALL_THICKNESS,
    BrickType,
)
from sim.lego.mass import compute_brick_mass

__all__ = [
    "BRICK_HEIGHT",
    "BRICK_TYPES",
    "BrickConnectors",
    "BrickType",
    "ConnectorPoint",
    "DEFAULT_BRICK_COLOR",
    "DENSITY_ABS",
    "INTER_BRICK_GAP",
    "STUD_COLLISION_RADIUS",
    "STUD_HEIGHT",
    "STUD_PITCH",
    "STUD_VISUAL_RADIUS",
    "TUBE_CAPSULE_COUNT",
    "TUBE_CAPSULE_HALF_HEIGHT",
    "TUBE_CAPSULE_RADIUS",
    "TUBE_RING_RADIUS",
    "WALL_THICKNESS",
    "compute_brick_mass",
    "generate_brick_body_xml",
    "generate_brick_mjcf",
    "get_brick_connectors",
    "write_brick_assets",
]
