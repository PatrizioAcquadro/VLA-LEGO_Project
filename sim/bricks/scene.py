"""Plain contact geometry shared by generated assets and live episodes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence

from sim.asset_loader import ROBOTS_DIR, SCENES_DIR
from sim.bricks.specs import BrickSpec

TABLE_CENTER = (0.45, 0.0)
TABLE_HALF_SIZE = (0.25, 0.30)
TABLE_TOP = 0.77


def add_brick(root: ET.Element, spec: BrickSpec, index: int, z: float) -> None:
    """One collision box also supplies visualization and approximate box inertia."""
    body = ET.SubElement(root, "body", name=f"brick_{index}", pos=f"0.45 0 {z:g}")
    ET.SubElement(body, "freejoint", name=f"brick_{index}_joint")
    ET.SubElement(
        body,
        "geom",
        name=f"brick_{index}_geom",
        type="box",
        size=" ".join(f"{x / 2:g}" for x in spec.size),
        mass=f"{spec.mass:g}",
        rgba=" ".join(f"{x:g}" for x in spec.color),
        friction="0.6 0.005 0.0001",
        solref="0.01 1",
        condim="3",
    )


def scene_xml(bricks: Sequence[BrickSpec], *, include_robot: bool = True) -> str:
    """Build an episode scene. Robot asset paths are resolved at runtime."""
    if not bricks:
        raise ValueError("An episode needs at least one brick")
    if include_robot:
        root = ET.parse(SCENES_DIR / "alex_upper_body.xml").getroot()
        compiler, include, world = (
            root.find("compiler"),
            root.find("include"),
            root.find("worldbody"),
        )
        if compiler is None or include is None or world is None:
            raise ValueError("Alex scene must contain compiler, include and worldbody")
        compiler.set("meshdir", str(ROBOTS_DIR / "alex"))
        include.set("file", str(ROBOTS_DIR / "alex" / "alex.xml"))
    else:
        root = ET.Element("mujoco", model="lightweight_bricks")
        ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="0.45 0 2", dir="0 0 -1", directional="true")
        ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05")

    root.set("model", "alex_bricks" if include_robot else "lightweight_bricks")
    for camera in list(world.findall("camera")):
        world.remove(camera)
    ET.SubElement(world, "camera", name="overhead", pos="0.45 0 1.6", fovy="50")
    ET.SubElement(
        world,
        "camera",
        name="third_person",
        pos="1.2 -0.9 1.35",
        xyaxes="0.768 0.64 0 -0.294 0.353 0.888",
        fovy="55",
    )
    ET.SubElement(
        world,
        "camera",
        name="workspace_closeup",
        pos="0.75 -0.45 1.2",
        xyaxes="0.832 0.555 0 -0.346 0.519 0.781",
        fovy="50",
    )
    ET.SubElement(
        world,
        "geom",
        name="table_surface",
        type="box",
        pos="0.45 0 0.75",
        size="0.25 0.30 0.02",
        rgba="0.55 0.46 0.35 1",
        friction="0.6 0.005 0.0001",
    )
    qpos: list[float] = []
    bottom = TABLE_TOP
    for i, spec in enumerate(bricks):
        z = bottom + spec.size[2] / 2
        add_brick(world, spec, i, z)
        qpos.extend((TABLE_CENTER[0], TABLE_CENTER[1], z, 1, 0, 0, 0))
        bottom += spec.size[2]
    suffix = " ".join(f"{x:g}" for x in qpos)
    for key in root.findall("keyframe/key"):
        key.set("qpos", key.get("qpos", "") + " " + suffix)
    return ET.tostring(root, encoding="unicode")
