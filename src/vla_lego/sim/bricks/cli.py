"""Export portable standalone MJCF assets for the configured lightweight bricks."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from vla_lego.sim.bricks.scene import TABLE_TOP, add_brick
from vla_lego.sim.bricks.specs import load_bricks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    bricks = load_bricks(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for i, spec in enumerate(bricks):
        root = ET.Element("mujoco", model=spec.name)
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="0 0 2", directional="true")
        ET.SubElement(world, "geom", type="plane", size="0 0 0.05", pos=f"0 0 {TABLE_TOP}")
        add_brick(world, spec, i, TABLE_TOP + spec.size[2] / 2)
        path = args.output_dir / f"brick_{i}.xml"
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding="unicode")
        print(f"{spec.name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
