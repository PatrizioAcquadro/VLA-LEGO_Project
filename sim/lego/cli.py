"""CLI entry point for LEGO brick asset generation (Phase 1.2.1).

Usage:
    vla-gen-bricks              # generate all standard bricks
    python -m sim.lego.cli      # same
"""

from __future__ import annotations

from sim.lego.brick_generator import write_brick_assets
from sim.lego.constants import BRICK_TYPES
from sim.lego.mass import compute_brick_mass


def main() -> None:
    """Generate LEGO brick MJCF assets and print summary."""
    paths = write_brick_assets()
    print(f"Generated {len(paths)} LEGO brick assets:")
    for path in paths:
        print(f"  {path}")

    print("\nBrick summary:")
    for name, brick in BRICK_TYPES.items():
        mass = compute_brick_mass(brick)
        print(
            f"  {name}: {brick.n_studs} studs, {brick.n_tubes} tubes, " f"mass={mass * 1000:.1f} g"
        )


if __name__ == "__main__":
    main()
