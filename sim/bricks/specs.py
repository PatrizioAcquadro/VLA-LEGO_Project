"""Metric dimensions and effective printed masses for lightweight cuboids."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from math import isfinite
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class BrickSpec:
    """A box centered on its body origin; size is XYZ in metres, mass in kg."""

    name: str
    size: tuple[float, float, float]
    mass: float
    color: tuple[float, float, float, float] = (0.8, 0.2, 0.1, 1.0)

    def __post_init__(self) -> None:
        if not self.name or len(self.size) != 3:
            raise ValueError("A brick needs a name and three dimensions")
        if any(not isfinite(x) or x <= 0 for x in (*self.size, self.mass)):
            raise ValueError("Dimensions and mass must be finite and positive")
        if len(self.color) != 4 or any(not isfinite(x) or not 0 <= x <= 1 for x in self.color):
            raise ValueError("Color must be four finite RGBA values in [0, 1]")


def load_bricks(path: str | Path | None = None) -> tuple[BrickSpec, ...]:
    """Load a catalogue; masses are explicit estimates, not solid-plastic density."""
    source = path if path is not None else files("configs") / "bricks" / "default.yaml"
    cfg = OmegaConf.load(str(source))
    bricks = tuple(
        BrickSpec(str(item.name), tuple(item.size), float(item.mass), tuple(item.color))
        for item in cfg.catalog
    )
    if not bricks:
        raise ValueError("The brick catalogue cannot be empty")
    return bricks
