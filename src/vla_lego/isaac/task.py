"""Visible five-block instructions and task-only scene specifications."""

from __future__ import annotations

from dataclasses import dataclass

from vla_lego.isaac.catalog import BlockPlacement, BlockSpec


@dataclass(frozen=True)
class TowerTask:
    placements: tuple[BlockPlacement, ...]
    targets: tuple[BlockSpec, ...]
    assembly_xy_m: tuple[float, float]

    def __post_init__(self) -> None:
        if len(self.targets) != 5:
            raise ValueError("The task requires exactly five ordered blocks.")
        remaining = [placement.spec for placement in self.placements]
        for target in self.targets:
            if target in remaining:
                remaining.remove(target)
        if not any(
            distractor.color == target.color and distractor.scale != target.scale
            for distractor in remaining
            for target in self.targets
        ):
            raise ValueError("A requested-color/wrong-size distractor is required.")
        if not any(
            distractor.scale == target.scale and distractor.color != target.color
            for distractor in remaining
            for target in self.targets
        ):
            raise ValueError("A requested-size/wrong-color distractor is required.")

    @property
    def requested_pieces_available(self) -> bool:
        remaining = [placement.spec for placement in self.placements]
        for target in self.targets:
            if target not in remaining:
                return False
            remaining.remove(target)
        return True

    @property
    def instruction(self) -> str:
        order = ", then ".join(target.description for target in self.targets)
        return f"Build a five-block tower inside the black square, bottom to top: {order}."


def initial_tower_task() -> TowerTask:
    """First loose-piece scene; its control/reach coverage is still unqualified."""
    red = BlockSpec("cube", "small", "red")
    blue = BlockSpec("cube", "large", "blue")
    green = BlockSpec("prism", "small", "green")
    yellow = BlockSpec("prism", "large", "yellow")
    purple = BlockSpec("cube", "small", "purple")
    return TowerTask(
        placements=(
            BlockPlacement(red, (0.45, 0.15)),
            BlockPlacement(blue, (0.45, -0.30)),
            BlockPlacement(green, (0.56, 0.30)),
            BlockPlacement(yellow, (0.56, -0.15)),
            BlockPlacement(purple, (0.45, -0.45)),
            BlockPlacement(BlockSpec("cube", "large", "red"), (0.60, 0.45)),
            BlockPlacement(BlockSpec("cube", "small", "orange"), (0.60, 0.0)),
        ),
        targets=(red, blue, green, yellow, purple),
        assembly_xy_m=(0.40, 0.0),
    )
