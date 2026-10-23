"""Approved pilot cases, with families fixed before any windowing or augmentation."""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from vla_lego.isaac.catalog import BlockPlacement, BlockSpec
from vla_lego.isaac.task import TowerTask

CODES = {
    "CS": ("cube", "small"),
    "CL": ("cube", "large"),
    "PS": ("prism", "small"),
    "PL": ("prism", "large"),
}
RESERVED = {("PS", "blue"), ("PL", "purple"), ("CS", "yellow"), ("CL", "green")}
FORMS = (
    "Build a five-block tower inside the black square. From bottom to top: {}.",
    "Inside the black square, stack these five blocks in this order, starting at the bottom: {}.",
    "Make a tower in the marked square: {}, listed from the base to the top.",
)


def piece(value: str) -> BlockSpec:
    color, code = value.strip().split()
    if (code, color) in RESERVED:
        raise ValueError(f"Reserved pilot combination: {value}")
    shape, scale = CODES[code]
    spec = BlockSpec(shape, scale, color)
    _ = spec.rgb  # Validate color without admitting reserved combinations.
    return spec


@dataclass(frozen=True)
class PilotCase:
    row: dict[str, str]
    ordinal: int

    @property
    def name(self) -> str:
        return self.row["case"]

    @property
    def category(self) -> str:
        return self.row["category"]

    @property
    def cooperative_level(self) -> int:
        return int(self.row["cooperative_placement"] or 0) - 1

    @property
    def error_level(self) -> int:
        return int(self.row["error_target_level"] or 0) - 1

    @property
    def mass_scale(self) -> float:
        return (
            float(self.row["variation"].split("mass factor ")[-1])
            if "mass factor" in self.row["variation"]
            else 1.0
        )

    @property
    def friction(self) -> tuple[float, float]:
        value = self.row["variation"]
        return (
            (0.45, 0.35)
            if "low 0.45" in value
            else (0.8, 0.65) if "high 0.8" in value else (0.6, 0.5)
        )

    @property
    def appearance(self) -> str:
        return (
            "neutral_side"
            if self.name.endswith("N4") or self.name in ("D1-N3", "D2-N3")
            else "light_wood"
        )

    def task(self) -> TowerTask:
        targets = tuple(piece(value) for value in self.row["order"].split(" > "))
        inventory = [*targets, *(piece(v) for v in self.row["distractors"].split("; "))]
        if self.category == "stop":
            inventory[inventory.index(piece(self.row["missing_piece"]))] = piece(
                self.row["replacement"]
            )
        if self.name.startswith("T") and self.name.endswith("N2"):
            base_row = dict(
                self.row,
                case=self.name[:-1] + "1",
                order=" > ".join(reversed(self.row["order"].split(" > "))),
                stabilizer="right" if self.row["stabilizer"] == "left" else "left",
                variation="nominal spread",
            )
            base = PilotCase(base_row, self.ordinal - 1).task()
            return replace(
                base,
                targets=targets,
                placements=tuple(
                    replace(p, xy_m=(p.xy_m[0], -p.xy_m[1]), yaw_deg=-p.yaw_deg)
                    for p in base.placements
                ),
                instruction_text=FORMS[self.ordinal % 3].format(
                    ", then ".join(
                        t.description.replace("prism", "rectangular prism") for t in targets
                    )
                ),
            )
        # Permute all eight source positions so location does not encode target/distractor.
        rng = random.Random(20261023 + self.ordinal * 101)
        slots = [(x, y) for x in (0.43, 0.54) for y in (-0.45, -0.15, 0.15, 0.45)]
        rng.shuffle(slots)
        inherited = self.name.startswith("T") and self.name.endswith("N3")
        if inherited:
            order = self.row["order"].split(" > ")
            base_row = dict(
                self.row,
                case=self.name[:-1] + "1",
                order=" > ".join(order[-2:] + order[:-2]),
                cooperative_placement="4",
                variation="nominal spread",
            )
            base = PilotCase(base_row, self.ordinal - 2).task()
            coordinates = {p.spec: p.xy_m for p in base.placements}
            slots = [coordinates[spec] for spec in inventory]
            # Swap the actual near/far size-confusable sources, then relocate
            # the layout by 6 cm and add the approved yaw changes.
            slots[3], slots[5] = slots[5], slots[3]
        if self.category != "stop":
            stabilizer = self.row["stabilizer"]
            requirements = [
                (self.cooperative_level - 1, stabilizer),
                (self.cooperative_level, "right" if stabilizer == "left" else "left"),
            ]
            if self.name.endswith("N1") and self.name.startswith("T"):
                requirements.append((1, "right" if stabilizer == "left" else "left"))
            protected = {index for index, _ in requirements}
            if inherited:
                protected.update((3, 5))
            for index, side in requirements:
                if (slots[index][1] > 0) != (side == "left"):
                    other = next(
                        j
                        for j, xy in enumerate(slots)
                        if j not in protected and (xy[1] > 0) == (side == "left")
                    )
                    slots[index], slots[other] = slots[other], slots[index]
            if (
                self.name.startswith("T")
                and self.name.endswith("N1")
                and slots[0][0] == slots[5][0]
            ):
                other = next(
                    j
                    for j, xy in enumerate(slots)
                    if j not in (0, 1, 2, 3, 5) and xy[0] != slots[0][0]
                )
                slots[5], slots[other] = slots[other], slots[5]
        relocated = (
            self.name.endswith("N3") or self.name.startswith("D") and self.name.endswith("N2")
        )
        placements = []
        for spec, (x, y) in zip(inventory, slots, strict=True):
            # Independent reachable layouts; N3 adds a 6 cm relocation and 15–35° yaw.
            if not inherited:
                x += ((self.ordinal % 3) - 1) * 0.01
            if relocated:
                y += (-0.06 if abs(y) > 0.3 else 0.06) * (1 if y > 0 else -1)
            yaw = rng.choice((-35.0, -25.0, 20.0, 30.0)) if relocated else 0.0
            placements.append(BlockPlacement(spec, (x, y), yaw_deg=yaw))
        return TowerTask(
            placements=tuple(placements),
            targets=targets,
            assembly_xy_m=(0.40, 0.0),
            instruction_text=FORMS[self.ordinal % 3].format(
                ", then ".join(t.description.replace("prism", "rectangular prism") for t in targets)
            ),
        )


def load_cases(path: Path) -> tuple[PilotCase, ...]:
    with path.open(newline="") as stream:
        cases = tuple(PilotCase(dict(row), i) for i, row in enumerate(csv.DictReader(stream)))
    names = {c.name for c in cases}
    counts = Counter((c.row["split"], c.category) for c in cases)
    expected = {
        ("train", "nominal"): 30,
        ("train", "recovery"): 15,
        ("train", "stop"): 5,
        ("development", "nominal"): 6,
        ("development", "recovery"): 3,
        ("development", "stop"): 1,
    }
    if len(names) != 60 or dict(counts) != expected:
        raise ValueError("Expected the approved 50+10 case allocation without duplicates.")
    families: dict[str, str] = {}
    for case in cases:
        family, split = case.row["family"], case.row["split"]
        if families.setdefault(family, split) != split or split != (
            "train" if family.startswith("T") else "development"
        ):
            raise ValueError("Scene families and descendants must stay within one split.")
        task = case.task()
        if len(task.placements) != 8 or task.requested_pieces_available != (
            case.category != "stop"
        ):
            raise ValueError(f"Incorrect eight-object inventory: {case.name}")
        if len({p.spec for p in task.placements}) != 8:
            raise ValueError(f"Inventory contains indistinguishable duplicates: {case.name}")
    return cases
