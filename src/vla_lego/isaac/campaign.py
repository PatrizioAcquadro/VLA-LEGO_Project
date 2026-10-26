"""Declared data-expansion cases; historical pilot membership remains unchanged."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vla_lego.isaac.catalog import BlockPlacement
from vla_lego.isaac.pilot import BUILD_REGION, PilotCase, piece
from vla_lego.isaac.task import SingleBlockTask, TowerTask

CONDITION = "alex003_data_expansion_v1"
RECOVERIES = ("missed_grasp", "slip_drop", "wrong_piece", "placement", "interrupted_progress")
EXPECTED = {
    ("train", "single_block"): 160,
    ("train", "tower"): 240,
    ("train", "recovery"): 75,
    ("train", "stop"): 25,
    ("calibration", "single_block"): 15,
    ("calibration", "tower"): 25,
    ("calibration", "recovery"): 10,
    ("calibration", "stop"): 10,
    ("confirmation", "single_block"): 15,
    ("confirmation", "tower"): 25,
    ("confirmation", "recovery"): 10,
    ("confirmation", "stop"): 10,
}


@dataclass(frozen=True)
class CollectionCase(PilotCase):
    declaration: dict

    @property
    def mass_scale(self) -> float:
        return float(self.declaration["mass_scale"])

    @property
    def friction(self) -> tuple[float, float]:
        a, b = self.declaration["friction"]
        return float(a), float(b)

    @property
    def appearance(self) -> str:
        return str(self.declaration["appearance"])

    def task(self) -> TowerTask:
        cls = SingleBlockTask if self.category == "single_block" else TowerTask
        return cls(
            placements=tuple(
                BlockPlacement(piece(p["piece"]), tuple(p["xy_m"]), p["yaw_deg"])
                for p in self.declaration["placements"]
            ),
            targets=tuple(piece(p) for p in self.declaration["targets"]),
            assembly_xy_m=tuple(self.declaration["assembly_xy_m"]),
            build_region_xy_m=BUILD_REGION,
            instruction_text=self.declaration["instruction"],
        )


def load_campaign(path: Path) -> tuple[CollectionCase, ...]:
    data = json.loads(path.read_text())
    if data.get("collection_condition") != CONDITION:
        raise ValueError("Expected the declared Alex003 expansion campaign")
    cases = []
    errors = {
        "missed_grasp": "missed grasp",
        "wrong_piece": "wrong distractor picked",
        "placement": "physically consequential incomplete placement",
        "slip_drop": "slip_drop",
        "interrupted_progress": "interrupted_progress",
    }
    for i, item in enumerate(data["cases"]):
        row = dict(item["row"])
        if row["category"] == "recovery":
            row["error"] = errors[row["mechanism"]]
            row["wrong_piece"] = next(
                p["piece"] for p in item["placements"] if p["piece"] not in item["targets"]
            )
        cases.append(CollectionCase(row, i, item))
    if Counter((c.row["partition"], c.category) for c in cases) != EXPECTED:
        raise ValueError("Campaign must contain the complete declared 500/60/60 allocation")
    if len({c.name for c in cases}) != len(cases):
        raise ValueError("Duplicate campaign case name")
    families: dict[str, str] = {}
    pairs: dict[tuple, list[CollectionCase]] = {}
    for case in cases:
        row, task = case.row, case.task()
        partition = row["partition"]
        if families.setdefault(row["family"], partition) != partition:
            raise ValueError("A generation family crosses partitions")
        if row["split"] != ("train" if partition == "train" else "development"):
            raise ValueError("Incorrect recorder split")
        if len(task.placements) != 8 or len({p.spec for p in task.placements}) != 8:
            raise ValueError("Every campaign scene requires eight visibly distinct pieces")
        if any(not math.isfinite(v) for p in task.placements for v in (*p.xy_m, p.yaw_deg)):
            raise ValueError("Non-finite scene layout")
        if case.category != "stop" and not task.requested_pieces_available:
            raise ValueError("Solvable demonstration is missing a required piece")
        if case.category == "recovery" and row["mechanism"] not in RECOVERIES:
            raise ValueError("Unknown recovery mechanism")
        if case.category in ("single_block", "tower") and partition == "train":
            pairs.setdefault((row["family"], row["repetition"]), []).append(case)
    for pair in pairs.values():
        if len(pair) != 2:
            raise ValueError("Every TRAIN reset requires both instruction alternatives")
        a, b = pair
        if a.task().placements != b.task().placements or a.task().targets == b.task().targets:
            raise ValueError("Instruction pairs must preserve the physical scene and change goals")
        if any(
            a.declaration[k] != b.declaration[k]
            for k in ("mass_scale", "friction", "appearance", "assembly_xy_m")
        ):
            raise ValueError("An instruction pair changes physics or destination")
    return tuple(cases)
