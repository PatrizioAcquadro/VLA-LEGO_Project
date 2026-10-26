"""Construct the fixed, family-separated 500/120 data allocation."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from vla_lego.isaac.campaign import CONDITION, RECOVERIES, load_campaign
from vla_lego.isaac.pilot import BUILD_REGION, CODES, FORMS, RESERVED, piece

PIECES = tuple(
    f"{color} {code}"
    for code in CODES
    for color in ("red", "blue", "green", "yellow", "orange", "purple")
    if (code, color) not in RESERVED
)
LEFT_PAIRS = (
    ("red CS", "red CL"),
    ("red PS", "red PL"),
    ("blue CS", "blue CL"),
    ("blue PL", "orange PL"),
    ("green CS", "green PS"),
    ("green PL", "yellow PL"),
    ("yellow CL", "yellow PS"),
    ("orange CS", "orange CL"),
    ("orange PS", "purple PS"),
    ("purple CS", "purple CL"),
)
RIGHT_PAIRS = (
    ("red CS", "blue CS"),
    ("green CS", "orange CS"),
    ("purple CS", "purple PS"),
    ("red CL", "blue CL"),
    ("yellow CL", "orange CL"),
    ("purple CL", "red PL"),
    ("red PS", "green PS"),
    ("yellow PS", "orange PS"),
    ("blue PL", "green PL"),
    ("yellow PL", "orange PL"),
)


def _inventory(targets: list[str], rng: random.Random) -> list[str]:
    inventory = list(dict.fromkeys(targets))
    groups = [[target] for target in targets] if len(targets) < 3 else [targets]
    for group in groups:
        for predicate in (
            lambda other, group=group: any(
                other.color == piece(t).color and other.scale != piece(t).scale for t in group
            ),
            lambda other, group=group: any(
                other.scale == piece(t).scale and other.color != piece(t).color for t in group
            ),
        ):
            if any(predicate(piece(p)) for p in inventory if p not in group):
                continue
            options = [p for p in PIECES if p not in inventory and predicate(piece(p))]
            inventory.append(rng.choice(options))
    while len(inventory) < 8:
        inventory.append(rng.choice([p for p in PIECES if p not in inventory]))
    return inventory


def _scene(inventory: list[str], seed: int, profile: str, *, single=False, left=True) -> dict:
    rng = random.Random(seed)
    slots = [(x, y) for x in (0.43, 0.52) for y in (-0.44, -0.15, 0.15, 0.44)]
    rng.shuffle(slots)
    requirements = (
        {0: left, 1: left} if single else {0: left, 1: not left, 2: left, 3: not left, 4: not left}
    )
    for i, positive in requirements.items():
        if (slots[i][1] > 0) != positive:
            j = next(
                j
                for j in range(8)
                if j not in requirements or j > i
                if (slots[j][1] > 0) == positive
            )
            slots[i], slots[j] = slots[j], slots[i]
    placements: list[dict] = []
    for name, (x, y) in zip(inventory, slots, strict=True):
        # Baseline repetitions still differ by centimeters, not pixel noise.
        distance = rng.uniform(0.02, 0.04) if profile != "layout" else rng.uniform(0.05, 0.09)
        y += (-1 if abs(y) > 0.3 else 1) * (1 if y > 0 else -1) * distance
        x += rng.uniform(-0.005, 0.005)
        placements.append(
            {
                "piece": name,
                "xy_m": [round(x, 5), round(y, 5)],
                "yaw_deg": rng.choice((-35, -20, 20, 30)) if profile == "layout" else 0,
            }
        )
    x0, x1, y0, y1 = BUILD_REGION
    candidates = [
        (rng.uniform(x0 + 0.01, x1 - 0.01), rng.uniform(y0 + 0.01, y1 - 0.01)) for _ in range(512)
    ]
    site = next(
        xy
        for xy in candidates
        if all(
            math.dist(xy, p["xy_m"]) > math.hypot(*piece(p["piece"]).dimensions_m[:2]) / 2 + 0.065
            for p in placements
        )
    )
    mass, friction, appearance = 1.0, (0.6, 0.5), "light_wood"
    if profile == "physics_appearance":
        axis = seed % 3
        if axis == 0:
            mass = rng.choice((0.7, 1.3))
        elif axis == 1:
            friction = rng.choice(((0.45, 0.35), (0.8, 0.65)))
        else:
            appearance = "neutral_side"
    return {
        "placements": placements,
        "assembly_xy_m": site,
        "mass_scale": mass,
        "friction": friction,
        "appearance": appearance,
    }


def _item(
    family: str,
    category: str,
    partition: str,
    targets: list[str],
    scene: dict,
    repetition: int,
    variant: int = 0,
    **extra,
) -> dict:
    instruction = FORMS[(repetition + variant) % 3].format(
        ", then ".join(piece(p).description.replace("prism", "rectangular prism") for p in targets)
    )
    if category == "single_block":
        instruction = (
            "Pick up {} and place it in a clear central area of the table, then release it."
            if repetition % 2 == 0
            else "Choose {}, carry it to a clear central place on the table and let go."
        ).format("the " + piece(targets[0]).description.replace("prism", "rectangular prism"))
    row = {
        "case": f"{family}-{repetition:02d}-{variant}",
        "family": family,
        "partition": partition,
        "split": "train" if partition == "train" else "development",
        "category": category,
        "repetition": str(repetition),
        "variant": str(variant),
        "cooperative_placement": "4",
        "error_target_level": str(1 + repetition % 5),
        "variation": extra.pop("profile", "baseline"),
    }
    if category == "recovery" and extra.get("mechanism") == "placement":
        row["error_target_level"] = str(2 + repetition % 4)
    if category == "recovery" and extra.get("mechanism") == "interrupted_progress":
        row["error_target_level"] = str(1 + repetition % 4)
    row.update(extra)
    return dict(row=row, targets=targets, instruction=instruction, **scene)


def build_declaration() -> dict:
    cases: list[dict] = []
    for f in range(20):
        left = f % 2 == 0
        pair = (LEFT_PAIRS if left else RIGHT_PAIRS)[f // 2]
        inventory = _inventory(list(pair), random.Random(81000 + f))
        for repetition in range(4):
            k = f * 4 + repetition
            profile = ("baseline", "baseline", "baseline", "layout", "physics_appearance")[k % 5]
            scene = _scene(inventory, 90000 + k, profile, single=True, left=left)
            cases.extend(
                _item(
                    f"TR-P{f+1:02d}",
                    "single_block",
                    "train",
                    [target],
                    scene,
                    repetition,
                    v,
                    profile=profile,
                )
                for v, target in enumerate(pair)
            )
    for f in range(24):
        targets = random.Random(72000 + f).sample(list(PIECES), 5)
        alternate = list(targets)
        a, b = (0, 1) if f < 12 else (3, 4)
        alternate[a], alternate[b] = alternate[b], alternate[a]
        inventory = _inventory(targets, random.Random(73000 + f))
        for repetition in range(5):
            profile = ("baseline", "baseline", "baseline", "layout", "physics_appearance")[
                repetition
            ]
            scene = _scene(inventory, 100000 + f * 5 + repetition, profile, left=f % 2 == 0)
            cases.extend(
                _item(
                    f"TR-T{f+1:02d}", "tower", "train", order, scene, repetition, v, profile=profile
                )
                for v, order in enumerate((targets, alternate))
            )
    _add_recoveries_and_stops(cases)
    _add_development(cases)
    initial = _initial_batch(cases)
    return {
        "collection_condition": CONDITION,
        "seed": 20261026,
        "cases": cases,
        "initial_cases": initial,
        "training_episode_limit": 500,
        "development_calibration": 60,
        "development_confirmation": 60,
    }


def _add_recoveries_and_stops(cases: list[dict]) -> None:
    for mechanism in RECOVERIES:
        for j in range(15):
            family = f"TR-R-{mechanism}-{j//3+1:02d}"
            seed = 200000 + RECOVERIES.index(mechanism) * 100 + j
            targets = random.Random(seed).sample(list(PIECES), 5)
            scene = _scene(
                _inventory(targets, random.Random(seed)), seed, "baseline", left=j % 2 == 0
            )
            cases.append(
                _item(
                    family,
                    "recovery",
                    "train",
                    targets,
                    scene,
                    j,
                    mechanism=mechanism,
                    correction_origin=("policy" if j % 5 in (1, 2) else "teacher"),
                )
            )
    reasons = (
        ["missing_color"] * 5 + ["missing_size"] * 5 + ["missing_shape"] * 5 + ["unreachable"] * 10
    )
    for j, reason in enumerate(reasons):
        cases.append(_stop_case(f"TR-S{j+1:02d}", "train", reason, 300000 + j, j))


def _stop_case(family: str, partition: str, reason: str, seed: int, j: int) -> dict:
    rng = random.Random(seed)
    level = (0, 2, 4)[j % 3]
    contrast = None
    if reason == "unreachable":
        targets = rng.sample(list(PIECES), 5)
    else:
        attribute = reason.removeprefix("missing_")
        attribute = "scale" if attribute == "size" else attribute
        pairs = [
            (a, b)
            for a in PIECES
            for b in PIECES
            if a != b
            and getattr(piece(a), attribute) != getattr(piece(b), attribute)
            and all(
                getattr(piece(a), k) == getattr(piece(b), k)
                for k in ("color", "shape", "scale")
                if k != attribute
            )
        ]
        selected, contrast = rng.choice(pairs)
        targets = rng.sample([p for p in PIECES if p not in (selected, contrast)], 4)
        targets.insert(level, selected)
    scene = _scene(_inventory(targets, rng), seed, "baseline", left=j % 2 == 0)
    selected = targets[level]
    placement = next(p for p in scene["placements"] if p["piece"] == selected)
    if reason == "unreachable":
        placement["xy_m"] = [0.84, 0.40 if j % 2 else -0.40]
    else:
        inventory = [p["piece"] for p in scene["placements"]]
        if contrast in inventory:
            placement["piece"] = rng.choice([p for p in PIECES if p not in inventory])
        else:
            placement["piece"] = contrast
    return _item(
        family,
        "stop",
        partition,
        targets,
        scene,
        j,
        stop_reason=reason,
        error_target_level=str(level + 1),
    )


def _add_development(cases: list[dict]) -> None:
    for offset, partition in enumerate(("calibration", "confirmation")):
        prefix = "DC" if partition == "calibration" else "DF"
        for f in range(5):
            targets = [PIECES[(3 * f + offset * 7) % 20]]
            inventory = _inventory(targets, random.Random(400000 + offset * 100 + f))
            for j in range(3):
                seed = 410000 + offset * 1000 + f * 10 + j
                profile = ("baseline", "layout", "physics_appearance")[j]
                scene = _scene(inventory, seed, profile, single=True, left=(f + j) % 2 == 0)
                cases.append(
                    _item(
                        f"{prefix}-P{f+1:02d}",
                        "single_block",
                        partition,
                        targets,
                        scene,
                        j,
                        profile=profile,
                    )
                )
        for f in range(5):
            seed = 420000 + offset * 1000 + f * 10
            targets = random.Random(seed).sample(list(PIECES), 5)
            inventory = _inventory(targets, random.Random(seed))
            for j in range(5):
                profile = ("baseline", "baseline", "baseline", "layout", "physics_appearance")[j]
                scene = _scene(inventory, seed + j, profile, left=f % 2 == 0)
                cases.append(
                    _item(
                        f"{prefix}-T{f+1:02d}",
                        "tower",
                        partition,
                        targets,
                        scene,
                        j,
                        profile=profile,
                    )
                )
        for i, mechanism in enumerate(RECOVERIES):
            for j in range(2):
                seed = 440000 + offset * 1000 + i * 10 + j
                targets = random.Random(seed).sample(list(PIECES), 5)
                scene = _scene(
                    _inventory(targets, random.Random(seed)), seed, "baseline", left=j == 0
                )
                cases.append(
                    _item(
                        f"{prefix}-R{i+1:02d}",
                        "recovery",
                        partition,
                        targets,
                        scene,
                        j,
                        mechanism=mechanism,
                        correction_origin="teacher",
                    )
                )
        reasons = ["missing_color", "missing_size", "missing_shape"] * 2 + ["unreachable"] * 4
        for j, reason in enumerate(reasons):
            cases.append(
                _stop_case(f"{prefix}-S{j+1:02d}", partition, reason, 460000 + offset * 1000 + j, j)
            )


def _initial_batch(cases: list[dict]) -> list[str]:
    selected = []
    for item in cases:
        row = item["row"]
        category, family, repetition = row["category"], row["family"], int(row["repetition"])
        include = row["partition"] == "train" and (
            (
                category == "single_block"
                and family in ("TR-P01", "TR-P02", "TR-P03", "TR-P04")
                and repetition == 0
            )
            or (
                category == "tower"
                and family in ("TR-T01", "TR-T02", "TR-T13", "TR-T14")
                and repetition == 0
            )
            or (category == "recovery" and repetition == 0)
            or (category == "stop" and family in ("TR-S01", "TR-S06", "TR-S16"))
        )
        if row["partition"] == "calibration" and repetition == 0:
            include = family in (
                "DC-P01",
                "DC-P02",
                "DC-T01",
                "DC-T02",
                "DC-R01",
                "DC-R02",
                "DC-S01",
            )
        if family == "DC-S07":
            include = True
        if include:
            selected.append(row["case"])
    if len(selected) != 32:
        raise ValueError("Initial collection must contain exactly24TRAIN and8development cases")
    return selected


def write_declaration(path: Path) -> dict:
    if path.exists():
        raise FileExistsError("Preserve the existing campaign declaration")
    declaration = build_declaration()
    path.write_text(json.dumps(declaration, indent=2) + "\n")
    load_campaign(path)
    return declaration
