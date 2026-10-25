import copy
import json

import pytest

from vla_lego.isaac.pilot import load_paired_cases


@pytest.fixture
def paired_data():
    inventory = [
        "red CS",
        "red CL",
        "blue PL",
        "green CS",
        "yellow PL",
        "purple CS",
        "orange PS",
        "green PL",
    ]
    positions = [(x, y) for x in (0.42, 0.51) for y in (-0.42, -0.14, 0.14, 0.42)]
    cases = []
    for variant, first in enumerate(inventory[:2]):
        targets = [first, "blue PL", "green CS", "yellow PL", "orange PS"]
        cases.append(
            {
                "row": {
                    "case": f"T6-G{variant}",
                    "family": "T6",
                    "split": "train",
                    "category": "nominal",
                    "order": " > ".join(targets),
                    "distractors": "; ".join(v for v in inventory if v not in targets),
                    "cooperative_placement": "3",
                    "variation": "nominal spread",
                },
                "placements": [
                    {"piece": v, "xy_m": list(xy)}
                    for v, xy in zip(inventory, positions, strict=True)
                ],
                "assembly_xy_m": [0.36, 0.0],
            }
        )
    return {"collection_condition": "marker_free_paired_instruction_v1", "cases": cases}


def test_paired_cases_keep_scene_fixed_and_change_first_request(tmp_path, paired_data):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(paired_data))
    (first, a), (second, b) = load_paired_cases(path)
    assert first.row["family"] == second.row["family"] == "T6"
    assert a.placements == b.placements
    assert a.assembly_xy_m == b.assembly_xy_m
    assert a.targets[0] != b.targets[0] and a.targets[1:] == b.targets[1:]
    assert a.instruction != b.instruction and len(a.targets) == 5


@pytest.mark.parametrize("defect", ["development", "layout", "physics", "same_goal", "unpaired"])
def test_paired_cases_reject_broken_experiment(tmp_path, paired_data, defect):
    data = copy.deepcopy(paired_data)
    if defect == "development":
        data["cases"][0]["row"]["split"] = "development"
    elif defect == "layout":
        data["cases"][1]["placements"][0]["xy_m"][0] += 0.03
    elif defect == "physics":
        data["cases"][1]["row"]["variation"] = "mass factor 1.2"
    elif defect == "same_goal":
        data["cases"][1]["row"]["order"] = data["cases"][0]["row"]["order"]
        data["cases"][1]["row"]["distractors"] = data["cases"][0]["row"]["distractors"]
    else:
        data["cases"].pop()
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_paired_cases(path)
