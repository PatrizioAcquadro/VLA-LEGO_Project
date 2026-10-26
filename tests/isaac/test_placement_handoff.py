"""Guard teacher-only continuation from an actual placed first piece."""

from copy import deepcopy
from dataclasses import replace

import pytest

from vla_lego.isaac.environment import TABLE_TOP
from vla_lego.isaac.policy_replay import preserved_first_placement
from vla_lego.isaac.task import initial_tower_task

PATHS = [
    f"/Robot/{side}_WSG32_{jaw}_JAW_LINK"
    for side in ("left", "right")
    for jaw in ("NEGATIVE", "POSITIVE")
]


def prefix():
    task = replace(initial_tower_task(), build_region_xy_m=(0.30, 0.45, -0.13, 0.13))
    blocks = [
        {
            "block_pose_w": [*p.xy_m, TABLE_TOP + p.spec.dimensions_m[2] / 2, 0, 0, 0, 1],
            "block_velocity_w": [0.0] * 6,
            "robot_contact_force_n": 0.0,
            "robot_contact_forces_n": [[0.0] * 3 for _ in PATHS],
            "block_contact_forces_n": [[0.0] * 3 for _ in task.placements],
            "nonrobot_contact_force_n": p.spec.mass_kg * 9.81,
        }
        for p in task.placements
    ]
    rows = []
    for tick in range(43):
        current = deepcopy(blocks)
        if tick == 1:
            current[0]["block_pose_w"][2] += 0.12
            current[0]["robot_contact_forces_n"][:2] = [[0.4, 0, 0], [-0.4, 0, 0]]
            current[0]["robot_contact_force_n"] = 0.8
            current[0]["nonrobot_contact_force_n"] = 0.0
        elif tick >= 2:
            current[0]["block_pose_w"][:2] = [0.36, 0.03]
        rows.append(
            {
                "tick": tick,
                "time_s": tick * 0.05,
                "decision": "continue",
                "robot": {"robot_contact_paths": PATHS},
                "blocks": current,
            }
        )
    return task, rows


def test_preserves_valid_first_placement_without_changing_scene_or_instruction():
    task, rows = prefix()
    original = deepcopy(rows)
    result = preserved_first_placement(task, rows)
    assert result["next_target_level"] == 1
    assert result["preserved_target_levels"] == [0]
    assert result["released_dwell_s"] == pytest.approx(2.0)
    assert not result["autonomous_full_task_success"]
    assert rows == original and len(task.targets) == 5


@pytest.mark.parametrize(
    "defect", ["loose", "held", "unsupported", "drift", "site", "short", "nonfinite"]
)
def test_rejects_incomplete_or_invalid_first_placement(defect):
    task, rows = prefix()
    if defect == "loose":
        rows[1]["blocks"] = deepcopy(rows[0]["blocks"])
    elif defect == "held":
        rows[-1]["blocks"][0]["robot_contact_force_n"] = 0.1
    elif defect == "unsupported":
        rows[-1]["blocks"][0]["nonrobot_contact_force_n"] = 0.0
    elif defect == "drift":
        rows[-1]["blocks"][0]["block_pose_w"][0] += 0.004
    elif defect == "site":
        for row in rows[2:]:
            row["blocks"][0]["block_pose_w"][0] = 0.48
    elif defect == "nonfinite":
        rows[-1]["blocks"][1]["block_pose_w"][0] = float("nan")
    else:
        rows.pop()
    with pytest.raises(ValueError, match="First piece lacks"):
        preserved_first_placement(task, rows)


def test_rejects_terminal_or_noncontiguous_prefix():
    task, rows = prefix()
    rows[-1]["decision"] = "complete"
    with pytest.raises(ValueError, match="applied contiguous"):
        preserved_first_placement(task, rows)
