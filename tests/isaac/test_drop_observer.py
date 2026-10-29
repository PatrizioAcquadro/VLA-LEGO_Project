"""Strict drop handoff waits through invalid intermediate states without admitting them."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from vla_lego.isaac.environment import TABLE_TOP
from vla_lego.isaac.policy_continuation import PolicyIntervention
from vla_lego.isaac.task import initial_tower_task


@pytest.mark.parametrize("invalid", ["contact", "unsupported", "nonfinite"])
@pytest.mark.parametrize("eventually_released", [True, False])
def test_transient_invalid_state_waits_and_final_invalid_state_is_rejected(
    invalid, eventually_released
):
    task = initial_tower_task()
    after = [
        {
            "block_pose_w": [
                *p.xy_m,
                TABLE_TOP + p.spec.dimensions_m[2] / 2,
                0,
                0,
                0,
                1,
            ],
            "block_velocity_w": [0.0] * 6,
            "robot_contact_force_n": 0.0,
            "nonrobot_contact_force_n": 0.3,
            "block_contact_forces_n": [[0.0] * 3 for _ in task.placements],
        }
        for p in task.placements
    ]
    after[0]["block_pose_w"][:2] = list(task.assembly_xy_m)
    before = deepcopy(after)
    before[1]["block_pose_w"][2] += 0.1
    before[1]["robot_contact_force_n"] = 1.0
    invalid_state = deepcopy(after)
    if invalid == "contact":
        invalid_state[2]["robot_contact_force_n"] = 0.128
    elif invalid == "unsupported":
        invalid_state[2]["nonrobot_contact_force_n"] = 0.0
    else:
        invalid_state[2]["block_velocity_w"][0] = float("nan")
    admitted = []
    env = SimpleNamespace(
        task_truth=lambda i: before[i],
        time=10,
        recorder=SimpleNamespace(admit_policy_intervention=admitted.append),
    )
    case = SimpleNamespace(
        row={
            "correction_origin": "policy",
            "mechanism": "slip_drop",
            "partition": "train",
        },
        error_level=1,
        cooperative_level=3,
    )
    observed = []
    intervention = PolicyIntervention.__new__(PolicyIntervention)

    def window(*args, stop_when, **kwargs):
        # Two released samples cannot carry across the invalid intermediate state.
        observed.extend(stop_when(after) for _ in range(2))
        observed.append(stop_when(invalid_state))
        final = after if eventually_released else invalid_state
        observed.extend(stop_when(final) for _ in range(3))
        return {"executed_motor_commands": 6}, before, final, False

    intervention._window = window
    if eventually_released:
        event = intervention.slip_drop(SimpleNamespace(env=env), case, task, 1)
        assert event["physically_verified"] and len(admitted) == 1
        assert observed[:5] == [None] * 5
        assert observed[-1]["error_basis"] == "actual_released_fall"
    else:
        with pytest.raises(RuntimeError, match="did not produce"):
            intervention.slip_drop(SimpleNamespace(env=env), case, task, 1)
        assert observed == [None] * 6 and not admitted
