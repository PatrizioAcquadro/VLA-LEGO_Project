"""Reject phases where a released first base cannot establish a drop error."""

from types import SimpleNamespace

import pytest

from vla_lego.isaac.policy_continuation import PolicyIntervention


@pytest.mark.parametrize("level", [-1, 0, 3, 4])
def test_drop_phase_is_rejected_before_truth_or_policy_commands(level):
    case = SimpleNamespace(
        row={
            "correction_origin": "policy",
            "mechanism": "slip_drop",
            "partition": "train",
        },
        error_level=level,
        cooperative_level=3,
    )
    intervention = PolicyIntervention.__new__(PolicyIntervention)
    # Missing task_truth/_window guarantees refusal precedes truth reads and commands.
    reference = SimpleNamespace(env=SimpleNamespace(recorder=None))
    with pytest.raises(ValueError, match="existing prefix"):
        intervention.slip_drop(reference, case, None, 0)
