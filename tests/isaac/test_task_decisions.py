"""Causality and terminal behavior contracts, independent of a trained success claim."""

import pytest

from vla_lego.pretrained.task_decisions import DecisionFilter, causal_history


def test_history_never_reads_future_and_pads_only_from_actual_start():
    times = [i * 0.05 for i in range(100)]
    assert causal_history(times, 10) == [0, 0, 0, 0, 0, 0, 10]
    selected = causal_history(times, 80)
    assert len(selected) == 7 and selected[-1] == 80 and max(selected) <= 80
    assert times[selected[0]] <= times[80] - 3


def test_uncertain_or_isolated_terminal_predictions_do_not_stop_execution():
    decision = DecisionFilter(0.9, 0.9)
    assert decision.update(0, [0.1, 0.8, 0.1]) == "continue"
    assert decision.update(0.5, [0.01, 0.98, 0.01]) == "continue"
    assert decision.update(1, [0.95, 0.04, 0.01]) == "continue"
    assert decision.update(1.5, [0.01, 0.01, 0.98]) == "continue"
    assert decision.update(2, [0.01, 0.01, 0.98]) == "abstain"
    with pytest.raises(ValueError, match="advance"):
        decision.update(2, [0.01, 0.01, 0.98])
