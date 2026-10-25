"""The global learning schedule and stopping decisions survive allocation boundaries."""

import pytest

from vla_lego.pretrained.pilot_campaign import (
    better_candidate,
    learning_rate_factor,
    persistent_deterioration,
    stopping_reason,
)


def history(*, train_slope=0.0, development_slope=0.0, decision_gain=0.0):
    return [
        {
            "step": step,
            "train": {"motor_mae_normalized": 1 - train_slope * step / 30_000},
            "development": {
                "motor_mae_normalized": 1 - development_slope * step / 30_000,
                "jaw_mae_m": 0.002,
                "decision_correct": 0.5 + decision_gain * step / 30_000,
            },
        }
        for step in range(0, 30_001, 1_000)
    ]


def test_global_schedule_has_one_warmup_and_finite_decay_floor():
    assert learning_rate_factor(0) == 0.001
    assert learning_rate_factor(999) == 1.0
    assert learning_rate_factor(1_000) == 1.0
    assert learning_rate_factor(30_000) > learning_rate_factor(60_000)
    assert learning_rate_factor(100_000) == pytest.approx(0.025)


def test_flat_early_diagnostics_do_not_end_substantial_base_training():
    assert stopping_reason(history()[:-1]) is None


def test_long_flat_development_does_not_get_hidden_by_falling_train_loss():
    assert stopping_reason(history(train_slope=0.5)) == "generalization_stagnation"
    assert stopping_reason(history()) == "optimization_stagnation"


def test_credible_motor_or_decision_progress_keeps_training():
    assert stopping_reason(history(development_slope=0.5)) is None
    assert stopping_reason(history(decision_gain=0.2)) is None


def test_candidate_uses_motor_progress_and_decisions_within_a_narrow_motor_band():
    previous = {"motor_mae_normalized": 1.0, "decision_correct": 0.5}
    assert better_candidate({"motor_mae_normalized": 0.95, "decision_correct": 0.5}, previous)
    assert better_candidate({"motor_mae_normalized": 1.01, "decision_correct": 0.6}, previous)
    assert not better_candidate({"motor_mae_normalized": 1.2, "decision_correct": 1.0}, previous)


def test_motor_improvement_cannot_hide_decision_or_rare_class_collapse():
    previous = {
        "motor_mae_normalized": 1.0,
        "decision_correct": 0.8,
        "decision_accuracy_by_class": {"continue": 0.8, "complete": 0.8, "abstain": 0.8},
    }
    assert not better_candidate({"motor_mae_normalized": 0.1, "decision_correct": 0.5}, previous)
    assert not better_candidate(
        {
            "motor_mae_normalized": 0.1,
            "decision_correct": 0.8,
            "decision_accuracy_by_class": {"continue": 1.0, "complete": 1.0, "abstain": 0.4},
        },
        previous,
    )


def test_large_persistent_held_out_regression_is_distinct_from_a_short_plateau():
    rows = history(train_slope=0.7)
    for row in rows[-6:]:
        row["development"]["motor_mae_normalized"] = 1.3
    assert persistent_deterioration(rows)
    assert not persistent_deterioration(rows[:-1])
    assert not persistent_deterioration(history())
