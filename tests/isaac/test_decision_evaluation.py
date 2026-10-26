"""A premature terminal cannot be erased by later correct frames or majority accuracy."""

import numpy as np
import pytest

from vla_lego.pretrained.decision_evaluation import (
    calibrated_probabilities,
    choose_thresholds,
    evaluate_sequences,
    probability_metrics,
    sequence_outcome,
)

THRESHOLDS = {"complete_threshold": 0.9, "abstain_threshold": 0.9, "persistence_s": 0.5}


def sequence(labels, kind="recorded_instruction"):
    return {
        "case": "case",
        "kind": kind,
        "labels": np.asarray(labels),
        "times": np.arange(len(labels)) / 2,
        "histories": {i: [] for i in range(len(labels))},
    }


def test_first_false_completion_is_failure_even_if_later_completion_is_correct():
    source = sequence([0, 0, 0, 1, 1])
    probabilities = np.asarray([[0.01, 0.98, 0.01]] * 5)
    result = sequence_outcome(source, probabilities, THRESHOLDS)
    assert result["false_completion"] and not result["correct_terminal"]
    assert result["terminal_time_s"] == 0.5
    assert result["delay_s"] is None


def test_calibration_requires_recall_and_complete_natural_and_negative_denominators():
    sources = [sequence([0, 0, 1, 1]) for _ in range(50)]
    sources += [sequence([2, 2, 2, 2]) for _ in range(10)]
    sources += [sequence([0, 0, 0, 0], "counterfactual_instruction") for _ in range(50)]
    values = [np.eye(3)[row["labels"]] for row in sources]
    assert evaluate_sequences(sources, values, THRESHOLDS)["passed"]
    missing_negatives = evaluate_sequences(sources[:60], values[:60], THRESHOLDS)
    assert not missing_negatives["passed"]
    never_stop = [np.asarray([[1, 0, 0]] * 4) for _ in sources]
    report = evaluate_sequences(sources, never_stop, THRESHOLDS)
    assert not report["passed"] and report["summary"]["true_completions"] == 0
    _, selected = choose_thresholds(sources, values)
    assert selected["passed"]


def test_uncalibrated_logits_do_not_bypass_probability_and_delay_reporting():
    logits = np.asarray([[1, 3, 2], [4, 0, 0]], dtype=float)
    calibration = {"temperature": 2, "bias": [0.1, -0.2, 0.1]}
    probabilities = calibrated_probabilities(logits, calibration)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    metrics = probability_metrics(np.eye(3)[[0, 1, 2]], [0, 1, 2])
    assert metrics["nll"] == pytest.approx(0)
    assert metrics["brier"] == 0 and metrics["ece_10_bins"] == 0
    source = sequence([0, 0, 1, 1])
    result = sequence_outcome(source, np.eye(3)[source["labels"]], THRESHOLDS)
    assert result["correct_terminal"] and result["delay_s"] == 0.5
