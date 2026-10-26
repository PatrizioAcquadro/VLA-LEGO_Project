"""Natural-sequence calibration and first-terminal outcomes for the temporal head."""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from vla_lego.pretrained.task_decisions import DecisionFilter


def calibrated_probabilities(logits, calibration):
    values = np.asarray(logits, dtype=np.float64) / calibration["temperature"]
    values += np.asarray(calibration["bias"])
    values -= values.max(axis=-1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def fit_calibration(logits, labels):
    """Fit temperature/class offsets on natural calibration rows, never balanced draws."""
    import torch
    from torch.nn import functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("Decision calibration requires native CUDA")
    values = torch.as_tensor(logits, dtype=torch.float32, device="cuda")
    targets = torch.as_tensor(labels, dtype=torch.long, device="cuda")
    temperature = torch.zeros((), device="cuda", requires_grad=True)
    bias = torch.zeros(3, device="cuda", requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(
            values / temperature.clamp(-3, 3).exp() + bias - bias.mean(), targets
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    result = {
        "temperature": float(temperature.detach().clamp(-3, 3).exp()),
        "bias": (bias.detach() - bias.detach().mean()).cpu().tolist(),
        "fit": "all natural recorded calibration observations; no balanced or confirmation rows",
    }
    if not math.isfinite(result["temperature"]) or not np.isfinite(result["bias"]).all():
        raise FloatingPointError("Non-finite probability calibration")
    return result


def probability_metrics(probabilities, labels):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    ece = 0.0
    for bin_index in range(10):
        start = bin_index / 10
        selected = (confidence >= start) & (
            (confidence < (bin_index + 1) / 10) if bin_index < 9 else (confidence <= 1)
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float((predicted[selected] == labels[selected]).mean())
                - float(confidence[selected].mean())
            )
    return {
        "observations": len(labels),
        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-12)).mean()),
        "brier": float(((probabilities - np.eye(3)[labels]) ** 2).sum(axis=1).mean()),
        "ece_10_bins": ece,
    }


def sequence_outcome(sequence, probabilities, thresholds):
    """The first terminal call ends the episode; later correct calls cannot erase it."""
    decision = DecisionFilter(**thresholds)
    first = None
    for index in sequence["histories"]:
        value = decision.update(float(sequence["times"][index]), probabilities[index])
        if value != "continue":
            first = (index, 1 if value == "complete" else 2)
            break
    labels = np.asarray(sequence["labels"])
    expected = int(labels[-1])
    index, predicted = first if first is not None else (None, 0)
    correct = bool(index is not None and predicted == expected and labels[index] == expected)
    delay = None
    if correct and expected:
        onset = int(np.flatnonzero(labels == expected)[0])
        delay = float(sequence["times"][index] - sequence["times"][onset])
    return {
        "case": sequence["case"],
        "kind": sequence["kind"],
        "expected": expected,
        "first_terminal": predicted,
        "terminal_time_s": None if index is None else float(sequence["times"][index]),
        "correct_terminal": correct,
        "false_completion": bool(predicted == 1 and labels[index] != 1),
        "false_abstention": bool(predicted == 2 and labels[index] != 2),
        "delay_s": delay,
    }


def evaluate_sequences(sequences, probabilities, thresholds):
    outcomes = [
        sequence_outcome(sequence, values, thresholds)
        for sequence, values in zip(sequences, probabilities, strict=True)
    ]
    recorded = [row for row in outcomes if row["kind"] == "recorded_instruction"]
    feasible = [row for row in recorded if row["expected"] == 1]
    impossible = [row for row in recorded if row["expected"] == 2]
    negatives = [row for row in outcomes if row["kind"] == "counterfactual_instruction"]
    summary = {
        "feasible": len(feasible),
        "impossible": len(impossible),
        "true_completions": sum(row["correct_terminal"] for row in feasible),
        "false_completions": sum(row["false_completion"] for row in recorded),
        "false_abstentions": sum(row["false_abstention"] for row in feasible),
        "true_abstentions": sum(row["correct_terminal"] for row in impossible),
        "counterfactuals": len(negatives),
        "counterfactual_terminals": sum(bool(row["first_terminal"]) for row in negatives),
    }
    passed = (
        summary["feasible"] == 50
        and summary["impossible"] == 10
        and summary["false_completions"] == 0
        and summary["true_completions"] >= 45
        and summary["false_abstentions"] <= 2
        and summary["true_abstentions"] >= 9
        and summary["counterfactuals"] == 50
        and summary["counterfactual_terminals"] == 0
    )
    return {"summary": summary, "passed": passed, "sequences": outcomes}


def selection_rank(report):
    summary = report["summary"]
    premature = summary["false_completions"] + summary["counterfactual_terminals"]
    shortfall = max(0, 45 - summary["true_completions"]) + max(0, 9 - summary["true_abstentions"])
    return (
        premature + max(0, summary["false_abstentions"] - 2) + shortfall,
        premature,
        summary["false_abstentions"],
        -summary["true_completions"] - summary["true_abstentions"],
    )


def choose_thresholds(sequences, probabilities):
    candidates = []
    for complete, abstain in product((0.5, 0.75, 0.9, 0.97, 0.995), repeat=2):
        thresholds = {
            "complete_threshold": complete,
            "abstain_threshold": abstain,
            "persistence_s": 0.5,
        }
        report = evaluate_sequences(sequences, probabilities, thresholds)
        candidates.append((selection_rank(report), complete + abstain, thresholds, report))
    _, _, thresholds, report = min(candidates, key=lambda row: row[:2])
    return thresholds, report
