"""Task decisions predicted alongside motor targets, never supplied as observations."""

import math

DECISION_NAMES = ("TASK_COMPLETE", "TASK_ABSTAIN")
DECISION_TARGETS = {"continue": (0.0, 0.0), "complete": (1.0, 0.0), "abstain": (0.0, 1.0)}


def decode_decision(values) -> str:
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected two finite learned task-decision values.")
    complete, abstain = (v >= 0.5 for v in values)
    if complete and abstain:
        return "conflict"
    return "complete" if complete else "abstain" if abstain else "continue"
