"""Render recorded motor-only or historical joint motor/decision learning curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history = json.loads((args.run / "history.json").read_text())
    state = json.loads((args.run / "status.json").read_text())
    if not history:
        parser.error("History contains no diagnostics; no curves can be reported")
    metrics = {
        "motor_flow": ("Fixed motor flow loss", 1),
        "decision_flow": ("Fixed decision flow loss", 1),
        "motor_mae_normalized": ("Normalized motor MAE", 1),
        "executed_motor_mae_normalized": ("Executed-prefix normalized motor MAE", 1),
        "hold_executed_motor_mae_normalized": ("Hold baseline: executed-prefix MAE", 1),
        "arm_mae_rad": ("Decoded arm MAE (rad)", 1),
        "jaw_mae_m": ("Decoded jaw MAE (mm)", 1000),
        "decision_correct": ("Balanced decision accuracy", 1),
    }
    metrics = {
        key: spec
        for key, spec in metrics.items()
        if any(
            row[split].get(key) is not None for row in history for split in ("train", "development")
        )
    }
    if not metrics:
        parser.error("History contains no supported offline metrics")
    rows = [
        {"step": record["step"]}
        | {
            f"{split}_{key}": record[split][key]
            for split in ("train", "development")
            for key in metrics
            if record[split].get(key) is not None
        }
        for record in history
    ]
    progress = []
    path = args.run / "progress.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            value = json.loads(line)
            progress.append((value["step"], value["loss"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "offline-curves.csv").open("w", newline="") as stream:
        columns = ["step", *dict.fromkeys(key for row in rows for key in row if key != "step")]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    panels = len(metrics) + bool(progress)
    figure, axes = plt.subplots(
        math.ceil(panels / 3),
        3,
        figsize=(15, 4 * math.ceil(panels / 3)),
        constrained_layout=True,
        squeeze=False,
    )
    available = iter(axes.flat)
    if progress:
        axis = next(available)
        x, loss = np.asarray(progress).T
        axis.plot(x, loss, alpha=0.2, color="tab:blue", linewidth=0.6)
        window = min(100, len(loss))
        smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
        axis.plot(x[window - 1 :], smooth, color="tab:blue", label="rolling train objective")
        if np.all(loss > 0):
            axis.set_yscale("log")
        axis.set_title("Sampled train flow objective")
    steps = [row["step"] for row in history]
    for key, (title, factor) in metrics.items():
        axis = next(available)
        for split, color in (("train", "tab:blue"), ("development", "tab:orange")):
            values = [row[split].get(key) for row in history]
            if any(value is not None for value in values):
                axis.plot(
                    steps,
                    [np.nan if value is None else value * factor for value in values],
                    label=split,
                    color=color,
                )
        axis.set_title(title)
        if key == "decision_correct":
            axis.set_ylim(0, 1.02)
    for axis in axes.flat[:panels]:
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in available:
        axis.set_visible(False)
    figure.suptitle("Alex003 SmolVLA — recorded offline learning diagnostics")
    figure.savefig(args.output / "learning-curves.png", dpi=150)
    plt.close(figure)
    text = [
        "# Alex003 offline learning diagnostics",
        "",
        f"Status: `{state['status']}`.",
        f"Optimizer updates: **{state['step']}**. Offline shortlist checkpoint: **{state.get('best_step')}**.",
        "",
        "![Offline learning curves](learning-curves.png)",
        "",
        "Metrics and split names retain their recorded meanings. Missing metrics are omitted from plots "
        "and left blank in CSV. Fixed diagnostics macro-average episodes and phases; they do not establish "
        "autonomous task success or select a physically qualified policy.",
        "",
        "Physical evaluations and decision calibration are reported separately with their original denominators.",
    ]
    (args.output / "training-report.md").write_text("\n".join(text) + "\n")
    print(
        json.dumps(
            {
                "history_points": len(history),
                "progress_rows": len(progress),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
