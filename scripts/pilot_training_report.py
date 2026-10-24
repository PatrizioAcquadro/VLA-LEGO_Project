"""Render comparable offline learning curves from the actual campaign history."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    history = json.loads((args.run / "history.json").read_text())
    state = json.loads((args.run / "status.json").read_text())
    rows = []
    for record in history:
        row = {"step": record["step"]}
        for split in ("train", "development"):
            for key in (
                "motor_flow",
                "decision_flow",
                "motor_mae_normalized",
                "arm_mae_rad",
                "jaw_mae_m",
                "decision_correct",
            ):
                row[f"{split}_{key}"] = record[split][key]
        rows.append(row)
    with (args.output / "offline-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    progress = []
    path = args.run / "progress.jsonl"
    if path.exists():
        with path.open() as stream:
            for line in stream:
                value = json.loads(line)
                progress.append((value["step"], value["loss"]))
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    if progress:
        x, loss = np.asarray(progress).T
        axes[0, 0].plot(x, loss, alpha=0.2, color="tab:blue", linewidth=0.6)
        window = min(100, len(loss))
        smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(x[window - 1 :], smooth, color="tab:blue", label="rolling train objective")
        axes[0, 0].set_yscale("log")
    steps = [row["step"] for row in history]
    for split, color in (("train", "tab:blue"), ("development", "tab:orange")):
        totals = [row[split]["motor_flow"] + 0.25 * row[split]["decision_flow"] for row in history]
        axes[0, 1].plot(steps, totals, label=split, color=color)
        for axis, key, factor in (
            (axes[0, 2], "motor_mae_normalized", 1),
            (axes[1, 0], "arm_mae_rad", 1),
            (axes[1, 1], "jaw_mae_m", 1000),
            (axes[1, 2], "decision_correct", 1),
        ):
            axis.plot(
                steps, [row[split][key] * factor for row in history], label=split, color=color
            )
    titles = (
        "Sampled train flow objective",
        "Fixed masked flow objective",
        "Normalized motor MAE",
        "Decoded arm MAE (rad)",
        "Decoded jaw MAE (mm)",
        "Balanced decision accuracy",
    )
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title)
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1, 2].set_ylim(0, 1.02)
    figure.suptitle("Alex003 SmolVLA pilot — offline learning diagnostics")
    figure.savefig(args.output / "learning-curves.png", dpi=150)
    plt.close(figure)
    text = [
        "# Alex003 pilot training",
        "",
        f"Status: `{state['status']}`.",
        f"Main optimizer updates: **{state['step']}**. Selected checkpoint: **{state.get('best_step')}**.",
        "",
        "![Offline learning curves](learning-curves.png)",
        "",
        "The train objective follows the declared sampling mixture; fixed diagnostics macro-average episodes and phases. "
        "Decision accuracy balances continue, complete and abstain. Offline errors do not establish autonomous task completion.",
        "",
        "The final native before/after panel is reported separately, with nominal, recovery and stop denominators.",
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
