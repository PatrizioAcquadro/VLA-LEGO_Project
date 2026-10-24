"""Global training schedule, resumable checkpoints and bounded stopping decisions."""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def learning_rate_factor(step: int, maximum: int = 100_000) -> float:
    if step < 1_000:
        return (step + 1) / 1_000
    progress = min(1.0, (step - 1_000) / (maximum - 1_000))
    return 0.025 + 0.975 * (1 + math.cos(math.pi * progress)) / 2


def stopping_reason(history: list[dict]) -> str | None:
    if not history or history[-1]["step"] < 30_000:
        return None
    end = history[-1]["step"]
    older = [row for row in history if row["step"] <= end - 15_000][-3:]
    recent = history[-3:]
    if len(older) != 3 or len(recent) != 3:
        return None

    def median(rows, split, key):
        return statistics.median(row[split][key] for row in rows)

    for key in ("motor_mae_normalized", "jaw_mae_m"):
        before, after = median(older, "development", key), median(recent, "development", key)
        if after < before * 0.98:
            return None
    if median(recent, "development", "decision_correct") > (
        median(older, "development", "decision_correct") + 0.02
    ):
        return None
    train_before = median(older, "train", "motor_mae_normalized")
    train_after = median(recent, "train", "motor_mae_normalized")
    return (
        "generalization_stagnation"
        if train_after < train_before * 0.98
        else "optimization_stagnation"
    )


def persistent_deterioration(history: list[dict]) -> bool:
    """Flag a large held-out regression sustained for at least 5k updates."""
    if len(history) < 10 or history[-1]["step"] < 10_000:
        return False
    recent, earlier = history[-6:], history[:-6]
    if recent[-1]["step"] - recent[0]["step"] < 5_000:
        return False
    reference = min(
        (earlier[i - 2 : i + 1] for i in range(2, len(earlier))),
        key=lambda rows: statistics.median(r["development"]["motor_mae_normalized"] for r in rows),
    )
    dev = statistics.median(r["development"]["motor_mae_normalized"] for r in reference)
    train = statistics.median(r["train"]["motor_mae_normalized"] for r in reference)
    decisions = statistics.median(r["development"]["decision_correct"] for r in reference)
    return bool(
        all(r["development"]["motor_mae_normalized"] > dev * 1.10 for r in recent)
        and statistics.median(r["train"]["motor_mae_normalized"] for r in recent[-3:])
        < train * 0.95
        and statistics.median(r["development"]["decision_correct"] for r in recent[-3:])
        < decisions + 0.05
    )


def better_candidate(current: dict, previous: dict | None) -> bool:
    if previous is None:
        return True
    motor, old = current["motor_mae_normalized"], previous["motor_mae_normalized"]
    return bool(
        motor < old * 0.98
        or (
            motor <= old * 1.02
            and current["decision_correct"] > previous["decision_correct"] + 0.02
        )
    )


def save_checkpoint(policy, preprocessor, postprocessor, optimizer, scheduler, directory, state):
    import random

    import numpy as np
    import torch

    directory = Path(directory)
    pending = directory.parent / f"incomplete-{directory.name}-{os.getpid()}-{time.time_ns()}"
    pending.mkdir(parents=True, exist_ok=False)
    weights = pending / "pretrained_model"
    policy.save_pretrained(weights)
    preprocessor.save_pretrained(weights)
    postprocessor.save_pretrained(weights)
    write_json(weights / "alex_interface.json", state["interface"])
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "state": state,
        },
        pending / "training_state.pt",
    )
    write_json(pending / "state.json", state)
    pending.rename(directory)
    pointer = directory.parent / "latest"
    temporary = directory.parent / "latest.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(directory.name)
    temporary.replace(pointer)


def prune_checkpoints(root: Path, best_step: int) -> None:
    """Retain zero/best, every 10k milestone and the latest two complete states."""
    complete = sorted(
        path for path in root.iterdir() if path.name.isdigit() and (path / "state.json").exists()
    )
    keep = {path.name for path in complete[-2:]}
    for path in complete:
        step = int(path.name)
        if step in (0, best_step) or step % 10_000 == 0 or path.name in keep:
            continue
        shutil.rmtree(path)


def restore_training(checkpoint: Path, optimizer, scheduler) -> dict:
    import random

    import numpy as np
    import torch

    saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    return dict(saved["state"])
