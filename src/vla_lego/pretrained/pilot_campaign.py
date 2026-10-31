"""Global motor schedule and resumable checkpoints."""

from __future__ import annotations

import json
import math
import os
import shutil
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


def prune_checkpoints(root: Path, best_step: int, *, resume_checkpoint: Path | None = None) -> None:
    """Retain milestones, recent states and the source still mapped by this process."""
    physical = root / "physical-best"
    if physical.exists() and not physical.is_symlink():
        raise ValueError("Physical best must be an explicit checkpoint link")
    retained = physical.resolve(strict=True) if physical.is_symlink() else None
    if retained is not None and (
        retained.parent != root.resolve()
        or not retained.name.isdigit()
        or not (retained / "state.json").is_file()
    ):
        raise ValueError("Physical best must identify a complete checkpoint in this campaign")
    complete = sorted(
        path for path in root.iterdir() if path.name.isdigit() and (path / "state.json").exists()
    )
    keep = {path.name for path in complete[-2:]}
    for path in complete:
        if retained is not None and path.resolve() == retained:
            continue
        if resume_checkpoint is not None and path.resolve() == resume_checkpoint.resolve():
            continue
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
