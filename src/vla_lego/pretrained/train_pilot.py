"""One resumable GPU campaign on the admitted Alex003 marker-free pilot."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
from pathlib import Path
from typing import Any

from vla_lego.pretrained.pilot_campaign import (
    better_candidate,
    learning_rate_factor,
    persistent_deterioration,
    prune_checkpoints,
    restore_training,
    save_checkpoint,
    stopping_reason,
    write_json,
)
from vla_lego.pretrained.pilot_learning import (
    autocast,
    diagnostics,
    load_policy,
    masked_loss,
    prepare_batch,
)
from vla_lego.pretrained.pilot_training_data import SAMPLING_DESCRIPTION, PilotTrainingData


def data_worker_init(_worker: int) -> None:
    # The trainer handles Slurm's early warning after completing its update.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--until-step", type=int, required=True)
    parser.add_argument("--maximum-step", type=int, default=100_000)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--action-representation",
        choices=("absolute", "observation_arm_offset"),
        default="absolute",
    )
    args = parser.parse_args()
    if args.microbatch < 1 or 64 % args.microbatch or not 0 < args.until_step <= args.maximum_step:
        parser.error("Microbatch must divide 64; allocation endpoint must fit the global schedule")
    if args.resume:
        # Pin the loaded source before checkpoint saves move the latest symlink.
        args.resume = args.resume.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    (args.output / "diagnostics").mkdir(exist_ok=True)
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    state: dict[str, Any] = {
        "status": "initializing",
        "step": 0,
        "pid": os.getpid(),
        "epoch_s": time.time(),
    }
    write_json(args.output / "status.json", state)
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        torch.set_num_threads(8)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        training = PilotTrainingData(
            args.corpus, "train", seed=args.seed, action_representation=args.action_representation
        )
        development = PilotTrainingData(
            args.corpus,
            "development",
            seed=args.seed + 1,
            action_representation=args.action_representation,
        )
        if len(training.episodes) != 50 or len(development.episodes) != 10:
            raise ValueError("This campaign requires all admitted 50+10 episodes")
        if training.families & development.families:
            raise ValueError("Training and development families overlap")
        statistics = json.loads(args.statistics.read_text())
        if "statistics" in statistics:
            if statistics["action_representation"] != args.action_representation:
                raise ValueError("Normalization and action representation disagree")
            statistics = statistics["statistics"]
        elif args.action_representation != "absolute":
            raise ValueError(
                "Offset normalization must explicitly declare its action representation"
            )
        interface = json.loads((args.corpus / "alex_interface.json").read_text())
        interface["action_representation"] = args.action_representation
        model_path = str(args.resume / "pretrained_model") if args.resume else args.model
        policy, pre, post = load_policy(model_path, statistics, resume=bool(args.resume))
        trainable = [p for p in policy.parameters() if p.requires_grad]
        count = sum(p.numel() for p in trainable)
        if not 80_000_000 < count < 130_000_000:
            raise ValueError(f"Unexpected adaptation scope: {count} trainable parameters")
        if not all(p.is_cuda for p in policy.parameters()):
            raise RuntimeError("Unexpected CPU model parameters")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support the declared bfloat16 path")
        optimizer = torch.optim.AdamW(
            trainable, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: learning_rate_factor(step, args.maximum_step)
        )
        if args.resume:
            state = restore_training(args.resume, optimizer, scheduler)
            if state["maximum_step"] != args.maximum_step or state["seed"] != args.seed:
                raise ValueError("Resume would change the declared campaign")
            if state["interface"] != interface or state["statistics"] != statistics:
                raise ValueError("Resume would change action semantics or normalization")
            if any(
                row["development"].get("diagnostic_protocol") != "startup-first-block-and-phase-v2"
                for row in state["history"]
            ):
                raise ValueError(
                    "Historical phase-only diagnostics cannot select a new-protocol candidate. "
                    "Preserve that campaign and start a separate adaptation."
                )
        else:
            state.update(
                interface=interface,
                seed=args.seed,
                maximum_step=args.maximum_step,
                trainable_parameters=count,
                total_parameters=sum(p.numel() for p in policy.parameters()),
                device=torch.cuda.get_device_name(),
                history=[],
                best_step=None,
                sampling=SAMPLING_DESCRIPTION,
                decision_loss_weight=0.25,
                batch_size=64,
                microbatch=args.microbatch,
                statistics=statistics,
                selection_scope="offline_shortlist_pending_physical_evaluation",
            )
        state.update(status="running", pid=os.getpid(), allocation_until=args.until_step)

        def evaluate_and_save():
            report = {"step": state["step"]}
            for name, dataset in (("train", training), ("development", development)):
                result = diagnostics(
                    policy,
                    pre,
                    post,
                    dataset,
                    statistics,
                    batch_size=args.microbatch,
                    seed=args.seed + 100,
                )
                write_json(args.output / "diagnostics" / f"{state['step']:06d}-{name}.json", result)
                report[name] = {k: v for k, v in result.items() if k != "details"}
            state["history"].append(report)
            previous = next(
                (
                    row["development"]
                    for row in state["history"]
                    if row["step"] == state["best_step"]
                ),
                None,
            )
            if better_candidate(report["development"], previous):
                state["best_step"] = state["step"]
            checkpoint = args.output / "checkpoints" / f"{state['step']:06d}"
            save_checkpoint(policy, pre, post, optimizer, scheduler, checkpoint, state)
            prune_checkpoints(
                args.output / "checkpoints", state["best_step"], resume_checkpoint=args.resume
            )
            write_json(args.output / "history.json", state["history"])
            print(json.dumps({"evaluation": report, "best_step": state["best_step"]}), flush=True)

        if not args.resume:
            evaluate_and_save()
        generator = torch.Generator().manual_seed(args.seed + 91)
        loader: DataLoader[dict] = DataLoader(
            training,
            batch_size=args.microbatch,
            sampler=range(state["step"] * 64, args.until_step * 64),
            num_workers=args.workers,
            pin_memory=True,
            generator=generator,
            prefetch_factor=2 if args.workers else None,
            multiprocessing_context="spawn" if args.workers else None,
            worker_init_fn=data_worker_init,
        )
        iterator = iter(loader)
        accumulation = 64 // args.microbatch
        policy.train()
        with (args.output / "progress.jsonl").open("a") as progress:
            while state["step"] < args.until_step:
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for _ in range(accumulation):
                    raw = next(iterator)
                    batch = prepare_batch(raw, pre)
                    with autocast():
                        loss, components = masked_loss(policy, batch)
                    (loss / accumulation).backward()
                    losses.append(
                        [
                            float(loss.detach()),
                            float(components["motor_flow"].detach()),
                            float(components["decision_flow"].detach()),
                        ]
                    )
                norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
                if not torch.isfinite(norm) or norm <= 0:
                    raise FloatingPointError("Invalid/zero trainable gradient norm")
                optimizer.step()
                scheduler.step()
                state["step"] += 1
                state["epoch_s"] = time.time()
                if state["step"] % 10 == 0:
                    record = {
                        "step": state["step"],
                        "epoch_s": state["epoch_s"],
                        "loss": float(np.mean([v[0] for v in losses])),
                        "motor_flow": float(np.mean([v[1] for v in losses])),
                        "decision_flow": float(np.mean([v[2] for v in losses])),
                        "lr": scheduler.get_last_lr()[0],
                        "gradient_norm": float(norm),
                        "allocation_wall_s": time.monotonic() - started,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    }
                    progress.write(json.dumps(record) + "\n")
                    progress.flush()
                    write_json(args.output / "status.json", {**state, "last_progress": record})
                due = state["step"] % 1_000 == 0
                if due or stop_requested or state["step"] == args.until_step:
                    evaluate_and_save()
                reason = None
                if state["step"] == args.maximum_step:
                    reason = "maximum_updates"
                elif due and persistent_deterioration(state["history"]):
                    reason = "persistent_development_deterioration"
                elif state["step"] % 10_000 == 0:
                    reason = stopping_reason(state["history"])
                if reason:
                    state.update(status="training_complete", stopping_reason=reason)
                    break
                if stop_requested:
                    state.update(status="allocation_interrupted")
                    break
        if state["status"] == "running":
            state["status"] = "allocation_complete"
        state.update(epoch_s=time.time(), allocation_wall_s=time.monotonic() - started)
        write_json(args.output / "status.json", state)
    except Exception as error:
        state.update(status="failed", error=repr(error), epoch_s=time.time())
        write_json(args.output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
