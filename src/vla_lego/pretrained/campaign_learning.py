"""Motor-only diagnostics and physical advancement for the expanded Alex corpus."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from vla_lego.pretrained.pilot_learning import collate_samples, masked_loss, prepare_batch

DIAGNOSTIC_PROTOCOL = "campaign-motor-only-v1"


def require_control_gate(output: Path, resume: Path | None) -> dict:
    """Full-task adaptation follows the substantial, physically evaluated control."""
    if resume is None:
        raise ValueError("Full-task adaptation must resume the20k single-block control")
    gate = json.loads((output / "single-block-gate.json").read_text())
    cases = gate["cases"]
    checkpoint = (output / gate["checkpoint"]).resolve(strict=True)
    if (
        len(cases) != 10
        or len({row["case"] for row in cases}) != 10
        or Counter(row["start"] for row in cases) != {"familiar": 5, "held_out": 5}
        or {row["active_arm"] for row in cases} != {"left", "right"}
        or sum(row["physical_success"] is True for row in cases) < 8
        or gate.get("teacher_assisted", True)
        or gate.get("completed_control_updates") != 20_000
    ):
        raise ValueError("The unassisted single-block control has not passed8/10")
    if int(resume.name) == 20_000 and checkpoint != resume.resolve():
        raise ValueError("The resumed control differs from the physically evaluated candidate")
    return dict(gate)


def better_motor_candidate(current: dict, previous: dict | None) -> bool:
    """Only an offline shortlist; physical rollouts own the retained best policy."""
    return previous is None or current["executed_motor_mae_normalized"] < (
        previous["executed_motor_mae_normalized"] * 0.98
    )


def diagnostics(policy, pre, post, dataset, statistics, *, batch_size, seed, single_only):
    import torch

    samples = dataset.diagnostic_samples(single_only=single_only)
    details = []
    policy.eval()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        with torch.inference_mode():
            for begin in range(0, len(samples), batch_size):
                selected = samples[begin : begin + batch_size]
                raw = collate_samples(dataset, selected)
                target = raw["action"].float().clone()
                valid = ~raw["action_is_pad"].clone()
                hold = raw["observation.state"][:, [*range(14), 16, 17]].float().clone()
                hold[:, :14] = 0
                batch = prepare_batch(raw, pre)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, flow = masked_loss(policy, batch)
                    prediction = policy.predict_action_chunk(
                        {k: v for k, v in batch.items() if k not in ("action", "action_is_pad")}
                    )
                prediction = post(prediction).float().cpu()
                if prediction.shape[-1] != 16 or not torch.isfinite(prediction).all():
                    raise FloatingPointError("Expected finite16-motor predictions")
                scales = torch.tensor(statistics["action"]["std"])
                error = (prediction - target).abs()
                for j, (number, sample, bucket) in enumerate(selected):
                    mask = valid[j]
                    prefix = mask.clone()
                    prefix[policy.config.n_action_steps :] = False
                    source = dataset.episodes[number]
                    raw_frame = source.eligible_frames[sample]
                    hold_error = (hold[j] - target[j]).abs()
                    details.append(
                        {
                            "episode": dataset.entries[number]["case"],
                            "category": dataset.entries[number]["category"],
                            "phase": bucket,
                            "frame": raw_frame,
                            "stage": source.annotations[raw_frame]["stage"],
                            "motor_mae_normalized": float((error[j, mask] / scales).mean()),
                            "executed_motor_mae_normalized": float(
                                (error[j, prefix] / scales).mean()
                            ),
                            "hold_executed_motor_mae_normalized": float(
                                (hold_error[prefix] / scales).mean()
                            ),
                            "arm_mae_rad": float(error[j, mask, :14].mean()),
                            "jaw_mae_m": float(error[j, mask, 14:].mean()),
                            "motor_flow": float(flow["motor_per_sample"][j]),
                        }
                    )
    keys = (
        "motor_mae_normalized",
        "executed_motor_mae_normalized",
        "hold_executed_motor_mae_normalized",
        "arm_mae_rad",
        "jaw_mae_m",
        "motor_flow",
    )

    def aggregate(rows):
        metrics = {}
        for key in keys:
            episodes = defaultdict(list)
            for row in rows:
                episodes[row["episode"]].append(row[key])
            metrics[key] = sum(sum(v) / len(v) for v in episodes.values()) / len(episodes)
        return metrics

    result = aggregate(details)
    result.update(
        by_category={
            category: aggregate([row for row in details if row["category"] == category])
            for category in sorted({row["category"] for row in details})
        },
        samples=len(details),
        episodes=len({row["episode"] for row in details}),
        details=details,
        diagnostic_protocol=DIAGNOSTIC_PROTOCOL,
        single_only=single_only,
        selection_scope="offline_shortlist_pending_physical_evaluation",
    )
    policy.train()
    return result
