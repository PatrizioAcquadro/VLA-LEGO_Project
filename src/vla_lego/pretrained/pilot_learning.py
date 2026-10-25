"""Native SmolVLA loss and fixed diagnostics for the Alex pilot."""

from __future__ import annotations

from typing import Any


def load_policy(model_path: str, statistics: dict, *, resume: bool = False):
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("Pilot adaptation requires an allocated native CUDA GPU")
    config = SmolVLAConfig.from_pretrained(model_path)
    config.device = "cuda"
    config.push_to_hub = False
    config.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (32,)),
        "observation.images.head": PolicyFeature(FeatureType.VISUAL, (3, 600, 960)),
    }
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (18,))}
    config.n_action_steps = 10
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.train_state_proj = True
    config.adapt_to_pi_aloha = False
    config.use_delta_joint_actions_aloha = False
    # The complete robot-pretrained checkpoint already contains these weights.
    config.load_vlm_weights = False
    config.pretrained_path = model_path
    policy = SmolVLAPolicy.from_pretrained(model_path, config=config).to("cuda")
    tensors = {
        name: {key: torch.tensor(value, dtype=torch.float32) for key, value in stats.items()}
        for name, stats in statistics.items()
    }
    pre, post = make_pre_post_processors(
        config, pretrained_path=model_path if resume else None, dataset_stats=tensors
    )
    return policy, pre, post


def masked_loss(policy, batch, *, noise=None, time=None):
    """Normalize each sample by its valid targets, keeping rare short events useful."""
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, image_masks = policy.prepare_images(batch)
    losses = policy.model.forward(
        images,
        image_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        policy.prepare_state(batch),
        policy.prepare_action(batch),
        noise=noise,
        time=time,
    )[:, :, :18]
    valid = (~batch["action_is_pad"]).to(losses.dtype)
    denominator = valid.sum(1).clamp_min(1)
    motor = (losses[:, :, :16].mean(-1) * valid).sum(1) / denominator
    decision = (losses[:, :, 16:18].mean(-1) * valid).sum(1) / denominator
    loss = motor.mean() + 0.25 * decision.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite masked pilot loss")
    return loss, {
        "motor_flow": motor.mean(),
        "decision_flow": decision.mean(),
        "motor_per_sample": motor,
        "decision_per_sample": decision,
    }


def collate_samples(dataset, samples):
    from torch.utils.data import default_collate

    return default_collate([dataset.episodes[e][i] for e, i, _ in samples])


def prepare_batch(raw, processor):
    result = processor(raw)
    result["action_is_pad"] = raw["action_is_pad"].to("cuda")
    return result


def diagnostics(
    policy, processor, postprocessor, dataset, statistics, *, batch_size: int, seed: int
):
    import torch

    samples = dataset.diagnostic_samples()
    values = []
    policy.eval()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        with torch.inference_mode():
            for begin in range(0, len(samples), batch_size):
                selected = samples[begin : begin + batch_size]
                raw = collate_samples(dataset, selected)
                target = raw["action"].float().clone()
                valid = ~raw["action_is_pad"].clone()
                hold_targets = raw["observation.state"][:, [*range(14), 16, 17]].float().clone()
                if dataset.action_representation == "observation_arm_offset":
                    hold_targets[:, :14] = 0.0
                batch = prepare_batch(raw, processor)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, flow = masked_loss(policy, batch)
                    prediction = policy.predict_action_chunk(
                        {
                            key: value
                            for key, value in batch.items()
                            if key not in ("action", "action_is_pad")
                        }
                    )
                prediction = postprocessor(prediction).float().cpu()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Non-finite decoded diagnostic actions")
                scales = torch.tensor(statistics["action"]["std"][:16])
                error = (prediction[:, :, :16] - target[:, :, :16]).abs()
                for j, (episode, sample, bucket) in enumerate(selected):
                    mask = valid[j]
                    source = dataset.episodes[episode]
                    raw_frame = source.eligible_frames[sample]
                    prefix = mask.clone()
                    prefix[policy.config.n_action_steps :] = False
                    hold_error = (hold_targets[j] - target[j, :, :16]).abs()
                    first_decision = prediction[j, 0, 16:18] >= 0.5
                    expected = target[j, 0, 16:18] >= 0.5
                    values.append(
                        {
                            "episode": dataset.cases[episode],
                            "phase": bucket,
                            "frame": raw_frame,
                            "stage": source.annotations[raw_frame]["stage"],
                            "motor_mae_normalized": float((error[j, mask] / scales).mean()),
                            "arm_mae_rad": float(error[j, mask, :14].mean()),
                            "jaw_mae_m": float(error[j, mask, 14:16].mean()),
                            "executed_motor_mae_normalized": float(
                                (error[j, prefix] / scales).mean()
                            ),
                            "hold_executed_motor_mae_normalized": float(
                                (hold_error[prefix] / scales).mean()
                            ),
                            "decision_correct": bool(torch.equal(first_decision, expected)),
                            "expected_decision": expected.int().tolist(),
                            "predicted_decision": first_decision.int().tolist(),
                            "motor_flow": float(flow["motor_per_sample"][j]),
                            "decision_flow": float(flow["decision_per_sample"][j]),
                        }
                    )
    metrics: dict[str, Any] = {}
    for key in (
        "motor_mae_normalized",
        "arm_mae_rad",
        "jaw_mae_m",
        "executed_motor_mae_normalized",
        "hold_executed_motor_mae_normalized",
        "decision_correct",
        "motor_flow",
        "decision_flow",
    ):
        # Each episode has one sample per represented phase; average within an
        # episode first so a long/more annotated demonstration cannot dominate.
        groups: dict[str, list[float]] = {}
        for value in values:
            groups.setdefault(value["episode"], []).append(float(value[key]))
        metrics[key] = sum(sum(v) / len(v) for v in groups.values()) / len(groups)
    metrics["samples"] = len(values)
    metrics["episodes"] = len(dataset.episodes)
    metrics["decision_accuracy_by_episode"] = metrics["decision_correct"]
    by_class = {}
    for name, target in (("continue", [0, 0]), ("complete", [1, 0]), ("abstain", [0, 1])):
        selected = [row for row in values if row["expected_decision"] == target]
        by_class[name] = sum(row["decision_correct"] for row in selected) / len(selected)
    metrics["decision_accuracy_by_class"] = by_class
    metrics["decision_correct"] = sum(by_class.values()) / len(by_class)
    metrics["details"] = values
    metrics["diagnostic_protocol"] = "startup-first-block-and-phase-v2"
    metrics["execution_horizon"] = policy.config.n_action_steps
    policy.train()
    return metrics


def autocast():
    import torch

    return torch.autocast("cuda", dtype=torch.bfloat16)
