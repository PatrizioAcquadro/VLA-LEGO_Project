"""Native SmolVLA motor policy and masked flow loss."""

from __future__ import annotations


def load_policy(
    model_path: str,
    statistics: dict,
    *,
    resume: bool = False,
):
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
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (16,))}
    config.tokenizer_max_length = 96
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
    width = batch["action"].shape[-1]
    if width != 16:
        raise ValueError("Expected 16 motor targets; task decisions are trained separately")
    losses = policy.model.forward(
        images,
        image_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        policy.prepare_state(batch),
        policy.prepare_action(batch),
        noise=noise,
        time=time,
    )[:, :, :width]
    valid = (~batch["action_is_pad"]).to(losses.dtype)
    denominator = valid.sum(1).clamp_min(1)
    motor = (losses[:, :, :16].mean(-1) * valid).sum(1) / denominator
    loss = motor.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite masked pilot loss")
    return loss, {
        "motor_flow": motor.mean(),
        "decision_flow": motor.new_zeros(()),
        "motor_per_sample": motor,
    }


def collate_samples(dataset, samples):
    from torch.utils.data import default_collate

    return default_collate([dataset.episodes[e][i] for e, i, _ in samples])


def prepare_batch(raw, processor):
    result = processor(raw)
    result["action_is_pad"] = raw["action_is_pad"].to("cuda")
    return result


def autocast():
    import torch

    return torch.autocast("cuda", dtype=torch.bfloat16)
