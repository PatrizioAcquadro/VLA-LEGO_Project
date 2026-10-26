"""Learned temporal task decisions over deployable observations, separate from motors."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

CLASSES = ("continue", "complete", "abstain")


def causal_history(times: Sequence[float], index: int, *, seconds=3.0, stride=0.5) -> list[int]:
    """Select a fixed past window; startup padding repeats the first actual observation."""
    if not 0 <= index < len(times) or seconds <= 0 or stride <= 0:
        raise ValueError("Invalid causal observation history")
    if abs(seconds / stride - round(seconds / stride)) > 1e-8:
        raise ValueError("History duration must contain whole observation strides")
    return [
        max(0, bisect.bisect_right(times, times[index] - offset * stride, hi=index + 1) - 1)
        for offset in reversed(range(round(seconds / stride) + 1))
    ]


@dataclass
class DecisionFilter:
    """Calibrated confidence and persistence; it never consults evaluator state."""

    complete_threshold: float
    abstain_threshold: float
    persistence_s: float = 0.5
    pending: str = "continue"
    since: float | None = None
    last_time: float | None = None

    def __post_init__(self):
        if not all(0.5 <= v <= 1 for v in (self.complete_threshold, self.abstain_threshold)):
            raise ValueError("Terminal thresholds require calibrated probabilities")
        if self.persistence_s <= 0:
            raise ValueError("Terminal calls require temporal persistence")

    def update(self, time_s: float, probabilities: Sequence[float]) -> str:
        if (
            len(probabilities) != 3
            or not math.isfinite(time_s)
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
            or abs(sum(probabilities) - 1) > 1e-4
        ):
            raise ValueError("Expected three normalized finite class probabilities")
        if self.last_time is not None and time_s <= self.last_time:
            raise ValueError("Decision observations must advance the causal clock")
        self.last_time = time_s
        selected = max(range(3), key=lambda k: probabilities[k])
        candidate = CLASSES[selected]
        threshold = self.complete_threshold if selected == 1 else self.abstain_threshold
        if not selected or probabilities[selected] < threshold:
            candidate = "continue"
        if candidate != self.pending or candidate == "continue":
            self.pending, self.since = candidate, time_s
        if (
            candidate != "continue"
            and self.since is not None
            and time_s - self.since >= self.persistence_s
        ):
            return candidate
        return "continue"


def make_temporal_head(feature_width: int, hidden_width: int = 256):
    """A compact trainable head; frozen VLM features retain instruction conditioning."""
    import torch
    from torch import nn

    class TemporalHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.normalize = nn.LayerNorm(feature_width)
            self.memory = nn.GRU(feature_width, hidden_width, batch_first=True)
            self.output = nn.Linear(hidden_width, 3)

        def forward(self, history):
            if history.ndim != 3 or history.shape[-1] != feature_width:
                raise ValueError("Expected batch × past observations × frozen features")
            if not torch.isfinite(history).all():
                raise ValueError("Non-finite decision observations")
            sequence, _ = self.memory(self.normalize(history))
            return self.output(sequence[:, -1])

    return TemporalHead()


def frozen_observation_features(policy, batch):
    """Fuse RGB/language with the frozen VLM; append observed normalized proprioception.

    Exclude its state token so changes to the motor state projector cannot invalidate
    cached classifier features. The VLM attention mask prevents image/language tokens
    from attending to state or action tokens.
    """
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if not torch.cuda.is_available():
        raise RuntimeError("Frozen visual/language feature extraction requires native CUDA")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        images, masks = policy.prepare_images(batch)
        embeddings, valid, attention = policy.model.embed_prefix(
            images,
            masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=policy.prepare_state(batch),
        )
        outputs, _ = policy.model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(valid, attention),
            position_ids=torch.cumsum(valid, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=True,
        )
        visible = valid & ~attention
        fused = (outputs[0].float() * visible[..., None]).sum(1) / visible.sum(1, keepdim=True)
        state = batch["observation.state"].reshape(fused.shape[0], -1).float()
        features = torch.cat((fused, state), dim=-1)
    if not torch.isfinite(features).all():
        raise FloatingPointError("Non-finite frozen observation features")
    return features
