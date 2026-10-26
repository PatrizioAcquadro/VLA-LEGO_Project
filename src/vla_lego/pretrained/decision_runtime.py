"""Past sensor features and calibrated categorical decisions, outside motor chunks."""

from __future__ import annotations

import json
import math
from pathlib import Path

from vla_lego.pretrained.task_decisions import (
    CLASSES,
    DecisionFilter,
    causal_history,
    make_temporal_head,
)


class TemporalDecisionRuntime:
    def __init__(self, checkpoint: Path, observation_normalization: dict):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Learned temporal decisions require native CUDA")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            saved["step"] < 1
            or saved["classes"] != list(CLASSES)
            or saved["history_s"] != 3
            or saved["history_stride_s"] != 0.5
            or saved["feature_protocol"]["format"] != "frozen-observation-features-v2"
            or saved["feature_protocol"]["statistics"]["observation.state"]
            != observation_normalization
        ):
            raise ValueError(
                "Decision checkpoint differs from the trained sensor/history interface"
            )
        self.head = make_temporal_head(saved["feature_width"]).to("cuda").eval()
        self.head.load_state_dict(saved["model"])
        self.calibration = saved["review"]["calibration"]
        self.bias = torch.tensor(self.calibration["bias"], device="cuda")
        self.filter = DecisionFilter(**saved["review"]["thresholds"])
        self.times: list[float] = []
        self.features: list = []
        confirmation = checkpoint.parent / "confirmation.json"
        self.evidence = {
            "checkpoint": str(checkpoint),
            "update": saved["step"],
            "calibration_passed": saved["review"]["passed"],
            "confirmation": json.loads(confirmation.read_text()) if confirmation.exists() else None,
            "autonomous_success": False,
        }

    def update(self, time_s, features):
        import torch

        if (
            not math.isfinite(time_s)
            or features.ndim != 2
            or features.shape[0] != 1
            or not features.is_cuda
            or not torch.isfinite(features).all()
            or (self.times and abs(time_s - self.times[-1] - 0.5) > 1e-6)
        ):
            raise ValueError("Temporal inference requires actual advancing2Hz CUDA observations")
        self.times.append(time_s)
        self.features.append(features.detach())
        # Retain the sample at/before the oldest requested time, never future padding.
        while len(self.times) > 1 and self.times[1] <= time_s - 3:
            self.times.pop(0)
            self.features.pop(0)
        indices = causal_history(self.times, len(self.times) - 1)
        history = torch.stack([self.features[i][0] for i in indices])[None].float()
        with torch.inference_mode():
            logits = self.head(history)
            values = (logits / self.calibration["temperature"] + self.bias).softmax(-1)
        probabilities = values[0].cpu().tolist()
        return {
            "value": self.filter.update(time_s, probabilities),
            "probabilities": probabilities,
            "classes": list(CLASSES),
        }
