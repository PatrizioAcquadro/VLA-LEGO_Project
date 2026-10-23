"""Model utility functions."""

import torch.nn as nn
from omegaconf import DictConfig


def get_model(cfg: DictConfig) -> nn.Module:
    """Factory function to create model from config.

    Args:
        cfg: Hydra configuration

    Returns:
        Instantiated model
    """
    model_type = cfg.model.architecture.type

    if model_type == "vlm":
        from vla_lego.models.vlm_backbone import load_vlm_backbone

        return load_vlm_backbone(cfg)
    elif model_type == "vla":
        from vla_lego.models.vla_model import load_vla_model

        return load_vla_model(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
