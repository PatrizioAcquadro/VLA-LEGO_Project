"""Maintained model utility contracts."""

import pytest
import torch
from omegaconf import OmegaConf


class TestModelUtils:
    """Test model utility functions."""

    def test_count_parameters(self):
        """Test parameter counting."""
        from models.utils import count_parameters

        model = torch.nn.Linear(100, 100)

        total = count_parameters(model, trainable_only=False)
        trainable = count_parameters(model, trainable_only=True)

        # 100*100 weights + 100 bias = 10100
        assert total == 10100
        assert trainable == 10100

    def test_count_parameters_frozen(self):
        """Test counting with frozen parameters."""
        from models.utils import count_parameters

        model = torch.nn.Linear(100, 100)
        model.weight.requires_grad = False

        total = count_parameters(model, trainable_only=False)
        trainable = count_parameters(model, trainable_only=True)

        assert total == 10100
        assert trainable == 100  # Only bias

    def test_format_params(self):
        """Test parameter formatting."""
        from models.utils import format_params

        assert format_params(1_500_000_000) == "1.5B"
        assert format_params(350_000_000) == "350.0M"
        assert format_params(12_000) == "12.0K"
        assert format_params(500) == "500"


def test_retired_model_type_is_rejected():
    from models import get_model

    cfg = OmegaConf.create({"model": {"architecture": {"type": "transformer"}}})
    with pytest.raises(ValueError, match="Unknown model type"):
        get_model(cfg)
