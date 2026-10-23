"""Model factory rejects retired architecture types."""

import pytest
from omegaconf import OmegaConf


def test_retired_model_type_is_rejected():
    from vla_lego.models import get_model

    cfg = OmegaConf.create({"model": {"architecture": {"type": "transformer"}}})
    with pytest.raises(ValueError, match="Unknown model type"):
        get_model(cfg)
