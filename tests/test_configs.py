"""Composition and override contracts for maintained model presets."""

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


@pytest.mark.parametrize("model", ["vlm", "vlm_dev", "vla", "vla_dev"])
@pytest.mark.parametrize("cluster", ["local", "gilbreth"])
def test_supported_configuration_composes(model, cluster):
    with initialize_config_module(config_module="vla_lego.configs", version_base=None):
        cfg = compose(config_name="config", overrides=[f"model={model}", f"cluster={cluster}"])
    assert cfg.model.architecture.type in {"vlm", "vla"}
    assert cfg.model.vlm.model_id == "Qwen/Qwen3.5-4B"
    assert cfg.sim.action.dim == 17
    if cfg.model.architecture.type == "vla":
        assert cfg.model.action_head.action_dim == 17
        assert cfg.model.action_head.state_dim == 52
    OmegaConf.to_container(cfg, resolve=True)


def test_default_and_overrides():
    with initialize_config_module(config_module="vla_lego.configs", version_base=None):
        cfg = compose(config_name="config", overrides=["experiment.seed=13"])
    assert cfg.model.name == "vla_dev"
    assert cfg.experiment.seed == 13
    assert "trainer" not in cfg and "data" not in cfg
