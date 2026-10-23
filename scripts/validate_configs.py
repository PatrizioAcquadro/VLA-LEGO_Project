"""Compose maintained model/cluster presets without loading model weights."""

from itertools import product

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


def main() -> int:
    for model, cluster in product(("vlm", "vlm_dev", "vla", "vla_dev"), ("local", "gilbreth")):
        with initialize_config_module(config_module="configs", version_base=None):
            cfg = compose(config_name="config", overrides=[f"model={model}", f"cluster={cluster}"])
        OmegaConf.to_container(cfg, resolve=True)
        assert cfg.model.vlm.model_id
        assert cfg.model.vlm.max_seq_length > 0
        print(f"PASS model={model} cluster={cluster}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
