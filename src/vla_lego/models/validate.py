"""Real-GPU VLM/VLA smoke with MuJoCo images and an explicit synthetic gradient probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from hydra import compose, initialize_config_module

from vla_lego.models.utils import get_model
from vla_lego.models.vla_model import VLAModel
from vla_lego.models.vlm_backbone import VLMBackbone, preprocess_images


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config", choices=("vlm", "vlm_dev", "vla", "vla_dev"), default="vla_dev"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke; check GPU visibility and permissions")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"model_config": args.model_config, "checks": {}, "learned_task_evaluation": False}
    try:
        with initialize_config_module(config_module="vla_lego.configs", version_base=None):
            cfg = compose(config_name="config", overrides=[f"model={args.model_config}"])
        torch.manual_seed(cfg.experiment.seed)
        model = get_model(cfg).eval()
        backbone = cast(VLMBackbone, model.backbone if isinstance(model, VLAModel) else model)
        device = next(backbone.parameters()).device
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError("Model is partially on CPU; GPU smoke cannot proceed")
        report.update(device=torch.cuda.get_device_name(device), model_id=backbone.info.model_id)

        from vla_lego.sim.bricks.episode import EpisodeManager
        from vla_lego.sim.camera import CAMERA_NAMES, MultiViewRenderer
        from vla_lego.sim.robot_state import AlexRobotState

        episode = EpisodeManager()
        episode.reset(seed=cfg.experiment.seed)
        with MultiViewRenderer(episode.model) as renderer:
            frame = renderer.capture(episode.data, step_index=0)
            images = [frame.views[name].rgb for name in CAMERA_NAMES]
        inputs = preprocess_images(
            backbone, images, "Describe the lightweight bricks on the table.", device=device
        )
        forward_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        }
        with torch.no_grad():
            output = backbone(**forward_inputs)
        if not torch.isfinite(output["logits"]).all():
            raise RuntimeError("Non-finite multimodal logits")
        report["checks"]["four_view_forward"] = {
            "logits_shape": list(output["logits"].shape),
            "images": "native MuJoCo RGB",
            "views": list(CAMERA_NAMES),
        }
        del output
        if isinstance(model, VLAModel):
            state_reader = AlexRobotState(episode.model)
            state = state_reader.normalize(state_reader.get_state(episode.data).to_flat_array())
            with torch.no_grad():
                actions = model.predict_actions(
                    **forward_inputs, robot_state=torch.tensor(np.array([state]), device=device)
                )
            if actions.shape != (1, 16, 17) or not torch.isfinite(actions).all():
                raise RuntimeError("Invalid action chunk")
            report["checks"]["inference"] = {
                "shape": list(actions.shape),
                "executed_on_robot": False,
            }
            # This bounded numerical probe checks gradient routing, not training quality.
            ids = torch.randint(100, 1000, (1, 8), device=device)
            batch = {
                "input_ids": ids,
                "text_labels": ids.clone(),
                "attention_mask": torch.ones(1, 25, device=device, dtype=torch.long),
                "robot_states": torch.tensor(np.array([[state]]), device=device),
                "action_chunks": torch.zeros(1, 1, 16, 17, device=device),
                "chunk_masks": torch.ones(1, 1, 16, device=device),
            }
            model.zero_grad(set_to_none=True)
            losses = model(batch)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise RuntimeError("Non-finite synthetic losses")
            losses["total_loss"].backward()
            if not any(
                p.grad is not None and torch.count_nonzero(p.grad)
                for p in model.action_output_head.parameters()
            ):
                raise RuntimeError("Action head has no gradient")
            if any(p.grad is not None for p in backbone.parameters()):
                raise RuntimeError("Frozen backbone received gradients")
            report["checks"]["synthetic_gradient_probe"] = {
                key: float(value.detach()) for key, value in losses.items()
            }
        report.update(success=True, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    except Exception as error:
        report.update(success=False, error=str(error))
        raise
    finally:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
