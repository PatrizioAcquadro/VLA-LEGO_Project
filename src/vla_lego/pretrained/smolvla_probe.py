"""Native pretrained SmolVLA smoke on recorded Alex sensors, without execution.

Run in the isolated LeRobot environment. This provisional tensor mapping is not
an Alex-trained policy, finalized normalization or a task-performance result.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="lerobot/smolvla_base")
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {"kind": "pretrained_sensor_smoke", "executed": False, "alex_trained": False}
    try:
        import numpy as np
        import torch
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from PIL import Image

        from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS

        if not torch.cuda.is_available():
            raise RuntimeError("This model probe requires a native CUDA GPU.")
        torch.manual_seed(20261023)
        recording = np.load(args.observation, allow_pickle=False)
        names = recording["joint_names"].tolist()
        order = (*ARM_JOINTS, "NECK_Z", "NECK_Y", *JAW_JOINTS)
        state = recording["joint_position"][[names.index(name) for name in order]]
        image = np.array(Image.open(args.image).convert("RGB"))
        cfg = SmolVLAConfig.from_pretrained(args.model_path)
        cfg.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(18,)),
            "observation.images.head": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, *image.shape[:2])
            ),
        }
        # A padded dimensional smoke only. No physical action semantics are assigned here.
        cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))}
        cfg.device = "cuda"
        started = time.monotonic()
        policy = SmolVLAPolicy.from_pretrained(args.model_path, config=cfg).to("cuda").eval()
        if not all(parameter.is_cuda for parameter in policy.parameters()):
            raise RuntimeError("CPU offload detected; stop instead of timing a mixed path.")
        tokenizer = policy.model.vlm_with_expert.processor.tokenizer
        language = tokenizer(
            "Pick up the red cube and place it on the table.\n",
            return_tensors="pt",
            truncation=True,
            max_length=cfg.tokenizer_max_length,
        )
        batch = {
            "observation.state": torch.from_numpy(state.astype(np.float32))[None].cuda(),
            "observation.images.head": torch.from_numpy(image.copy())
            .permute(2, 0, 1)[None]
            .cuda()
            .float()
            / 255,
            "observation.language.tokens": language["input_ids"].cuda(),
            "observation.language.attention_mask": language["attention_mask"].cuda().bool(),
        }
        load_seconds = time.monotonic() - started
        timings = []
        for _ in range(4):
            policy.reset()
            torch.cuda.synchronize()
            start = time.monotonic()
            actions = policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            timings.append(time.monotonic() - start)
            if not bool(torch.isfinite(actions).all()):
                raise RuntimeError("Non-finite pretrained action chunk.")
        np.save(args.output_dir / "unexecuted_actions.npy", actions.cpu().numpy())
        report.update(
            passed=True,
            model_path=args.model_path,
            device=torch.cuda.get_device_name(),
            parameter_count=sum(p.numel() for p in policy.parameters()),
            action_shape=list(actions.shape),
            state_order=list(order),
            load_seconds=load_seconds,
            inference_seconds=timings,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            normalization="raw diagnostic inputs; train-only Alex statistics still required",
        )
    except Exception:
        report.update(passed=False, error=traceback.format_exc())
        raise
    finally:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
