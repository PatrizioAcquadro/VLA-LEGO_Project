"""Bounded native SmolVLA adaptation on admitted local Alex episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="lerobot/smolvla_base")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.steps < 2 or args.batch_size < 1:
        parser.error("Require at least two optimization steps and a positive batch size.")
    admission = json.loads((args.dataset / "source_admission.json").read_text())
    if admission["split"] != "train" or not admission["admitted_episodes"]:
        raise ValueError("Training accepts only admitted training families.")

    import torch
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.scripts.lerobot_train import train

    if not torch.cuda.is_available():
        raise RuntimeError("Alex VLA adaptation requires the authorized native CUDA GPU.")
    torch.set_num_threads(8)
    policy = SmolVLAConfig.from_pretrained(args.model_path)
    policy.pretrained_path = Path(args.model_path)
    policy.device = "cuda"
    policy.push_to_hub = False
    # Derive the real 18-state/head-camera inputs from this dataset. Retaining
    # the base checkpoint's six-state/three-camera config would silently mis-map Alex.
    policy.input_features = None
    policy.n_action_steps = 10
    policy.freeze_vision_encoder = True
    policy.train_expert_only = True
    policy.train_state_proj = True
    policy.scheduler_warmup_steps = min(20, args.steps // 2)
    policy.scheduler_decay_steps = args.steps
    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=info["repo_id"] if "repo_id" in info else "local/vla-lego-single-block-train",
            root=str(args.dataset.resolve()),
            video_backend="pyav",
            eval_split=0.0,
        ),
        policy=policy,
        output_dir=args.output_dir.resolve(),
        job_name="alex-smolvla-pilot",
        seed=args.seed,
        num_workers=2,
        batch_size=args.batch_size,
        steps=args.steps,
        env_eval_freq=0,
        eval_steps=0,
        log_freq=10,
        save_freq=max(1, args.steps // 2),
        save_checkpoint_to_hub=False,
        wandb=WandBConfig(enable=False),
    )
    print(
        json.dumps(
            {
                "kind": "bounded_training_diagnostic",
                "gpu": torch.cuda.get_device_name(),
                "train_episodes": admission["admitted_episodes"],
                "steps": args.steps,
                "autonomous_success_claim": False,
                "hub_upload": False,
            }
        ),
        flush=True,
    )
    train(cfg)


if __name__ == "__main__":
    main()
