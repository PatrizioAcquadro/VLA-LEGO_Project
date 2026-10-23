"""Measure trained action errors on predefined admitted training observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 100, 200, 400, 500, 600, 620, 640, 660, 700, 800, 1000],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20261023, 20261027)))
    parser.add_argument("--denoising-steps", type=int)
    args = parser.parse_args()
    admission = json.loads((args.dataset / "source_admission.json").read_text())
    if admission["split"] != "train":
        raise ValueError("This fitting diagnostic accepts training data only.")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("Model diagnostics require native CUDA.")
    torch.set_num_threads(8)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to("cuda").eval()
    if args.denoising_steps is not None:
        if args.denoising_steps < 1:
            raise ValueError("Denoising steps must be positive.")
        policy.config.num_steps = args.denoising_steps
    if not all(p.is_cuda for p in policy.parameters()):
        raise RuntimeError("Unexpected CPU policy parameters.")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint
    )
    dataset = LeRobotDataset(
        "local/vla-lego-single-block-train",
        root=args.dataset,
        delta_timestamps={"action": [i / 20 for i in range(policy.config.chunk_size)]},
        video_backend="pyav",
    )
    predictions, targets, masks, frame_ids, seeds = [], [], [], [], []
    for index in args.frames:
        if not 0 <= index < len(dataset):
            raise ValueError(f"Frame {index} outside dataset of length {len(dataset)}")
        sample = dataset[index]
        batch = preprocessor(
            {key: sample[key] for key in ("observation.state", "observation.images.head", "task")}
        )
        for seed in args.seeds:
            torch.manual_seed(seed)
            policy.reset()
            with torch.inference_mode():
                prediction = postprocessor(policy.predict_action_chunk(batch))[0].cpu().numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError("Non-finite trained action prediction.")
            predictions.append(prediction)
            targets.append(sample["action"].numpy())
            masks.append(~sample["action_is_pad"].numpy())
            frame_ids.append(index)
            seeds.append(seed)
    predicted, expected, valid = np.stack(predictions), np.stack(targets), np.stack(masks)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        prediction=predicted,
        target=expected,
        valid=valid,
        frame=np.asarray(frame_ids),
        seed=np.asarray(seeds),
    )
    names = dataset.meta.features["action"]["names"]
    rmse = {}
    for horizon in (1, 10, policy.config.chunk_size):
        errors = (predicted[:, :horizon] - expected[:, :horizon])[valid[:, :horizon]]
        rmse[str(horizon)] = dict(
            zip(names, np.sqrt(np.mean(errors**2, axis=0)).tolist(), strict=True)
        )
    report = {
        "kind": "training_action_error_diagnostic",
        "checkpoint": str(args.checkpoint),
        "frames": args.frames,
        "seeds": args.seeds,
        "chunks": len(predictions),
        "denoising_steps": policy.config.num_steps,
        "physical_units": "arm radians; WSG leader meters",
        "per_joint_rmse_by_prefix": rmse,
        "model_inputs": ["observation.state", "observation.images.head", "task"],
        "training_families": admission["families"],
        "action_representation": admission.get("action_representation", "absolute"),
        "autonomous_or_generalization_claim": False,
        "device": torch.cuda.get_device_name(),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
