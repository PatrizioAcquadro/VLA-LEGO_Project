"""Convert admitted causal reference episodes to a local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--codec", default="h264_nvenc")
    args = parser.parse_args()

    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    admitted = []
    outcomes = []
    families: dict[str, str] = {}
    for directory in args.recordings:
        metadata = json.loads((directory / "episode.json").read_text())
        family, split = metadata["generation_family"], metadata["split"]
        if family in families and families[family] != split:
            raise ValueError(f"Generation family spans splits: {family}")
        families[family] = split
        if split != args.split:
            raise ValueError(f"Source split {split} differs from requested output {args.split}")
        if not metadata.get("control_clock_enforced", False):
            reason = "unverified_control_clock"
        elif not metadata["verified_success"] or metadata["supervision_role"] != "verified_nominal":
            reason = "context_only"
        else:
            reason = "admitted_verified_nominal"
            admitted.append((directory, metadata))
        outcomes.append(
            {
                "recording": str(directory.resolve()),
                "frames": metadata["frames"],
                "physical_success": metadata["verified_success"],
                "admission": reason,
                "generation_family": family,
            }
        )
    if not admitted:
        raise ValueError("No verified, clock-aligned positive episodes. Preserve raw context.")
    first = admitted[0][1]
    if len(first["state_names"]) != 18 or len(first["action_names"]) != 16 or first["fps"] != 20:
        raise ValueError("Expected the provisional 18-state/16-action, 20 Hz Alex interface.")
    features = {
        "observation.state": {"dtype": "float32", "shape": (18,), "names": first["state_names"]},
        "action": {"dtype": "float32", "shape": (16,), "names": first["action_names"]},
        "observation.images.head": {
            "dtype": "video",
            "shape": (600, 960, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        robot_type="purdue_alex003_wsg32",
        fps=20,
        features=features,
        use_videos=True,
        image_writer_threads=4,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec=args.codec, crf=18, g=2, extra_options={"bf": "0"}),
        encoder_threads=4,
    )
    total = 0
    max_age = 0.0
    for directory, metadata in admitted:
        if (
            metadata["state_names"] != first["state_names"]
            or metadata["action_names"] != first["action_names"]
        ):
            raise ValueError("Sources use different signal ordering.")
        count = 0
        with (directory / "frames.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                expected_time = metadata["start_time_s"] + count / 20
                if row["frame"] != count or abs(row["observation_time_s"] - expected_time) > 1e-6:
                    raise ValueError("Frame index or control timestamp is discontinuous.")
                age = row["observation_time_s"] - row["image_time_s"]
                if age < -1e-8 or row["command_time_s"] > row["observation_time_s"] + 1e-8:
                    raise ValueError("Future sensor or command timestamp.")
                state = np.asarray(row["joint_position"], dtype=np.float32)
                action = np.asarray(row["action"], dtype=np.float32)
                if (
                    state.shape != (18,)
                    or action.shape != (16,)
                    or not (np.isfinite(state).all() and np.isfinite(action).all())
                ):
                    raise ValueError("Invalid sensor/action tensor.")
                image_path = (directory / row["head_rgb"]).resolve()
                if not image_path.is_relative_to(directory.resolve()):
                    raise ValueError("Image path escapes its recording.")
                image = np.asarray(Image.open(image_path).convert("RGB"))
                if image.shape != (600, 960, 3):
                    raise ValueError("Head image does not match declared sensor geometry.")
                dataset.add_frame(
                    {
                        "observation.state": state,
                        "action": action,
                        "observation.images.head": image,
                        "task": metadata["instruction"],
                    }
                )
                max_age = max(max_age, age)
                count += 1
        if count != metadata["frames"]:
            raise ValueError("Recording row count differs from its finalized metadata.")
        dataset.save_episode(parallel_encoding=False)
        total += count
    dataset.finalize()
    report = {
        "source_outcomes": outcomes,
        "attempted_sources": len(outcomes),
        "admitted_episodes": len(admitted),
        "admitted_frames": total,
        "split": args.split,
        "families": families,
        "max_camera_age_s": max_age,
        "action_targets": "actually applied position targets; requested targets retained in raw data",
        "privileged_inputs_loaded": False,
        "codec": args.codec,
        "raw_sources_preserved": True,
        "hub_upload": False,
    }
    (args.output_dir / "source_admission.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
