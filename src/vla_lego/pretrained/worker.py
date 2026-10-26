"""Reload a native trained policy and serve deployable inputs over one local socket."""

from __future__ import annotations

import argparse
import json
import socket
import time
import traceback
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from vla_lego.pretrained.transport import receive_message, send_message

if TYPE_CHECKING:
    from torch import Tensor


def blend_jaw_overlap(actions: Tensor, previous: Tensor | None, horizon: int) -> Tensor:
    """Average jaw targets predicted for the same times by adjacent chunks."""
    if actions.ndim != 3 or actions.shape[-1] < 16 or not 1 <= horizon <= actions.shape[1]:
        raise ValueError("Expected a motor chunk containing the executed horizon.")
    served = actions[:, :horizon].clone()
    if previous is not None:
        if previous.ndim != 3 or previous.shape[::2] != actions.shape[::2]:
            raise ValueError("Overlapping chunks must have matching batches and action widths.")
        overlap = min(horizon, previous.shape[1] - horizon)
        if overlap > 0:
            served[:, :overlap, 14:16] = (
                actions[:, :overlap, 14:16] + previous[:, horizon : horizon + overlap, 14:16]
            ) / 2
    return served


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--execution-horizon", type=int)
    parser.add_argument("--jaw-overlap", action="store_true")
    parser.add_argument("--legacy-absolute-interface", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--decision-checkpoint", type=Path)
    args = parser.parse_args()
    if args.socket.exists():
        raise FileExistsError(args.socket)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {"kind": "native_policy_runtime", "checkpoint": str(args.checkpoint)}
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bound = False
    video = video_stream = None
    try:
        import numpy as np
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS
        from vla_lego.pretrained.decisions import DECISION_NAMES

        interface_path = args.checkpoint / "alex_interface.json"
        if interface_path.exists():
            interface = json.loads(interface_path.read_text())
        elif args.legacy_absolute_interface:
            interface = {
                "action_representation": "absolute",
                "state_names": [*ARM_JOINTS, "NECK_Z", "NECK_Y", *JAW_JOINTS],
                "action_joint_names": [*ARM_JOINTS, *JAW_JOINTS],
                "control_hz": 20,
            }
        else:
            raise ValueError(
                "Checkpoint lacks its Alex action interface; legacy absolute requires an explicit flag."
            )
        if (
            interface["action_representation"]
            not in ("absolute", "measured_arm_offset", "observation_arm_offset")
            or interface["action_joint_names"] != [*ARM_JOINTS, *JAW_JOINTS]
            or interface["control_hz"] != 20
            or interface.get("task_decision_names", []) not in ([], list(DECISION_NAMES))
        ):
            raise ValueError("Checkpoint uses an incompatible Alex command interface.")

        if not torch.cuda.is_available():
            raise RuntimeError("Policy inference requires native CUDA.")
        torch.set_num_threads(8)
        torch.manual_seed(args.seed)
        started = time.monotonic()
        policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to("cuda").eval()
        decision_names = interface.get("task_decision_names", [])
        separate_decisions = interface.get("task_decision_module") == (
            "separate_causal_temporal_classifier"
        )
        decision_runtime = None
        if separate_decisions:
            from vla_lego.pretrained.decision_runtime import TemporalDecisionRuntime
            from vla_lego.pretrained.task_decisions import CLASSES

            if (
                decision_names
                or interface.get("task_decision_classes") != list(CLASSES)
                or args.decision_checkpoint is None
            ):
                raise ValueError(
                    "Motor-only campaign checkpoints require their separate learned head"
                )
            decision_runtime = TemporalDecisionRuntime(
                args.decision_checkpoint, interface["observation_normalization"]
            )
        elif args.decision_checkpoint is not None:
            raise ValueError("A separate head requires the declared motor-only campaign interface")
        action_width = 16 + len(decision_names)
        if tuple(policy.config.output_features["action"].shape) != (action_width,):
            raise ValueError("Checkpoint action width differs from its declared interface.")
        if not all(parameter.is_cuda for parameter in policy.parameters()):
            raise RuntimeError("Unexpected CPU policy parameters.")
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, pretrained_path=args.checkpoint
        )
        policy.reset()
        horizon = (
            policy.config.n_action_steps
            if args.execution_horizon is None
            else args.execution_horizon
        )
        if not 1 <= horizon <= policy.config.chunk_size:
            raise ValueError("Execution horizon must fit the learned prediction chunk.")
        if separate_decisions and horizon != 10:
            raise ValueError("This calibrated temporal head observes the policy at2Hz")
        report.update(
            load_seconds=time.monotonic() - started,
            device=torch.cuda.get_device_name(),
            execution_horizon=horizon,
            alex_interface=interface,
            served_action_representation=(
                "absolute"
                if interface["action_representation"] == "observation_arm_offset"
                else interface["action_representation"]
            ),
            seed=args.seed,
            jaw_overlap=args.jaw_overlap,
            input_keys=["observation.state", "observation.images.head", "task"],
            inference_seconds=[],
            decision_evidence=None if decision_runtime is None else decision_runtime.evidence,
        )
        if args.record_video:
            import av

            video = av.open(
                str(args.output_dir / "observations.mp4"),
                mode="w",
                options={"movflags": "faststart"},
            )
            video_stream = video.add_stream("h264_nvenc", rate=Fraction(20, horizon))
            video_stream.width, video_stream.height = 960, 600
            video_stream.pix_fmt = "yuv420p"
            video_stream.time_base = Fraction(horizon, 20)
            video_stream.options = {"cq": "23", "bf": "0"}
            report.update(
                video_fps=20 / horizon,
                video_frames=0,
                video_scope="actual policy observations; simulation time",
            )
        listener.bind(str(args.socket))
        bound = True
        listener.listen(1)
        listener.settimeout(180)
        print(json.dumps({"ready": True, "socket": str(args.socket)}), flush=True)
        connection, _ = listener.accept()
        previous_actions = None
        instruction = None
        with connection:
            connection.settimeout(120)
            while True:
                request, image_bytes = receive_message(connection)
                if request.get("close"):
                    break
                if request.get("prepare"):
                    # Long physical recovery prefixes keep the transport alive;
                    # this message never enters model observations or inference.
                    send_message(connection, {"ready": True})
                    continue
                state = np.asarray(request["state"], dtype=np.float32)
                if instruction is None:
                    instruction = request["instruction"]
                elif request["instruction"] != instruction:
                    raise ValueError("A new instruction requires a fresh temporal episode")
                arm_anchor = state[:14].copy()
                if request["state_names"] != interface.get(
                    "position_names", interface["state_names"]
                ):
                    raise ValueError(
                        "Observation signal ordering differs from checkpoint training."
                    )
                shape = tuple(request["image_shape"])
                if state.shape != (18,) or shape != (600, 960, 3) or not np.isfinite(state).all():
                    raise ValueError("Policy requires actual Alex18/head RGB inputs.")
                if interface.get("include_arm_velocity", False):
                    velocity = np.asarray(request["velocity"], dtype=np.float32)
                    if velocity.shape != (18,) or not np.isfinite(velocity).all():
                        raise ValueError("Checkpoint requires actual finite arm velocities.")
                    state = np.concatenate((state, velocity[:14]))
                expected_state_dim = policy.config.input_features["observation.state"].shape[0]
                if (
                    state.shape != (expected_state_dim,)
                    or len(interface["state_names"]) != expected_state_dim
                ):
                    raise ValueError("Measured state construction differs from trained features.")
                image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape).copy()
                if video is not None and video_stream is not None:
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    frame.pts = report["video_frames"]
                    frame.time_base = Fraction(horizon, 20)
                    for packet in video_stream.encode(frame):
                        video.mux(packet)
                    report["video_frames"] += 1
                batch = preprocessor(
                    {
                        "observation.state": torch.from_numpy(state),
                        "observation.images.head": torch.from_numpy(image).permute(2, 0, 1).float()
                        / 255,
                        "task": request["instruction"],
                    }
                )
                torch.cuda.synchronize()
                started = time.monotonic()
                with torch.inference_mode():
                    learned_decision = None
                    if decision_runtime is not None:
                        from vla_lego.pretrained.task_decisions import frozen_observation_features

                        learned_decision = decision_runtime.update(
                            request["observation_time_s"],
                            frozen_observation_features(policy, batch),
                        )
                    predictions = policy.predict_action_chunk(batch)
                    actions = predictions if args.jaw_overlap else predictions[:, :horizon]
                    actions = postprocessor(actions)
                    if interface["action_representation"] == "observation_arm_offset":
                        actions[:, :, :14] += torch.as_tensor(arm_anchor, device=actions.device)
                    if args.jaw_overlap:
                        served = blend_jaw_overlap(actions, previous_actions, horizon)
                        previous_actions = actions.detach()
                        actions = served
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                if actions.shape != (1, horizon, action_width):
                    raise ValueError(f"Unexpected learned action shape {actions.shape}")
                if not bool(torch.isfinite(actions).all()):
                    raise ValueError("Learned actions must be finite.")
                report["inference_seconds"].append(elapsed)
                send_message(
                    connection,
                    {
                        "actions": actions.cpu().tolist()[0],
                        "inference_seconds": elapsed,
                        "action_representation": (
                            "absolute"
                            if interface["action_representation"] == "observation_arm_offset"
                            else interface["action_representation"]
                        ),
                        "task_decision_names": decision_names,
                        "task_decision": learned_decision,
                    },
                )
        report.update(passed=True, peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
    except Exception:
        report.update(passed=False, error=traceback.format_exc())
        raise
    finally:
        if video is not None and video_stream is not None:
            try:
                for packet in video_stream.encode():
                    video.mux(packet)
                video.close()
            except Exception:
                report["video_error"] = traceback.format_exc()
        listener.close()
        if bound:
            args.socket.unlink(missing_ok=True)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
