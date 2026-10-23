"""Reload a native trained policy and serve deployable inputs over one local socket."""

from __future__ import annotations

import argparse
import json
import socket
import time
import traceback
from pathlib import Path

from vla_lego.pretrained.transport import receive_message, send_message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--execution-horizon", type=int)
    parser.add_argument("--legacy-absolute-interface", action="store_true")
    args = parser.parse_args()
    if args.socket.exists():
        raise FileExistsError(args.socket)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {"kind": "native_policy_runtime", "checkpoint": str(args.checkpoint)}
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bound = False
    try:
        import numpy as np
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS

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
            interface["action_representation"] not in ("absolute", "measured_arm_offset")
            or interface["action_joint_names"] != [*ARM_JOINTS, *JAW_JOINTS]
            or interface["control_hz"] != 20
        ):
            raise ValueError("Checkpoint uses an incompatible Alex command interface.")

        if not torch.cuda.is_available():
            raise RuntimeError("Policy inference requires native CUDA.")
        torch.set_num_threads(8)
        torch.manual_seed(args.seed)
        started = time.monotonic()
        policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to("cuda").eval()
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
        report.update(
            load_seconds=time.monotonic() - started,
            device=torch.cuda.get_device_name(),
            execution_horizon=horizon,
            alex_interface=interface,
            seed=args.seed,
            input_keys=["observation.state", "observation.images.head", "task"],
            inference_seconds=[],
        )
        listener.bind(str(args.socket))
        bound = True
        listener.listen(1)
        listener.settimeout(180)
        print(json.dumps({"ready": True, "socket": str(args.socket)}), flush=True)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(120)
            while True:
                request, image_bytes = receive_message(connection)
                if request.get("close"):
                    break
                state = np.asarray(request["state"], dtype=np.float32)
                if request["state_names"] != interface["state_names"]:
                    raise ValueError(
                        "Observation signal ordering differs from checkpoint training."
                    )
                shape = tuple(request["image_shape"])
                if state.shape != (18,) or shape != (600, 960, 3) or not np.isfinite(state).all():
                    raise ValueError("Policy requires actual Alex18/head RGB inputs.")
                image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape).copy()
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
                    actions = policy.predict_action_chunk(batch)[:, :horizon]
                    actions = postprocessor(actions)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                if actions.shape != (1, horizon, 16):
                    raise ValueError(f"Unexpected learned action shape {actions.shape}")
                if not bool(torch.isfinite(actions).all()):
                    raise ValueError("Learned actions must be finite.")
                report["inference_seconds"].append(elapsed)
                send_message(
                    connection,
                    {
                        "actions": actions.cpu().tolist()[0],
                        "inference_seconds": elapsed,
                        "action_representation": interface["action_representation"],
                    },
                )
        report.update(passed=True, peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
    except Exception:
        report.update(passed=False, error=traceback.format_exc())
        raise
    finally:
        listener.close()
        if bound:
            args.socket.unlink(missing_ok=True)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
