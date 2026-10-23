"""Bounded autonomous tower execution; task truth is used only for independent scoring."""

from __future__ import annotations

import argparse
import json
import socket
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from vla_lego.pretrained.decisions import DECISION_NAMES, decode_decision
from vla_lego.pretrained.transport import receive_message, send_message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=6000)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--missing-target", type=int, choices=range(5))
    args = parser.parse_args()
    if args.ticks < 1:
        parser.error("--ticks must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "autonomous_tower_diagnostic",
        "learned": True,
        "passed": False,
        "case_passed": False,
        "teacher_commands": False,
        "autonomy_started": False,
        "qualification": False,
        "simulation_paused_during_inference": True,
        "tick_budget": args.ticks,
        "head_capture_hz": 20,
    }
    app = None
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(120)
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        import numpy as np
        from PIL import Image

        from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS, AssemblyEnvironment
        from vla_lego.isaac.evaluation import TowerEvaluator
        from vla_lego.isaac.task import initial_tower_task

        task = initial_tower_task(mirror=args.mirror)
        if args.missing_target is not None:
            task = task.without_target(args.missing_target)
        env = AssemblyEnvironment(
            args.cache_dir.resolve(), placements=task.placements, assembly_xy_m=task.assembly_xy_m
        )
        env.step(400)
        connection.connect(str(args.socket))
        report.update(task=asdict(task), robot_joint_names=env.robot.data.joint_names)
        names = (*ARM_JOINTS, *JAW_JOINTS)
        ids = [env.robot.data.joint_names.index(n) for n in names]
        limits = env.robot.data.joint_pos_limits.torch[0, ids].cpu().numpy()
        evaluator = TowerEvaluator(task, env.contact_robot_paths)
        evaluator.update(env.time, [env.task_truth(i) for i in range(len(env.blocks))])
        chunk, inference_seconds = [], []
        representation = decision_names = None
        terminal, executed, clipped_ticks = "budget", 0, 0
        report["autonomy_started"] = True
        with (args.output_dir / "trace.jsonl").open("w") as trace:
            for tick in range(args.ticks):
                env.capture()
                if not chunk:
                    observation = env.observe()
                    send_message(
                        connection,
                        {
                            "state": observation["joint_position"].tolist(),
                            "state_names": list(observation["joint_names"]),
                            "velocity": observation["joint_velocity"].tolist(),
                            "image_shape": list(observation["head_rgb"].shape),
                            "instruction": task.instruction,
                        },
                        observation["head_rgb"].tobytes(),
                    )
                    response, _ = receive_message(connection)
                    current_representation = response["action_representation"]
                    current_names = response.get("task_decision_names", [])
                    if current_representation not in (
                        "absolute",
                        "measured_arm_offset",
                    ) or current_names not in ([], list(DECISION_NAMES)):
                        raise ValueError("Unknown learned control/decision interface.")
                    if representation is not None and (
                        representation != current_representation or decision_names != current_names
                    ):
                        raise ValueError("Learned interface changed inside the episode.")
                    representation, decision_names = current_representation, current_names
                    chunk = response["actions"]
                    if not chunk:
                        raise ValueError("Empty learned action chunk.")
                    inference_seconds.append(response["inference_seconds"])
                    if tick % 200 == 0:
                        Image.fromarray(observation["head_rgb"]).save(
                            args.output_dir / f"head-{tick:04d}.png"
                        )
                action = np.asarray(chunk.pop(0), dtype=np.float32)
                if action.shape != (16 + len(decision_names),) or not np.isfinite(action).all():
                    raise ValueError("Learned action differs from the declared finite interface.")
                decision = decode_decision(action[16:]) if decision_names else "continue"
                requested = action[:16].copy()
                if representation == "measured_arm_offset":
                    requested[:14] += env.robot.data.joint_pos.torch[0, ids[:14]].cpu().numpy()
                bounded = np.clip(requested, limits[:, 0], limits[:, 1])
                clipped_ticks += int(np.any(requested != bounded))
                if decision == "continue":
                    env.set_joint_targets(dict(zip(names, bounded.tolist(), strict=True)))
                    env.step(10)
                    executed += 1
                    truth = [env.task_truth(i) for i in range(len(env.blocks))]
                    evaluator.update(env.time, truth)
                else:
                    # A model decision freezes the last applied targets, including any grasp.
                    terminal = decision
                    truth = [env.task_truth(i) for i in range(len(env.blocks))]
                trace.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "time_s": env.time,
                            "learned_action": action.tolist(),
                            "decision": decision,
                            "absolute_request": requested.tolist(),
                            "bounded_request": bounded.tolist(),
                            "blocks": truth,
                            "applied_joint_target": env.target[0].tolist(),
                            "robot": env.robot_diagnostics(),
                        }
                    )
                    + "\n"
                )
                if decision != "continue":
                    break
                if tick % 100 == 0:
                    trace.flush()
                    print(json.dumps({"tick": tick, "time_s": env.time}), flush=True)
            hold = []
            for _ in range(40):
                env.capture()
                env.step(10)
                truth = [env.task_truth(i) for i in range(len(env.blocks))]
                evaluator.update(env.time, truth)
                hold.append(truth)
                trace.write(
                    json.dumps(
                        {
                            "phase": "terminal_hold",
                            "time_s": env.time,
                            "blocks": truth,
                            "robot": env.robot_diagnostics(),
                        }
                    )
                    + "\n"
                )
        evaluation = evaluator.result()
        justified_stop = (
            terminal == "abstain"
            and not task.requested_pieces_available
            and all(b["robot_contact_force_n"] < 1e-6 for row in hold for b in row)
        )
        passed = evaluation["passed"] and terminal == "complete"
        report.update(
            evaluation=evaluation,
            tower_constructed=evaluation["passed"],
            passed=passed,
            justified_stop=justified_stop,
            case_passed=justified_stop if args.missing_target is not None else passed,
            terminal_decision=terminal,
            learned_completion_signal=bool(decision_names),
            task_decision_names=decision_names,
            executed_ticks=executed,
            clipped_request_ticks=clipped_ticks,
            inference_seconds=inference_seconds,
            action_representation=representation,
        )
        env.capture()
        Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "final.png")
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        try:
            send_message(connection, {"close": True})
        except OSError:
            pass
        connection.close()
        report["wall_seconds_before_app_close"] = time.monotonic() - started
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in report.items() if k not in ("task", "inference_seconds")}),
            flush=True,
        )
        if app is not None:
            app.close(exit_code=0 if report["case_passed"] else 1)


if __name__ == "__main__":
    main()
