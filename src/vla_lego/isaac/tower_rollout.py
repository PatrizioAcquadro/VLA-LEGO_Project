"""Bounded autonomous tower execution; task truth is used only for independent scoring."""

from __future__ import annotations

import argparse
import json
import socket
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    parser.add_argument("--pilot-corpus", type=Path)
    parser.add_argument("--pilot-case")
    parser.add_argument("--pilot-split", choices=("train", "development"), default="development")
    parser.add_argument("--teacher-start-stage")
    parser.add_argument("--campaign-cases", type=Path)
    parser.add_argument("--campaign-case")
    parser.add_argument("--campaign-partition", choices=("train", "calibration", "confirmation"))
    parser.add_argument("--campaign-corpus", type=Path)
    args = parser.parse_args()
    if args.ticks < 1:
        parser.error("--ticks must be positive")
    if bool(args.pilot_corpus) != bool(args.pilot_case):
        parser.error("Pilot corpus and case must be supplied together")
    if args.pilot_case and (args.mirror or args.missing_target is not None):
        parser.error("Recorded pilot cases cannot be combined with legacy scene variants")
    if args.campaign_case and (args.pilot_case or args.mirror or args.missing_target is not None):
        parser.error("Campaign cases cannot be combined with pilot or legacy scene variants")
    if bool(args.campaign_cases) != bool(args.campaign_case) or bool(args.campaign_case) != bool(
        args.campaign_partition
    ):
        parser.error("Campaign declaration, case and explicit partition are required together")
    if args.campaign_corpus and not args.campaign_case:
        parser.error("A campaign corpus requires its declared evaluation case")
    if args.teacher_start_stage and not (args.pilot_case or args.campaign_case):
        parser.error("A conditional demonstration handoff requires a recorded pilot case")
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

        source_report = directory = campaign_case = None
        if args.campaign_case:
            from vla_lego.isaac.campaign import load_campaign

            campaign_case = next(
                case
                for case in load_campaign(args.campaign_cases)
                if case.name == args.campaign_case
            )
            if campaign_case.row["partition"] != args.campaign_partition:
                raise ValueError("Campaign evaluation would cross the declared family partition")
            task = campaign_case.task()
            if campaign_case.category == "recovery" or args.teacher_start_stage:
                if args.campaign_corpus is None:
                    raise ValueError("A physical recovery prefix requires its admitted recording")
                entries = json.loads((args.campaign_corpus / "admitted-episodes.json").read_text())[
                    "episodes"
                ]
                entry = next(row for row in entries if row["case"] == campaign_case.name)
                directory = (args.campaign_corpus / entry["recording"]).resolve()
                if (
                    not entry["admitted"]
                    or entry["partition"] != args.campaign_partition
                    or not directory.is_relative_to(args.campaign_corpus.resolve())
                ):
                    raise ValueError("Recovery prefix is not the declared admitted source")
                source_report = json.loads((directory.parent / "report.json").read_text())
        elif args.pilot_case:
            from vla_lego.isaac.pilot_scenario import recorded_scenario

            task, source_report, directory = recorded_scenario(
                args.pilot_corpus, args.pilot_case, split=args.pilot_split
            )
        else:
            task = initial_tower_task(mirror=args.mirror)
            if args.missing_target is not None:
                task = task.without_target(args.missing_target)
        assumptions: dict[str, Any] = (
            {
                "mass_scale": campaign_case.mass_scale,
                "effective_friction": campaign_case.friction,
                "table_lighting": campaign_case.appearance,
            }
            if campaign_case
            else (
                source_report.get("simulation_assumptions", {"table_lighting": "light_wood"})
                if source_report
                else {}
            )
        )
        env = AssemblyEnvironment(
            args.cache_dir.resolve(),
            placements=task.placements,
            assembly_xy_m=None if task.build_region_xy_m is not None else task.assembly_xy_m,
            mass_scale=assumptions.get("mass_scale", 1.0),
            friction=tuple(assumptions.get("effective_friction", (0.6, 0.5))),
            appearance=assumptions.get("table_lighting", "legacy"),
            parked_start=campaign_case is not None,
        )
        env.step(400)
        connection.connect(str(args.socket))
        report.update(task=asdict(task), robot_joint_names=env.robot.data.joint_names)
        names = (*ARM_JOINTS, *JAW_JOINTS)
        ids = [env.robot.data.joint_names.index(n) for n in names]
        limits = env.robot.data.joint_pos_limits.torch[0, ids].cpu().numpy()
        if campaign_case and campaign_case.category == "single_block":
            from vla_lego.isaac.pick_evaluation import PickPlaceEvaluator
            from vla_lego.isaac.task import SingleBlockTask

            assert isinstance(task, SingleBlockTask)
            evaluator: PickPlaceEvaluator | TowerEvaluator = PickPlaceEvaluator(
                task, env.contact_robot_paths
            )
        else:
            evaluator = TowerEvaluator(task, env.contact_robot_paths)
        tools = None
        if task.build_region_xy_m is not None:
            from vla_lego.isaac.control import ToolController

            tools = ToolController(env)

        def update_evaluation():
            truth = [env.task_truth(i) for i in range(len(env.blocks))]
            positions = (
                {side: tools.pose(side)[0, :3].tolist() for side in ("left", "right")}
                if tools is not None
                else None
            )
            evaluator.update(env.time, truth, tool_positions=positions)
            return truth

        update_evaluation()
        report.update(
            pilot_case=args.pilot_case,
            pilot_split=args.pilot_split if args.pilot_case else None,
            teacher_prefix_ticks=0,
            campaign_case=None if campaign_case is None else campaign_case.row,
            simulation_assumptions=assumptions,
        )
        if args.teacher_start_stage:
            from vla_lego.isaac.pilot_scenario import replay_demonstration_handoff

            assert directory is not None

            def handoff_tick(prefix_tick):
                if prefix_tick % 100 == 0:
                    send_message(connection, {"prepare": True})
                    receive_message(connection)
                update_evaluation()

            report.update(
                replay_demonstration_handoff(env, directory, args.teacher_start_stage, handoff_tick)
            )
            report["ordered_levels_at_handoff"] = evaluator.result()["max_ordered_supported_levels"]
        elif source_report and (
            campaign_case.category == "recovery"
            if campaign_case
            else source_report["pilot_case"]["category"] == "recovery"
        ):
            from vla_lego.isaac.pilot_scenario import replay_recovery

            assert directory is not None

            def recovery_tick(prefix_tick):
                if prefix_tick % 100 == 0:
                    send_message(connection, {"prepare": True})
                    receive_message(connection)
                update_evaluation()

            report.update(replay_recovery(env, task, source_report, directory, recovery_tick))
            report["ordered_levels_at_handoff"] = evaluator.result()["max_ordered_supported_levels"]
        report["teacher_commands"] = report["teacher_prefix_ticks"] > 0
        report["autonomous_from_start"] = not report["teacher_commands"]
        chunk: list = []
        inference_seconds: list[float] = []
        representation = None
        decision_names: list[str] = []
        categorical_decision = None
        separate_decisions = None
        terminal, executed, clipped_ticks = "budget", 0, 0
        report["autonomy_started"] = True
        budget = (
            args.ticks
            - report["teacher_prefix_ticks"]
            - (40 if args.pilot_case or campaign_case else 0)
        )
        if budget < 1:
            raise ValueError("The recovery prefix consumes the declared whole-case budget")
        report["policy_tick_budget"] = budget
        with (args.output_dir / "trace.jsonl").open("w") as trace:
            for tick in range(budget):
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
                            "observation_time_s": env.time,
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
                    categorical_decision = response.get("task_decision")
                    has_separate = categorical_decision is not None
                    if separate_decisions is not None and separate_decisions != has_separate:
                        raise ValueError("The learned decision module changed inside the episode")
                    separate_decisions = has_separate
                    if categorical_decision is not None:
                        from vla_lego.pretrained.task_decisions import CLASSES

                        probabilities = categorical_decision["probabilities"]
                        if (
                            decision_names
                            or categorical_decision["classes"] != list(CLASSES)
                            or categorical_decision["value"] not in CLASSES
                            or len(probabilities) != 3
                            or not np.isfinite(probabilities).all()
                            or min(probabilities) < 0
                            or max(probabilities) > 1
                            or abs(sum(probabilities) - 1) > 1e-4
                        ):
                            raise ValueError("Invalid separate learned task-decision message")
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
                decision = (
                    categorical_decision["value"]
                    if categorical_decision is not None
                    else decode_decision(action[16:]) if decision_names else "continue"
                )
                requested = action[:16].copy()
                if representation == "measured_arm_offset":
                    requested[:14] += env.robot.data.joint_pos.torch[0, ids[:14]].cpu().numpy()
                bounded = np.clip(requested, limits[:, 0], limits[:, 1])
                clipped_ticks += int(np.any(requested != bounded))
                if decision == "continue":
                    env.set_joint_targets(dict(zip(names, bounded.tolist(), strict=True)))
                    env.step(10)
                    executed += 1
                    truth = update_evaluation()
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
                            "categorical_decision": categorical_decision,
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
                truth = update_evaluation()
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
            and all(
                b["nonrobot_contact_force_n"] > 1e-6
                or any(force[2] > 1e-6 for force in b["block_contact_forces_n"])
                for row in hold
                for b in row
            )
            and len(evaluation["dwell_corner_displacement_m"]) == len(env.blocks)
            and max(evaluation["dwell_corner_displacement_m"]) <= 0.003
        )
        campaign_stop = campaign_case is not None and campaign_case.category == "stop"
        if campaign_stop:
            from vla_lego.isaac.campaign_faults import stop_evidence

            justified_stop = bool(
                terminal == "abstain"
                and all(stop_evidence(task, campaign_case, row)["passed"] for row in hold)
                and max(evaluation["dwell_corner_displacement_m"]) <= 0.003
            )
        passed = evaluation["passed"] and terminal == "complete"
        report.update(
            evaluation=evaluation,
            tower_constructed=evaluation["passed"] and len(task.targets) == 5,
            single_block_completed=evaluation["passed"] and len(task.targets) == 1,
            passed=passed,
            justified_stop=justified_stop,
            case_passed=(
                justified_stop if campaign_stop or not task.requested_pieces_available else passed
            ),
            terminal_decision=terminal,
            learned_completion_signal=bool(decision_names) or bool(separate_decisions),
            separate_categorical_decisions=separate_decisions,
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
