"""Contact-only five-block reference with overlapping stabilize/place actions."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--rgb-storage", choices=("png", "ffv1"), default="png")
    parser.add_argument("--family")
    parser.add_argument("--split", choices=("train", "development"))
    parser.add_argument("--event-settling", action="store_true")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--paired-cases", type=Path)
    parser.add_argument("--campaign-cases", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--policy-source-run", type=Path)
    parser.add_argument("--policy-prefix-ticks", type=int)
    parser.add_argument("--preserve-first-placement", action="store_true")
    parser.add_argument("--policy-socket", type=Path)
    parser.add_argument("--policy-reference", type=Path)
    parser.add_argument("--policy-window-ticks", type=int, default=1800)
    parser.add_argument(
        "--motion-measurement",
        choices=("pose_dwell", "solver_velocity", "pose_step_peaks"),
        default="pose_dwell",
    )
    args = parser.parse_args()
    selectors = (args.cases, args.paired_cases, args.campaign_cases)
    if sum(bool(p) for p in selectors) > 1 or any(selectors) != bool(args.case):
        parser.error("Provide one case file with --case")
    if args.record and (not args.family or not args.split):
        parser.error("--record requires --family and --split before collection")
    if bool(args.policy_source_run) != bool(args.policy_prefix_ticks):
        parser.error("Policy correction requires both source run and applied prefix length")
    if args.policy_source_run and (
        not args.record or not (args.cases or args.campaign_cases) or args.split != "train"
    ):
        parser.error("Own-policy corrections require a recorded training case")
    if args.preserve_first_placement and not args.policy_source_run:
        parser.error("Preserving a placement requires an actual training-policy prefix")
    if bool(args.policy_socket) != bool(args.policy_reference) or args.policy_window_ticks < 1:
        parser.error("An in-place policy window requires socket/reference and a positive budget")
    if args.policy_socket and (
        not args.campaign_cases
        or not args.record
        or args.split != "train"
        or args.policy_source_run
    ):
        parser.error("In-place corrections require a separate recorded TRAIN campaign attempt")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict = {
        "kind": "physical_tower_reference",
        "learned": False,
        "passed": False,
        "qualification": False,
        "motion_measurement": args.motion_measurement,
        "settling": "measured_velocity" if args.event_settling else "fixed_wait",
        "mirrored": args.mirror,
    }
    app = None
    reference = None
    recorder = None
    policy_intervention = None
    cooperation: list[dict] = []
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        from PIL import Image

        from vla_lego.isaac.environment import TABLE_TOP, AssemblyEnvironment
        from vla_lego.isaac.evaluation import COOPERATION_S, PERIOD_S, TowerEvaluator, _corners
        from vla_lego.isaac.recording import EpisodeRecorder
        from vla_lego.isaac.reference import PhysicalReference
        from vla_lego.isaac.task import initial_tower_task

        pilot_case = None
        paired_task = None
        collection_condition = "marker_free_continuous_site_v1"
        if args.paired_cases:
            from vla_lego.isaac.pilot import load_paired_cases

            pilot_case, paired_task = next(
                (c, t) for c, t in load_paired_cases(args.paired_cases) if c.name == args.case
            )
            collection_condition = "marker_free_paired_instruction_v1"
        if args.campaign_cases:
            from vla_lego.isaac.campaign import CONDITION, load_campaign

            pilot_case = next(c for c in load_campaign(args.campaign_cases) if c.name == args.case)
            paired_task = pilot_case.task()
            collection_condition = CONDITION
            if pilot_case.row.get("correction_origin") == "policy" and (
                not args.policy_socket
                or pilot_case.row["mechanism"]
                not in (
                    "interrupted_progress",
                    "slip_drop",
                    "missed_grasp",
                    "wrong_piece",
                    "placement",
                )
                or (
                    pilot_case.row["mechanism"] in ("slip_drop", "missed_grasp", "wrong_piece")
                    and pilot_case.error_level >= pilot_case.cooperative_level
                )
            ):
                raise ValueError(
                    "Policy-origin recovery needs its actual supported in-place mechanism"
                )
            if args.policy_socket and pilot_case.row.get("correction_origin") != "policy":
                raise ValueError("In-place policy collection cannot relabel a teacher-origin case")
        if args.cases:
            from vla_lego.isaac.pilot import load_cases

            pilot_case = next(c for c in load_cases(args.cases) if c.name == args.case)
            if pilot_case.category == "stop":
                raise ValueError("Stop cases use the stop collector.")
        task = (
            paired_task
            if paired_task is not None
            else pilot_case.task() if pilot_case else initial_tower_task(mirror=args.mirror)
        )
        if args.policy_source_run:
            source = json.loads((args.policy_source_run / "report.json").read_text())
            if source.get("pilot_case") != args.case or source.get("pilot_split") != "train":
                raise ValueError("Correction source does not belong to the requested training case")
            if task.build_region_xy_m is not None:
                # Restore the recorded teacher choice; the site is unmarked and
                # is neither a policy input nor a new physical scene object.
                task = replace(task, assembly_xy_m=tuple(source["task"]["assembly_xy_m"]))
            if json.loads(json.dumps(asdict(task))) != source["task"]:
                raise ValueError("Generated correction scene differs from the actual policy scene")
            if not task.requested_pieces_available:
                raise ValueError("A continuation demonstration requires a completable inventory")
        single_block = len(task.targets) == 1
        campaign_stop = bool(args.campaign_cases and pilot_case and pilot_case.category == "stop")
        cooperative_level = (
            len(task.targets) + 1
            if single_block or campaign_stop
            else pilot_case.cooperative_level if pilot_case else 3
        )
        report["task_kind"] = "single_block" if single_block else "tower"
        if single_block or campaign_stop:
            report["task_kind"] = "single_block" if single_block else "justified_stop"
            report["kind"] = f"physical_{report['task_kind']}_reference"
        stabilizer_quaternion = [0.0, 0.0, 0.0, 1.0]
        stabilizer_clear_quaternion = [
            0.0,
            -math.sin(math.radians(10)),
            0.0,
            math.cos(math.radians(10)),
        ]
        stabilizer_clear_z = (
            TABLE_TOP + sum(t.dimensions_m[2] for t in task.targets[: cooperative_level + 1]) + 0.04
        )
        roles: tuple[tuple[str, int], ...] = (
            ("left", 0),
            ("right", 180),
            ("left", -90),
            ("right", 90),
            ("right", 90),
        )
        if args.mirror:
            roles = tuple(("right" if side == "left" else "left", -yaw) for side, yaw in roles)
        if pilot_case:
            if (args.family, args.split) != (pilot_case.row["family"], pilot_case.row["split"]):
                raise ValueError("Collection family/split differs from the approved case.")
            roles = tuple(
                next(
                    (
                        ("left" if p.xy_m[1] > 0 else "right", -90 if p.xy_m[1] > 0 else 90)
                        for p in task.placements
                        if p.spec == target
                    ),
                    ("left", -90),
                )
                for target in task.targets
            )
        if single_block or campaign_stop:
            placer_side = roles[0][0]
            stabilizer_side = "right" if placer_side == "left" else "left"
        else:
            stabilizer_side, placer_side = (
                roles[cooperative_level - 1][0],
                roles[cooperative_level][0],
            )
        stabilizer_y = 0.20 if stabilizer_side == "left" else -0.20
        stabilizer_exit_y = 0.40 if stabilizer_side == "left" else -0.40
        stabilizer_exit_path = None
        stabilizer_level = 0 if single_block or campaign_stop else cooperative_level - 1
        if pilot_case and not (single_block or campaign_stop):
            from vla_lego.isaac.stabilizer_planning import stabilized_level

            stabilizer_level = stabilized_level(task, cooperative_level)
            report["stabilized_layer"] = stabilizer_level + 1
        stabilizer_grip_floor: float | None = None
        env = AssemblyEnvironment(
            args.cache_dir.resolve(),
            placements=task.placements,
            assembly_xy_m=None if task.build_region_xy_m is not None else task.assembly_xy_m,
            measure_pose_motion=args.motion_measurement == "pose_step_peaks",
            mass_scale=pilot_case.mass_scale if pilot_case else 1.0,
            friction=pilot_case.friction if pilot_case else (0.6, 0.5),
            appearance=pilot_case.appearance if pilot_case else "legacy",
            parked_start=bool(args.campaign_cases),
        )
        env.step(400)
        if args.policy_socket:
            from vla_lego.isaac.policy_continuation import PolicyIntervention

            policy_intervention = PolicyIntervention(
                args.policy_socket, args.policy_reference, task.instruction
            )
        correction_source = None
        reference_orientations = None
        if args.policy_source_run:
            from vla_lego.isaac.control import ToolController
            from vla_lego.isaac.policy_replay import replay_prefix

            controller = ToolController(env)
            reference_orientations = {
                side: controller.pose(side)[0, 3:].clone() for side in ("left", "right")
            }
            correction_source = replay_prefix(env, args.policy_source_run, args.policy_prefix_ticks)
            if not correction_source["exact_native_parity"]:
                raise RuntimeError("Policy-error replay does not reproduce its recorded state")
            correction_source["source_terminal_decision"] = source["terminal_decision"]
            if args.preserve_first_placement:
                from itertools import islice

                from vla_lego.isaac.policy_replay import preserved_first_placement

                with (args.policy_source_run / "trace.jsonl").open() as stream:
                    correction_source["first_placement"] = preserved_first_placement(
                        task,
                        (json.loads(line) for line in islice(stream, args.policy_prefix_ticks)),
                    )
            report["correction_source"] = correction_source
        if args.record:
            recorder = EpisodeRecorder(
                args.output_dir / "recording",
                task.instruction,
                family=args.family,
                split=args.split,
                task_decisions=True,
                collection_case=pilot_case.row if pilot_case else None,
                collection_condition=collection_condition if pilot_case else None,
                correction_source=correction_source,
                png_compress_level=6 if args.campaign_cases else 1,
                min_free_bytes=512 * 1024**2 if args.campaign_cases else 0,
                image_storage=args.rgb_storage,
            )
            env.recorder = recorder
        report["task"] = asdict(task)
        if pilot_case:
            report["pilot_case"] = pilot_case.row
            report["collection_condition"] = collection_condition
            report["simulation_assumptions"] = {
                "mass_scale": pilot_case.mass_scale,
                "effective_friction": pilot_case.friction,
                "table_lighting": pilot_case.appearance,
                "measured_print_properties": False,
            }
        report["robot_joint_names"] = env.robot.data.joint_names
        report["physics_solver_iterations"] = list(env.solver_iterations)
        indices = [
            next((i for i, p in enumerate(task.placements) if p.spec == t), -1)
            for t in task.targets
        ]
        contact_ids = {
            side: [i for i, path in enumerate(env.contact_robot_paths) if f"{side}_WSG32" in path]
            for side in ("left", "right")
        }
        if not all(contact_ids.values()):
            raise RuntimeError("Cannot identify actual WSG bodies for cooperation evidence.")
        if single_block:
            from vla_lego.isaac.pick_evaluation import PickPlaceEvaluator
            from vla_lego.isaac.task import SingleBlockTask

            assert isinstance(task, SingleBlockTask)
            evaluator: Any
            evaluator = PickPlaceEvaluator(task, env.contact_robot_paths)
        else:
            evaluator = TowerEvaluator(
                task, env.contact_robot_paths, motion_measurement=args.motion_measurement
            )

        def sample_cooperation(stage: str, truth: list[dict]) -> None:
            assert reference is not None
            if policy_intervention is not None:
                policy_intervention.keepalive(env.time)
            evaluator.update(
                env.time,
                truth,
                tool_positions={
                    side: reference.ctrl.pose(side)[0, :3].tolist() for side in ("left", "right")
                },
            )
            if args.campaign_cases and recorder is not None and not campaign_stop:
                recorder.task_decision = "complete" if evaluator.result()["passed"] else "continue"
            if (
                stabilizer_grip_floor is not None
                and not (policy_intervention is not None and policy_intervention.active)
                and stabilizer_side in reference.hold_targets
                and not any(
                    evaluator._gripping(truth[i], stabilizer_side)
                    and sum(
                        math.sqrt(sum(v * v for v in truth[i]["robot_contact_forces_n"][jaw]))
                        for jaw in evaluator.jaws[stabilizer_side]
                    )
                    >= 1.0
                    for i in indices[:cooperative_level]
                )
            ):
                # Maintain a 1 N grip margin with normal, force-limited jaw commands.
                name = f"{stabilizer_side}_WSG32_JAW_OPENING"
                previous = float(env.target[0, env.robot.data.joint_names.index(name)])
                env.set_joint_targets({name: max(stabilizer_grip_floor, previous - 0.0005)})
            if not stage.startswith(f"level{cooperative_level}_"):
                return
            lower, upper = truth[indices[stabilizer_level]], truth[indices[cooperative_level]]
            stabilizer = sum(
                sum(v * v for v in lower["robot_contact_forces_n"][i]) ** 0.5
                for i in contact_ids[stabilizer_side]
            )
            placer = sum(
                sum(v * v for v in upper["robot_contact_forces_n"][i]) ** 0.5
                for i in contact_ids[placer_side]
            )
            support = upper["block_contact_forces_n"][indices[cooperative_level - 1]][2]
            if stabilizer > 0.5 and placer > 0.5 and support > 0.05:
                cooperation.append(
                    {
                        "time_s": env.time,
                        "stage": stage,
                        "stabilizer_n": stabilizer,
                        "placer_n": placer,
                        "support_n": support,
                    }
                )

        with (args.output_dir / "trace.jsonl").open("w") as trace:
            reference = PhysicalReference(
                env,
                trace,
                args.output_dir,
                sample_cooperation,
                event_settling=args.event_settling,
            )
            report["teacher_rotation_residual_scale"] = reference.ctrl.rotation_scale
            if args.campaign_cases:
                from isaaclab.utils.math import quat_from_matrix

                # A resting reset must not redefine the source grasp frame.
                reference.initial_orientations = {
                    side: quat_from_matrix(
                        solver.forward(env.target.new_zeros((1, 7)))[0][:, :3, :3]
                    )[0]
                    for side, solver in reference.solvers.items()
                }
            if (
                args.campaign_cases
                and not single_block
                and not campaign_stop
                and not args.policy_source_run
            ):
                from vla_lego.isaac.assembly_planning import choose_build_site

                selected_xy, planning = choose_build_site(
                    reference, task, indices, roles, cooperative_level
                )
                task = replace(task, assembly_xy_m=tuple(selected_xy))
                evaluator.task = task
                report["task"] = asdict(task)
                report["teacher_build_site"] = planning
            if reference_orientations is not None:
                # Preserve the shared initial grasp convention, not the failed policy's wrist pose.
                reference.initial_orientations = reference_orientations

            def verify_support(
                last_level: int, stage: str, *, held_side: str | None = None
            ) -> None:
                expected_bottom = TABLE_TOP
                for prior in range(last_level + 1):
                    state = env.task_truth(indices[prior])
                    expected_z = expected_bottom + task.targets[prior].dimensions_m[2] / 2
                    support_n = (
                        state["nonrobot_contact_force_n"]
                        if prior == 0
                        else state["block_contact_forces_n"][indices[prior - 1]][2]
                    )
                    supported = support_n > 1e-6 or (
                        held_side is not None and evaluator._gripping(state, held_side)
                    )
                    if abs(state["block_pose_w"][2] - expected_z) >= 0.003 or not supported:
                        raise RuntimeError(
                            f"Incorrect physical support at level {prior} after {stage}"
                        )
                    expected_bottom += task.targets[prior].dimensions_m[2]

            def engage_stabilizer():
                nonlocal stabilizer_quaternion, stabilizer_clear_quaternion, stabilizer_clear_z
                nonlocal stabilizer_grip_floor, stabilizer_exit_path
                state = env.task_truth(indices[stabilizer_level])
                lower = state["block_pose_w"][:3]
                nominal_width = task.targets[stabilizer_level].dimensions_m[1]
                width = nominal_width
                stabilizer_pose = [lower[0] - 0.01, lower[1], lower[2]]
                approach = (
                    ("high", [lower[0] - 0.01, stabilizer_y, 1.02]),
                    ("center", [lower[0] - 0.01, lower[1], 1.02]),
                    ("above", [lower[0] - 0.01, lower[1], lower[2] + 0.06]),
                    ("grasp", stabilizer_pose),
                )
                withdraw = [*stabilizer_pose[:2], stabilizer_pose[2] + 0.06]
                stabilizer_exit_path = [
                    (
                        "exit",
                        [withdraw[0], stabilizer_exit_y, stabilizer_clear_z],
                        stabilizer_clear_quaternion,
                    )
                ]
                if pilot_case:
                    from vla_lego.isaac.stabilizer_planning import plan_stabilizer

                    selected, selected_cost, failures = plan_stabilizer(
                        reference, task, indices, cooperative_level, stabilizer_side
                    )
                    report["stabilizer_planning_failures"] = failures
                    if selected is None:
                        raise RuntimeError(
                            "No collision-free full stabilizer approach at the actual tower"
                        )
                    (
                        approach,
                        stabilizer_quaternion,
                        stabilizer_pose,
                        width,
                        stabilizer_clear_quaternion,
                        stabilizer_clear_z,
                        withdraw,
                        stabilizer_exit_path,
                    ) = selected
                    report["stabilizer_grasp_joint_travel_cost"] = selected_cost
                for name, xyz in approach:
                    reference.move(
                        f"stabilizer_{name}",
                        stabilizer_side,
                        xyz,
                        quaternion_xyzw=(
                            [0.0, 0, 0, 1.0]
                            if pilot_case and name == "high"
                            else stabilizer_quaternion
                        ),
                        # Change orientation in the clear high pose, then descend straight.
                        linear=pilot_case is not None and name == "grasp",
                    )
                if pilot_case:
                    reference.hold_targets[stabilizer_side] = reference.ctrl.pose(stabilizer_side)[
                        0
                    ].tolist()
                    _, holding_ids, _ = reference.ctrl.frames[stabilizer_side]
                    reference.hold_postures[stabilizer_side] = env.robot.data.joint_pos.torch[
                        0, holding_ids
                    ].tolist()
                for closure in range(18):
                    reference.grip(
                        "stabilizer_close" if closure == 0 else "stabilizer_acquire_contact",
                        stabilizer_side,
                        max(nominal_width, width - closure * 0.002),
                    )
                    reference.wait(3, "stabilizer_confirm_contact")
                    if any(
                        evaluator._gripping(env.task_truth(i), stabilizer_side)
                        for i in indices[:cooperative_level]
                    ):
                        break
                    verify_support(cooperative_level - 1, "stabilizer acquisition")
                    if not pilot_case or width - closure * 0.002 <= nominal_width:
                        raise RuntimeError("No actual bilateral stabilizer contact")
                if pilot_case:
                    # Freeze the acquired physical grasp instead of creeping toward a nominal pose.
                    reference.hold_targets[stabilizer_side] = reference.ctrl.pose(stabilizer_side)[
                        0
                    ].tolist()
                    reference.hold_postures[stabilizer_side] = env.robot.data.joint_pos.torch[
                        0, holding_ids
                    ].tolist()
                    jaw = env.robot.data.joint_names.index(f"{stabilizer_side}_WSG32_JAW_OPENING")
                    stabilizer_grip_floor = float(env.robot.data.joint_pos_limits.torch[0, jaw, 0])
                return withdraw

            def interrupt_coordination(index, side, withdraw):
                assert recorder is not None
                if not any(
                    evaluator._gripping(env.task_truth(i), stabilizer_side)
                    for i in indices[:cooperative_level]
                ) or not evaluator._gripping(env.task_truth(index), side):
                    raise RuntimeError("Coordination interruption needs two actual active grasps")
                _, holding_ids, _ = reference.ctrl.frames[side]
                reference.hold_targets[side] = reference.ctrl.pose(side)[0].tolist()
                reference.hold_postures[side] = env.robot.data.joint_pos.torch[
                    0, holding_ids
                ].tolist()
                recorder.positive_imitation = False
                reference.hold_targets.pop(stabilizer_side)
                reference.hold_postures.pop(stabilizer_side)
                reference.grip("fault_interrupted_coordination_release", stabilizer_side, None)
                reference.move(
                    "fault_interrupted_coordination_retract",
                    stabilizer_side,
                    withdraw,
                    quaternion_xyzw=stabilizer_quaternion,
                    linear=True,
                )
                reference.move(
                    "fault_interrupted_coordination_clear",
                    stabilizer_side,
                    [*withdraw[:2], stabilizer_clear_z],
                    quaternion_xyzw=stabilizer_clear_quaternion,
                )
                for name, xyz, rotation in stabilizer_exit_path:
                    reference.move(
                        f"fault_interrupted_coordination_{name}",
                        stabilizer_side,
                        xyz,
                        quaternion_xyzw=rotation,
                    )
                verify_support(cooperative_level - 1, "interrupted coordination")
                if any(
                    evaluator._gripping(env.task_truth(i), stabilizer_side)
                    for i in range(len(env.blocks))
                ) or not evaluator._gripping(env.task_truth(index), side):
                    raise RuntimeError("The intended loss of coordination was not established")
                recorder.positive_imitation = True
                return {
                    "type": "interrupted_progress",
                    "mechanism": "premature_stabilizer_withdrawal",
                    "teacher_induced": True,
                    "physically_verified": True,
                    "completed_levels": cooperative_level,
                    "stabilizer": stabilizer_side,
                    "placer": side,
                    "stabilizer_grasp_before": True,
                    "stabilizer_grasp_at_intervention": False,
                    "placer_grasp_at_intervention": True,
                    "error_end_s": env.time,
                }

            for side in ("left", "right"):
                reference.park(side)
            env.capture()
            Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "initial.png")
            env.stage.Export(str(args.output_dir / "scene.usda"))
            start_level = 1 if args.preserve_first_placement else 0
            bottom = TABLE_TOP + sum(t.dimensions_m[2] for t in task.targets[:start_level])
            if start_level:
                verify_support(0, "preserved policy placement and park")
            stabilizer_withdraw = None
            build_xy = task.assembly_xy_m
            # Per-piece yaw is a teacher planning choice, never a policy observation.
            plan = list(zip(task.targets, indices, roles, strict=True))
            remaining = list(enumerate(plan))
            displaced_sources = set()
            while remaining:
                level, (target, index, (side, base_yaw)) = remaining.pop(0)
                if campaign_stop and pilot_case and level == pilot_case.error_level:
                    break
                if level < start_level:
                    continue
                if (
                    policy_intervention is not None
                    and pilot_case is not None
                    and pilot_case.row["mechanism"] == "interrupted_progress"
                    and level == pilot_case.error_level + 1
                    and "fault" not in report
                ):
                    assert recorder is not None
                    report["fault"] = policy_intervention.interrupted_progress(
                        reference,
                        pilot_case,
                        task,
                        completed_levels=level,
                        ticks=args.policy_window_ticks,
                    )
                    report["correction_source"] = recorder.correction_source
                    start_level = report["fault"]["completed_levels"]
                    bottom = TABLE_TOP + sum(t.dimensions_m[2] for t in task.targets[:start_level])
                    verify_support(start_level - 1, "actual policy interruption")
                    if level < start_level:
                        continue
                if level and pilot_case:
                    build_xy = tuple(env.task_truth(indices[0])["block_pose_w"][:2])
                prefix = f"level{level}"
                yaw = float(base_yaw)
                if (
                    pilot_case
                    and pilot_case.category == "recovery"
                    and level == pilot_case.error_level
                    and pilot_case.row["error"] != "physically consequential incomplete placement"
                    and pilot_case.row["error"] not in ("slip_drop", "interrupted_progress")
                ):
                    if policy_intervention is not None:
                        assert recorder is not None
                        method = (
                            policy_intervention.missed_grasp
                            if pilot_case.row["mechanism"] == "missed_grasp"
                            else policy_intervention.wrong_piece
                        )
                        report["fault"] = method(
                            reference, pilot_case, task, index, ticks=args.policy_window_ticks
                        )
                        report["correction_source"] = recorder.correction_source
                        if report["fault"]["type"] == "wrong_piece":
                            from vla_lego.isaac.campaign_faults import return_wrong_piece

                            return_wrong_piece(reference, task, report["fault"])
                        for hand in ("left", "right"):
                            reference.grip("pick_correction_open", hand, None)
                        idle_hand = "right" if side == "left" else "left"
                        report["policy_hand_retirement"] = {
                            "side": idle_hand,
                            "minimum_predicted_jaw_clearance_m": reference.park(
                                idle_hand, check_clearance=True
                            ),
                        }
                    else:
                        from vla_lego.isaac.pilot_faults import pick_fault

                        report["fault"] = pick_fault(reference, pilot_case, task, index)
                    verify_support(level - 1, "pick fault correction")
                if index in displaced_sources:
                    from vla_lego.isaac.campaign_faults import upright_released_prism

                    upright_released_prism(reference, task, index, side)
                if level == cooperative_level:
                    stabilizer_withdraw = engage_stabilizer()
                block = env.task_truth(index)["block_pose_w"][:3]
                grasp_height = (
                    reference.grasp_height(side, target.dimensions_m[2], yaw)
                    if pilot_case
                    else 0.06
                )
                if pilot_case:
                    report.setdefault("source_grasp_heights_m", {})[prefix] = grasp_height
                    if index in displaced_sources:
                        from vla_lego.isaac.campaign_faults import resting_yaw

                        yaw += resting_yaw(env.task_truth(index)["block_pose_w"], target)
                    else:
                        yaw += task.placements[index].yaw_deg
                    from vla_lego.isaac.assembly_planning import pick_route

                    yaw = reference.grasp_yaw(
                        side,
                        yaw,
                        pick_route(
                            task,
                            level,
                            block,
                            build_xy,
                            side,
                            cooperative_level,
                            grasp_height=grasp_height,
                        ),
                        object_index=index,
                    )
                for stage, xyz in (
                    ("edge", [0.40, 0.40 if side == "left" else -0.40, 1.05]),
                    (
                        "source_high",
                        [block[0], block[1], 1.02],
                    ),
                    ("approach", [block[0], block[1], block[2] + grasp_height + 0.04]),
                    ("descend", [block[0], block[1], block[2] + grasp_height]),
                ):
                    reference.move(
                        f"{prefix}_{stage}",
                        side,
                        xyz,
                        yaw,
                        linear=pilot_case is not None and stage in ("approach", "descend"),
                    )
                    if stage == "source_high":
                        # A policy prefix may leave the jaws closed before the next pick.
                        reference.grip(f"{prefix}_open", side, None)
                reference.grip(f"{prefix}_grip", side, target.dimensions_m[1])
                release_z = bottom + target.dimensions_m[2] / 2 + grasp_height
                lift_z = block[2] + grasp_height + 0.10
                transit_z = release_z + 0.025
                lift_xyz = [block[0], block[1], lift_z]
                lift_plan = reference.lift_orientation(side, lift_xyz, yaw) if pilot_case else None
                if lift_plan:
                    report.setdefault("loaded_lift_replans", {})[prefix] = lift_plan
                reference.move(
                    f"{prefix}_lift",
                    side,
                    lift_xyz,
                    yaw,
                    quaternion_xyzw=lift_plan["quaternion_xyzw"] if lift_plan else None,
                    linear=pilot_case is not None,
                    linear_step_m=0.02,
                )
                if env.task_truth(index)["block_pose_w"][2] < block[2] + 0.07:
                    raise RuntimeError(f"No verified contact-only lift at {prefix}")
                if lift_plan:
                    reference.move(f"{prefix}_lift_align", side, lift_xyz, yaw)
                if (
                    args.campaign_cases
                    and pilot_case
                    and pilot_case.category == "recovery"
                    and pilot_case.row["mechanism"] == "interrupted_progress"
                    and pilot_case.row.get("correction_origin") == "teacher"
                    and level == pilot_case.error_level == cooperative_level
                ):
                    report["fault"] = interrupt_coordination(index, side, stabilizer_withdraw)
                    stabilizer_withdraw = engage_stabilizer()
                    report["fault"]["stabilization_restored"] = any(
                        evaluator._gripping(env.task_truth(i), stabilizer_side)
                        for i in indices[:cooperative_level]
                    )
                    if not report["fault"]["stabilization_restored"]:
                        raise RuntimeError(
                            "Interrupted stabilizer grasp was not physically restored"
                        )
                    reference.hold_targets.pop(side)
                    reference.hold_postures.pop(side)
                    assert recorder is not None
                    recorder.fault_verified = True
                if (
                    args.campaign_cases
                    and pilot_case
                    and pilot_case.category == "recovery"
                    and level == pilot_case.error_level
                    and pilot_case.row["mechanism"] == "slip_drop"
                ):
                    from vla_lego.isaac.campaign_faults import drop_and_recover, recover_dropped

                    if policy_intervention is not None:
                        assert recorder is not None
                        report["fault"] = policy_intervention.slip_drop(
                            reference, pilot_case, task, index, ticks=args.policy_window_ticks
                        )
                        report["correction_source"] = recorder.correction_source
                        verify_support(level - 1, "actual policy drop")
                        yaw = recover_dropped(reference, task, index, side, yaw, replan=True)
                        block = report["fault"]["recovery_source_pose_w"][:3]
                        lift_z = block[2] + grasp_height + 0.10
                    else:
                        report["fault"] = drop_and_recover(
                            reference, pilot_case, task, index, side, yaw
                        )
                carry_z = transit_z
                if pilot_case:
                    held = env.task_truth(index)["block_pose_w"]
                    top = (
                        max(
                            c[2]
                            for c in _corners(
                                task.targets[level - 1].dimensions_m,
                                env.task_truth(indices[level - 1])["block_pose_w"],
                            )
                        )
                        if level
                        else TABLE_TOP
                    )
                    lowest = min(c[2] for c in _corners(target.dimensions_m, held))
                    carry_z = max(
                        lift_z,
                        transit_z,
                        float(reference.ctrl.pose(side)[0, 2]) + top + 0.025 - lowest,
                    )
                    if carry_z > lift_z:
                        # Gain payload clearance while still away from the tower.
                        reference.move(
                            f"{prefix}_carry_raise",
                            side,
                            [block[0], block[1], carry_z],
                            yaw,
                            linear=True,
                        )
                if level >= cooperative_level:
                    # Gain clearance away from the existing tower before the
                    # carried object's footprint crosses into the assembly area.
                    for waypoint, y in enumerate((0.20, 0.10)):
                        reference.move(
                            f"{prefix}_carry_clearance_{waypoint}",
                            side,
                            [build_xy[0], build_xy[1] + (y if side == "left" else -y), carry_z],
                            yaw,
                            linear=pilot_case is not None,
                            linear_step_m=0.02,
                        )
                if pilot_case:
                    if level:
                        build_xy = tuple(env.task_truth(indices[level - 1])["block_pose_w"][:2])
                    reference.move(
                        f"{prefix}_transport_high",
                        side,
                        [*build_xy, carry_z],
                        yaw,
                        linear=True,
                        linear_step_m=0.02,
                    )
                reference.move(
                    f"{prefix}_transport",
                    side,
                    [*build_xy, transit_z],
                    yaw,
                    linear=pilot_case is not None,
                )
                if pilot_case:
                    # Use the physically held grasp offset, not its nominal 60 mm value.
                    held_z = env.task_truth(index)["block_pose_w"][2]
                    tool_z = float(reference.ctrl.pose(side)[0, 2])
                    release_z = bottom + target.dimensions_m[2] / 2 + tool_z - held_z
                if (
                    pilot_case
                    and pilot_case.category == "recovery"
                    and level == pilot_case.error_level
                    and pilot_case.row["error"] == "physically consequential incomplete placement"
                    and "fault" not in report
                ):
                    if policy_intervention is not None:
                        from vla_lego.isaac.campaign_faults import recover_held_placement

                        assert recorder is not None
                        report["fault"] = policy_intervention.placement(
                            reference,
                            pilot_case,
                            task,
                            index,
                            side,
                            stabilizer_side=stabilizer_side if level == cooperative_level else None,
                            ticks=args.policy_window_ticks,
                        )
                        report["correction_source"] = recorder.correction_source
                        recovery_mode = report["fault"].get("recovery_mode")
                        if recovery_mode == "return_held_then_rebuild":
                            from vla_lego.isaac.campaign_faults import return_held_piece

                            if level == cooperative_level:
                                # Retain the measured hand pose, not the old policy prefix target.
                                _, joint_ids, _ = reference.ctrl.frames[stabilizer_side]
                                reference.hold_targets[stabilizer_side] = reference.ctrl.pose(
                                    stabilizer_side
                                )[0].tolist()
                                reference.hold_postures[stabilizer_side] = (
                                    env.robot.data.joint_pos.torch[0, joint_ids].tolist()
                                )
                                stabilizer_grip_floor = None
                            return_held_piece(reference, task, report["fault"])
                            if level == cooperative_level:
                                reference.grip(
                                    "placement_correction_stabilizer_open", stabilizer_side, None
                                )
                                reference.hold_targets.pop(stabilizer_side, None)
                                reference.hold_postures.pop(stabilizer_side, None)
                                tool = reference.ctrl.pose(stabilizer_side)[0].tolist()
                                reference.move(
                                    "placement_correction_stabilizer_clear",
                                    stabilizer_side,
                                    [*tool[:2], max(tool[2] + 0.06, stabilizer_clear_z)],
                                    quaternion_xyzw=tool[3:],
                                    linear=True,
                                )
                            verify_support(level - 1, "held placement returned to source")
                        if recovery_mode in (
                            "rebuild_released_prefix",
                            "return_held_then_rebuild",
                        ):
                            completed = report["fault"]["completed_levels"]
                            # Re-execute the same physical assembly primitives from the loss.
                            remaining = list(enumerate(plan))[completed : level + 1] + remaining
                            displaced_sources.update(indices[completed : level + 1])
                            bottom = TABLE_TOP + sum(
                                t.dimensions_m[2] for t in task.targets[:completed]
                            )
                            continue
                        if level == cooperative_level:
                            _, joint_ids, _ = reference.ctrl.frames[stabilizer_side]
                            reference.hold_targets[stabilizer_side] = reference.ctrl.pose(
                                stabilizer_side
                            )[0].tolist()
                            reference.hold_postures[stabilizer_side] = (
                                env.robot.data.joint_pos.torch[0, joint_ids].tolist()
                            )
                        release_z = recover_held_placement(
                            reference,
                            replace(task, assembly_xy_m=build_xy),
                            index,
                            side,
                            yaw,
                            bottom,
                        )
                    else:
                        from vla_lego.isaac.pilot_faults import placement_fault

                        report["fault"] = placement_fault(
                            reference,
                            pilot_case,
                            replace(task, assembly_xy_m=build_xy),
                            indices,
                            side,
                            yaw,
                            release_z,
                            **(
                                {
                                    "horizontal_offset_m": 0.8
                                    * min(
                                        target.dimensions_m[0],
                                        task.targets[level - 1].dimensions_m[0],
                                    )
                                }
                                if args.campaign_cases
                                else {}
                            ),
                        )
                reference.move(
                    f"{prefix}_place",
                    side,
                    [*build_xy, release_z],
                    yaw,
                    linear=pilot_case is not None,
                )
                if level == cooperative_level:
                    # A single contact sample can precede a bounce. Establish
                    # persistent load transfer with the same bounded robot descent.
                    for descent in range(7):
                        for _ in range(math.ceil(COOPERATION_S / PERIOD_S) + 2):
                            reference.wait(1, f"{prefix}_cooperative_load_transfer")
                            coordinated = any(
                                event["stabilizer"] == stabilizer_side
                                and event["upper_level"] == cooperative_level
                                for event in evaluator.result()["cooperation_events"]
                            )
                            if coordinated:
                                break
                        if coordinated:
                            break
                        if descent < 6:
                            reference.move(
                                f"{prefix}_touch_{descent + 1}",
                                side,
                                [*build_xy, release_z - (descent + 1) * 0.001],
                                yaw,
                                linear=pilot_case is not None,
                            )
                    if not coordinated:
                        raise RuntimeError("No persistent physical cooperative load transfer.")
                reference.wait(5 if args.event_settling else 20, f"{prefix}_supported_hold")
                reference.grip(f"{prefix}_release", side, None)
                retreat_z = release_z + target.dimensions_m[2] / 2 + 0.025
                if pilot_case:
                    retreat_z = max(retreat_z, 1.02)
                retreat = [*build_xy, retreat_z]
                if pilot_case and level == len(task.targets) - 1:
                    from vla_lego.isaac.clearance import plan_released_retreat

                    retreat[1] += 0.15 if side == "left" else -0.15
                    route = plan_released_retreat(
                        env,
                        reference.ctrl,
                        reference.solvers[side],
                        side,
                        [*retreat, *reference.orientation(side, yaw).tolist()],
                    )
                    for step, waypoint in enumerate(route):
                        reference.move(
                            (
                                f"{prefix}_retreat"
                                if step == len(route) - 1
                                else f"{prefix}_retreat_vertical"
                            ),
                            side,
                            waypoint,
                            yaw,
                            linear=True,
                        )
                else:
                    reference.move(
                        f"{prefix}_retreat", side, retreat, yaw, linear=pilot_case is not None
                    )
                if pilot_case:
                    # Clear the released stack before a joint-space return can dip.
                    reference.move(
                        f"{prefix}_clear_tower",
                        side,
                        [
                            build_xy[0],
                            build_xy[1] + (0.15 if side == "left" else -0.15),
                            retreat_z,
                        ],
                        yaw,
                        linear=True,
                        linear_step_m=0.02,
                    )
                pending_progress_handoff = (
                    policy_intervention is not None
                    and pilot_case is not None
                    and pilot_case.row["mechanism"] == "interrupted_progress"
                    and level == pilot_case.error_level
                    and "fault" not in report
                )
                # Hand over after release/clearance, without an artificial outward park.
                if not pending_progress_handoff:
                    reference.move(
                        f"{prefix}_exit", side, [0.40, 0.40 if side == "left" else -0.40, 1.05], yaw
                    )
                if not pilot_case:
                    reference.park(side)
                reference.wait(10, f"{prefix}_support_check")
                # The stabilizing hand can support an aligned layer until withdrawal.
                verify_support(
                    level - 1 if level == cooperative_level else level,
                    prefix,
                    held_side=stabilizer_side if level == cooperative_level else None,
                )
                if level == cooperative_level:
                    if (
                        env.task_truth(index)["block_contact_forces_n"][indices[level - 1]][2]
                        <= 1e-6
                    ):
                        raise RuntimeError("Lost physical support before stabilizer withdrawal")
                    if stabilizer_withdraw is None or stabilizer_exit_path is None:
                        raise RuntimeError("Missing physical stabilizer setup.")
                    reference.hold_targets.pop(stabilizer_side, None)
                    reference.hold_postures.pop(stabilizer_side, None)
                    reference.grip("stabilizer_release", stabilizer_side, None)
                    reference.move(
                        "stabilizer_retract",
                        stabilizer_side,
                        stabilizer_withdraw,
                        quaternion_xyzw=stabilizer_quaternion,
                        linear=pilot_case is not None,
                    )
                    reference.move(
                        "stabilizer_clear_tower",
                        stabilizer_side,
                        [*stabilizer_withdraw[:2], stabilizer_clear_z],
                        # Pitch up 20 degrees to keep the proximal wrist reachable.
                        quaternion_xyzw=stabilizer_clear_quaternion,
                    )
                    for name, xyz, rotation in stabilizer_exit_path:
                        reference.move(
                            f"stabilizer_{name}",
                            stabilizer_side,
                            xyz,
                            quaternion_xyzw=rotation,
                        )
                    if not pilot_case:
                        reference.park(stabilizer_side)
                    verify_support(level, "stabilizer withdrawal")
                bottom += target.dimensions_m[2]
            reference.wait(40, "final_released_dwell")
            stop_evaluation = None
            if campaign_stop:
                from vla_lego.isaac.campaign_faults import verify_stop

                stop_evaluation = verify_stop(task, pilot_case, env)
                report["verified_stop"] = stop_evaluation["passed"]
                if recorder is not None and stop_evaluation["passed"]:
                    recorder.task_decision = "abstain"
                    reference.wait(20, "verified_justified_abstention")
            elif recorder is not None and evaluator.result()["passed"]:
                recorder.task_decision = "complete"
                reference.wait(20 if args.campaign_cases else 1, "verified_complete")
            env.capture()
            Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "final.png")
            report.update(
                completed_reference_sequence=not campaign_stop,
                tower_success=(
                    not single_block and not campaign_stop and evaluator.result()["passed"]
                ),
                single_block_success=single_block and evaluator.result()["passed"],
                final_tool_poses={
                    side: reference.ctrl.pose(side)[0].tolist() for side in ("left", "right")
                },
                final_truth=[env.task_truth(i) for i in range(len(env.blocks))],
                cooperation_samples=cooperation,
                passed=(
                    stop_evaluation["passed"]
                    if stop_evaluation is not None
                    else evaluator.result()["passed"]
                ),
                evaluation=stop_evaluation if campaign_stop else evaluator.result(),
                qualification=False,
            )
    except Exception:
        report.update(error=traceback.format_exc())
        raise
    finally:
        if args.policy_socket:
            window = args.output_dir / "policy-window.jsonl"
            report["policy_window_predictions"] = (
                sum(1 for _ in window.open()) if window.exists() else 0
            )
            report["policy_window_started"] = report["policy_window_predictions"] > 0
            report["policy_executed_commands"] = 0 if recorder is None else recorder.policy_frames
            report["autonomous_full_task_success"] = False
        if policy_intervention is not None:
            try:
                policy_intervention.close()
            except OSError:
                pass
        if recorder is not None:
            recorder.finish(
                verified_success=bool(report.get("passed"))
                and not report.get("verified_stop", False),
                verified_stop=bool(report.get("verified_stop", False)),
                reason=report.get(
                    "error",
                    (
                        "complete task admitted"
                        if report.get("passed")
                        else "full task evaluator rejected"
                    ),
                ),
            )
            if recorder.video_error is not None:
                report["passed"] = False
                report["recording_error"] = recorder.video_error
        if reference is not None:
            report["stages"] = reference.stages
            report["max_native_limit_projection_rad"] = reference.max_limit_projection
        report["cooperation_samples"] = cooperation
        report["wall_seconds_before_app_close"] = time.monotonic() - started
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if app is not None:
            app.close(exit_code=0 if report.get("passed") else 1)


if __name__ == "__main__":
    main()
