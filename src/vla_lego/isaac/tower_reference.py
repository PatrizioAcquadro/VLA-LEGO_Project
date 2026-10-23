"""Contact-only five-block reference with overlapping stabilize/place actions."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--family")
    parser.add_argument("--split", choices=("train", "development"))
    parser.add_argument("--event-settling", action="store_true")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--case")
    parser.add_argument(
        "--motion-measurement",
        choices=("pose_dwell", "solver_velocity", "pose_step_peaks"),
        default="pose_dwell",
    )
    args = parser.parse_args()
    if bool(args.cases) != bool(args.case):
        parser.error("--cases and --case must be provided together")
    if args.record and (not args.family or not args.split):
        parser.error("--record requires --family and --split before collection")
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
    cooperation: list[dict] = []
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
        from PIL import Image

        from vla_lego.isaac.environment import TABLE_TOP, AssemblyEnvironment
        from vla_lego.isaac.evaluation import COOPERATION_S, PERIOD_S, TowerEvaluator
        from vla_lego.isaac.recording import EpisodeRecorder
        from vla_lego.isaac.reference import PhysicalReference
        from vla_lego.isaac.task import initial_tower_task

        pilot_case = None
        if args.cases:
            from vla_lego.isaac.pilot import load_cases

            pilot_case = next(c for c in load_cases(args.cases) if c.name == args.case)
            if pilot_case.category == "stop":
                raise ValueError("Stop cases use the stop collector.")
        task = pilot_case.task() if pilot_case else initial_tower_task(mirror=args.mirror)
        cooperative_level = pilot_case.cooperative_level if pilot_case else 3
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
                (
                    "left" if p.xy_m[1] > 0 else "right",
                    -90 if p.xy_m[1] > 0 else 90,
                )
                for level, target in enumerate(task.targets)
                for p in task.placements
                if p.spec == target
            )
        stabilizer_side, placer_side = roles[cooperative_level - 1][0], roles[cooperative_level][0]
        stabilizer_y = 0.20 if stabilizer_side == "left" else -0.20
        env = AssemblyEnvironment(
            args.cache_dir.resolve(),
            placements=task.placements,
            assembly_xy_m=None if task.build_region_xy_m is not None else task.assembly_xy_m,
            measure_pose_motion=args.motion_measurement == "pose_step_peaks",
            mass_scale=pilot_case.mass_scale if pilot_case else 1.0,
            friction=pilot_case.friction if pilot_case else (0.6, 0.5),
            appearance=pilot_case.appearance if pilot_case else "legacy",
        )
        env.step(400)
        if args.record:
            recorder = EpisodeRecorder(
                args.output_dir / "recording",
                task.instruction,
                family=args.family,
                split=args.split,
                task_decisions=True,
                collection_case=pilot_case.row if pilot_case else None,
                collection_condition="marker_free_continuous_site_v1" if pilot_case else None,
            )
            env.recorder = recorder
        report["task"] = asdict(task)
        if pilot_case:
            report["pilot_case"] = pilot_case.row
            report["collection_condition"] = "marker_free_continuous_site_v1"
            report["simulation_assumptions"] = {
                "mass_scale": pilot_case.mass_scale,
                "effective_friction": pilot_case.friction,
                "table_lighting": pilot_case.appearance,
                "measured_print_properties": False,
            }
        report["robot_joint_names"] = env.robot.data.joint_names
        report["physics_solver_iterations"] = list(env.solver_iterations)
        indices = [
            next(i for i, p in enumerate(task.placements) if p.spec == t) for t in task.targets
        ]
        contact_ids = {
            side: [i for i, path in enumerate(env.contact_robot_paths) if f"{side}_WSG32" in path]
            for side in ("left", "right")
        }
        if not all(contact_ids.values()):
            raise RuntimeError("Cannot identify actual WSG bodies for cooperation evidence.")
        evaluator = TowerEvaluator(
            task, env.contact_robot_paths, motion_measurement=args.motion_measurement
        )

        def sample_cooperation(stage: str, truth: list[dict]) -> None:
            assert reference is not None
            evaluator.update(
                env.time,
                truth,
                tool_positions={
                    side: reference.ctrl.pose(side)[0, :3].tolist() for side in ("left", "right")
                },
            )
            if not stage.startswith(f"level{cooperative_level}_"):
                return
            lower, upper = truth[indices[cooperative_level - 1]], truth[indices[cooperative_level]]
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

            def verify_support(last_level: int, stage: str) -> None:
                expected_bottom = TABLE_TOP
                for prior in range(last_level + 1):
                    state = env.task_truth(indices[prior])
                    expected_z = expected_bottom + task.targets[prior].dimensions_m[2] / 2
                    support_n = (
                        state["nonrobot_contact_force_n"]
                        if prior == 0
                        else state["block_contact_forces_n"][indices[prior - 1]][2]
                    )
                    if abs(state["block_pose_w"][2] - expected_z) >= 0.003 or support_n <= 1e-6:
                        raise RuntimeError(
                            f"Incorrect physical support at level {prior} after {stage}"
                        )
                    expected_bottom += task.targets[prior].dimensions_m[2]

            def engage_stabilizer():
                nonlocal stabilizer_quaternion, stabilizer_clear_quaternion, stabilizer_clear_z
                from isaaclab.utils.math import quat_apply, quat_mul

                from vla_lego.isaac.evaluation import _corners

                state = env.task_truth(indices[cooperative_level - 1])
                lower = state["block_pose_w"][:3]
                nominal_width = task.targets[cooperative_level - 1].dimensions_m[1]
                width = nominal_width
                stabilizer_pose = [lower[0] - 0.01, lower[1], lower[2]]
                approach = (
                    ("high", [lower[0] - 0.01, stabilizer_y, 1.02]),
                    ("center", [lower[0] - 0.01, lower[1], 1.02]),
                    ("above", [lower[0] - 0.01, lower[1], lower[2] + 0.06]),
                    ("grasp", stabilizer_pose),
                )
                if pilot_case:
                    reference.admit_stabilizer_clearance(stabilizer_side)
                    _, joint_ids, _ = reference.ctrl.frames[stabilizer_side]
                    selected = None
                    failures = []
                    preferred_sign = -1 if stabilizer_side == "left" else 1
                    angles = (
                        0,
                        preferred_sign * 20,
                        -preferred_sign * 20,
                        preferred_sign * 35,
                        -preferred_sign * 35,
                        preferred_sign * 60,
                        -preferred_sign * 60,
                        preferred_sign * 70,
                        -preferred_sign * 70,
                        preferred_sign * 90,
                        -preferred_sign * 90,
                    )
                    for yaw_deg in angles:
                        for pitch_deg, reach in ((0, 0.045), (10, 0.060)):
                            yaw, pitch = math.radians(yaw_deg) / 2, math.radians(pitch_deg) / 2
                            candidate = quat_mul(
                                env.target.new_tensor([0, 0, math.sin(yaw), math.cos(yaw)]),
                                env.target.new_tensor([0, math.sin(pitch), 0, math.cos(pitch)]),
                            )
                            finger = quat_apply(
                                candidate, env.target.new_tensor([1.0, 0, 0])
                            ).tolist()
                            axis = quat_apply(
                                candidate, env.target.new_tensor([0.0, 1, 0])
                            ).tolist()
                            projections = [
                                sum(v * a for v, a in zip(corner, axis, strict=True))
                                for corner in _corners(
                                    task.targets[cooperative_level - 1].dimensions_m,
                                    state["block_pose_w"],
                                )
                            ]
                            gap = max(projections) - min(projections)
                            if gap > 0.070:
                                continue
                            pose = [v - reach * d for v, d in zip(lower, finger, strict=True)]
                            orientation = candidate.tolist()
                            path = (
                                ("high", [0.40, stabilizer_y, 1.02]),
                                ("above", [pose[0], pose[1], pose[2] + 0.06]),
                                ("grasp", pose),
                            )
                            seed = env.robot.data.joint_pos.torch[0, joint_ids]
                            phase = "approach"
                            try:
                                for name, xyz in path:
                                    phase = name
                                    rotation = [0.0, 0, 0, 1.0] if name == "high" else orientation
                                    seed = reference.solvers[stabilizer_side].solve(
                                        env.target.new_tensor([*xyz, *rotation]), seed
                                    )
                                phase = "retract"
                                seed = reference.solvers[stabilizer_side].solve(
                                    env.target.new_tensor(
                                        [pose[0], pose[1], pose[2] + 0.06, *orientation]
                                    ),
                                    seed,
                                )
                                clear_z = (
                                    TABLE_TOP
                                    + sum(
                                        t.dimensions_m[2]
                                        for t in task.targets[: cooperative_level + 1]
                                    )
                                    + 0.06
                                )
                                retreat = None
                                for clear_pitch in (0, -20, 20):
                                    half = math.radians(clear_pitch) / 2
                                    rotation = quat_mul(
                                        candidate,
                                        env.target.new_tensor(
                                            [0, math.sin(half), 0, math.cos(half)]
                                        ),
                                    ).tolist()
                                    branch = seed
                                    try:
                                        for xyz in (
                                            [pose[0], pose[1], clear_z],
                                            [pose[0], stabilizer_y, clear_z],
                                        ):
                                            branch = reference.solvers[stabilizer_side].solve(
                                                env.target.new_tensor([*xyz, *rotation]), branch
                                            )
                                        retreat = rotation
                                        break
                                    except RuntimeError:
                                        continue
                                if retreat is None:
                                    raise RuntimeError("No collision-free withdrawal orientation")
                                selected = path, orientation, pose, gap, retreat, clear_z
                                break
                            except RuntimeError as error:
                                failures.append(
                                    {
                                        "yaw_deg": yaw_deg,
                                        "pitch_deg": pitch_deg,
                                        "phase": phase,
                                        "reason": str(error),
                                    }
                                )
                        if selected is not None:
                            break
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
                    ) = selected
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
                        linear=pilot_case is not None and name == "grasp",
                    )
                if pilot_case:
                    reference.hold_targets[stabilizer_side] = [
                        *stabilizer_pose,
                        *stabilizer_quaternion,
                    ]
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
                return stabilizer_pose

            for side in ("left", "right"):
                reference.park(side)
            env.capture()
            Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "initial.png")
            env.stage.Export(str(args.output_dir / "scene.usda"))
            bottom = TABLE_TOP
            stabilizer_pose = None
            build_xy = task.assembly_xy_m
            # Per-piece yaw is a teacher planning choice, never a policy observation.
            for level, (target, index, (side, base_yaw)) in enumerate(
                zip(task.targets, indices, roles, strict=True)
            ):
                if level and pilot_case:
                    build_xy = tuple(env.task_truth(indices[0])["block_pose_w"][:2])
                prefix = f"level{level}"
                yaw = float(base_yaw)
                if (
                    pilot_case
                    and pilot_case.category == "recovery"
                    and level == pilot_case.error_level
                    and pilot_case.row["error"] != "physically consequential incomplete placement"
                ):
                    from vla_lego.isaac.pilot_faults import pick_fault

                    report["fault"] = pick_fault(reference, pilot_case, task, index)
                    verify_support(level - 1, "pick fault correction")
                if level == cooperative_level:
                    stabilizer_pose = engage_stabilizer()
                block = env.task_truth(index)["block_pose_w"][:3]
                if pilot_case:
                    yaw += task.placements[index].yaw_deg
                    release = bottom + target.dimensions_m[2] / 2 + 0.06
                    yaw = reference.grasp_yaw(
                        side,
                        yaw,
                        [
                            [0.40, 0.40 if side == "left" else -0.40, 1.05],
                            [min(block[0], 0.46), block[1], 1.02],
                            [block[0], block[1], block[2] + 0.10],
                            [block[0], block[1], block[2] + 0.06],
                            [block[0], block[1], block[2] + 0.16],
                            *(
                                [[*build_xy, block[2] + 0.16]]
                                if release + 0.025 < block[2] + 0.16
                                else []
                            ),
                            [*build_xy, release + 0.025],
                            [*build_xy, release],
                            [*build_xy, release + target.dimensions_m[2] / 2 + 0.025],
                        ],
                        object_index=index,
                    )
                for stage, xyz in (
                    ("edge", [0.40, 0.40 if side == "left" else -0.40, 1.05]),
                    (
                        "source_high",
                        [min(block[0], 0.46) if pilot_case else block[0], block[1], 1.02],
                    ),
                    ("approach", [block[0], block[1], block[2] + 0.10]),
                    ("descend", [block[0], block[1], block[2] + 0.06]),
                ):
                    reference.move(
                        f"{prefix}_{stage}",
                        side,
                        xyz,
                        yaw,
                        linear=pilot_case is not None and stage == "descend",
                    )
                reference.grip(f"{prefix}_grip", side, target.dimensions_m[1])
                release_z = bottom + target.dimensions_m[2] / 2 + 0.06
                lift_z = block[2] + 0.16
                transit_z = release_z + 0.025
                reference.move(f"{prefix}_lift", side, [block[0], block[1], lift_z], yaw)
                if env.task_truth(index)["block_pose_w"][2] < block[2] + 0.07:
                    raise RuntimeError(f"No verified contact-only lift at {prefix}")
                if pilot_case and transit_z < lift_z:
                    # Cross the loose inventory at lift height before descending
                    # over the assembly area, rather than sweeping through props.
                    reference.move(
                        f"{prefix}_transport_high",
                        side,
                        [*build_xy, lift_z],
                        yaw,
                        linear=True,
                        linear_step_m=0.02,
                    )
                if level >= cooperative_level:
                    # Gain clearance away from the existing tower before the
                    # carried object's footprint crosses into the assembly area.
                    for waypoint, y in enumerate((0.20, 0.10)):
                        reference.move(
                            f"{prefix}_carry_clearance_{waypoint}",
                            side,
                            [build_xy[0], build_xy[1] + (y if side == "left" else -y), transit_z],
                            yaw,
                        )
                reference.move(f"{prefix}_transport", side, [*build_xy, transit_z], yaw)
                if (
                    pilot_case
                    and pilot_case.category == "recovery"
                    and level == pilot_case.error_level
                    and pilot_case.row["error"] == "physically consequential incomplete placement"
                ):
                    from vla_lego.isaac.pilot_faults import placement_fault

                    report["fault"] = placement_fault(
                        reference,
                        pilot_case,
                        replace(task, assembly_xy_m=build_xy),
                        indices,
                        side,
                        yaw,
                        release_z,
                    )
                reference.move(f"{prefix}_place", side, [*build_xy, release_z], yaw)
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
                            )
                    if not coordinated:
                        raise RuntimeError("No persistent physical cooperative load transfer.")
                reference.wait(5 if args.event_settling else 20, f"{prefix}_supported_hold")
                reference.grip(f"{prefix}_release", side, None)
                retreat_z = release_z + target.dimensions_m[2] / 2 + 0.025
                reference.move(f"{prefix}_retreat", side, [*build_xy, retreat_z], yaw)
                reference.move(
                    f"{prefix}_exit", side, [0.40, 0.40 if side == "left" else -0.40, 1.05], yaw
                )
                reference.park(side)
                reference.wait(10, f"{prefix}_support_check")
                # A held piece may settle when the restraining jaws withdraw.
                # Require actual load support now and full geometry after withdrawal.
                verify_support(level - 1 if level == cooperative_level else level, prefix)
                if level == cooperative_level:
                    if (
                        env.task_truth(index)["block_contact_forces_n"][indices[level - 1]][2]
                        <= 1e-6
                    ):
                        raise RuntimeError("Lost physical support before stabilizer withdrawal")
                    if stabilizer_pose is None:
                        raise RuntimeError("Missing physical stabilizer setup.")
                    reference.hold_targets.pop(stabilizer_side, None)
                    reference.grip("stabilizer_release", stabilizer_side, None)
                    reference.move(
                        "stabilizer_retract",
                        stabilizer_side,
                        [*stabilizer_pose[:2], stabilizer_pose[2] + 0.06],
                        quaternion_xyzw=stabilizer_quaternion,
                    )
                    reference.move(
                        "stabilizer_clear_tower",
                        stabilizer_side,
                        [*stabilizer_pose[:2], stabilizer_clear_z],
                        # Pitch up 20 degrees to keep the proximal wrist reachable.
                        quaternion_xyzw=stabilizer_clear_quaternion,
                    )
                    reference.move(
                        "stabilizer_exit",
                        stabilizer_side,
                        [stabilizer_pose[0], stabilizer_y, stabilizer_clear_z],
                        quaternion_xyzw=stabilizer_clear_quaternion,
                    )
                    reference.park(stabilizer_side)
                    verify_support(level, "stabilizer withdrawal")
                bottom += target.dimensions_m[2]
            reference.wait(40, "final_released_dwell")
            if recorder is not None and evaluator.result()["passed"]:
                recorder.task_decision = "complete"
                reference.wait(1, "verified_complete")
            env.capture()
            Image.fromarray(env.observe()["head_rgb"]).save(args.output_dir / "final.png")
            report.update(
                completed_reference_sequence=True,
                final_tool_poses={
                    side: reference.ctrl.pose(side)[0].tolist() for side in ("left", "right")
                },
                final_truth=[env.task_truth(i) for i in range(len(env.blocks))],
                cooperation_samples=cooperation,
                passed=evaluator.result()["passed"],
                evaluation=evaluator.result(),
                qualification=False,
            )
    except Exception:
        report.update(error=traceback.format_exc())
        raise
    finally:
        if recorder is not None:
            recorder.finish(
                verified_success=bool(report.get("passed")),
                reason=report.get(
                    "error",
                    (
                        "complete task admitted"
                        if report.get("passed")
                        else "full task evaluator rejected"
                    ),
                ),
            )
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
