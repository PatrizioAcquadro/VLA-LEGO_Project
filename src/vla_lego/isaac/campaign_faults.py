"""Observable collection faults and corrections produced only by robot commands."""

from __future__ import annotations

import math

from vla_lego.isaac.environment import TABLE_TOP


def drop_and_recover(reference, case, task, index: int, side: str, yaw: float) -> dict:
    env, recorder = reference.env, reference.env.recorder
    before = env.task_truth(index)["block_pose_w"][:3]
    recorder.positive_imitation = False
    reference.grip(f"level{case.error_level}_fault_drop_open", side, None)
    reference.wait(20, "fault_drop_settling")
    after = env.task_truth(index)
    source = after["block_pose_w"][:3]
    height = task.placements[index].spec.dimensions_m[2]
    verified = (
        before[2] - source[2] >= 0.07
        and abs(source[2] - TABLE_TOP - height / 2) < 0.01
        and after["robot_contact_force_n"] < 1e-6
        and after["nonrobot_contact_force_n"] > 1e-6
        and 0.25 <= source[0] <= 0.55
        and abs(source[1]) < 0.5
    )
    if not verified:
        raise RuntimeError("Dropped block did not establish a released, reachable recovery state")
    event = {
        "type": "slip_drop",
        "teacher_induced": True,
        "target_level": case.error_level + 1,
        "physically_verified": True,
        "fall_m": before[2] - source[2],
        "error_end_s": env.time,
    }
    recorder.positive_imitation = True
    recover_dropped(reference, task, index, side, yaw)
    recorder.fault_verified = True
    return event


def resting_yaw(pose, spec) -> float:
    """Orientation of the actual resting piece, including displaced source blocks."""
    x, y, z, w = pose[3:]
    if spec.shape == "prism" and abs(1 - 2 * (x * x + y * y)) < math.cos(math.radians(15)):
        raise RuntimeError(
            "Tipped prism requires a verified uprighting correction, not a nominal pick"
        )
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _held_return_waypoints(tool, block, dimensions, xy, obstacle_top):
    """Keep the measured tool-to-piece offset when returning a rotated grasp."""
    from vla_lego.isaac.evaluation import _corners

    offset = [a - b for a, b in zip(tool[:3], block[:3], strict=True)]
    bottom = min(c[2] for c in _corners(dimensions, block))
    release_z = TABLE_TOP + block[2] - bottom + offset[2]
    clear_z = max(tool[2], release_z + 0.12, 1.02, tool[2] + obstacle_top + 0.025 - bottom)
    target_xy = [a + b for a, b in zip(xy, offset[:2], strict=True)]
    return [
        ("raise", [*tool[:2], clear_z]),
        ("return", [*target_xy, clear_z]),
        ("lower", [*target_xy, release_z]),
    ]



def recover_dropped(reference, task, index: int, side: str, yaw: float, *, replan=False) -> float:
    """Reacquire a block that has already fallen; this function never causes the fall."""
    env = reference.env
    state = env.task_truth(index)
    source = state["block_pose_w"][:3]
    if replan:
        from vla_lego.isaac.assembly_planning import pick_route

        actual_yaw = resting_yaw(state["block_pose_w"], task.placements[index].spec)
        level = next(
            i for i, target in enumerate(task.targets) if target == task.placements[index].spec
        )
        yaw = reference.grasp_yaw(
            side,
            (-90 if side == "left" else 90) + actual_yaw,
            pick_route(task, level, source, task.assembly_xy_m, side, 3),
            object_index=index,
        )
        reference.grip("drop_recovery_open", side, None)
        for name, xyz in (
            ("edge", [0.4, 0.4 if side == "left" else -0.4, 1.05]),
            ("source_high", [min(source[0], 0.46), source[1], 1.02]),
        ):
            reference.move("drop_recovery_" + name, side, xyz, yaw)
    for name, xyz in (
        ("above", [source[0], source[1], source[2] + 0.16]),
        ("approach", [source[0], source[1], source[2] + 0.10]),
        ("descend", [source[0], source[1], source[2] + 0.06]),
    ):
        reference.move("drop_recovery_" + name, side, xyz, yaw, linear=True)
    reference.grip("drop_recovery_grip", side, task.placements[index].spec.dimensions_m[1])
    reference.move(
        "drop_recovery_lift",
        side,
        [source[0], source[1], source[2] + 0.16],
        yaw,
        linear=True,
        linear_step_m=0.02,
    )
    if env.task_truth(index)["block_pose_w"][2] - source[2] < 0.07:
        raise RuntimeError("Drop correction failed physical re-acquisition")
    return yaw


def return_wrong_piece(reference, task, event) -> None:
    """Return the actually held wrong piece before resuming the instructed tower."""
    return_held_piece(reference, task, event, stage_prefix="wrong_piece_correction")


def return_held_piece(
    reference, task, event, *, stage_prefix="placement_correction_return"
) -> None:
    """Physically release a held piece at its vacant source before another pick."""
    from vla_lego.isaac.evaluation import CONTACT_ZERO_N, POSITION_TOLERANCE_M, _corners

    env = reference.env
    index, side = event["object_index"], event["side"]
    xy = event["return_source_xy_m"]
    spec = task.placements[index].spec
    for i, placement in enumerate(task.placements):
        if i != index and math.dist(xy, env.task_truth(i)["block_pose_w"][:2]) < (
            math.hypot(*spec.dimensions_m[:2]) / 2
            + math.hypot(*placement.spec.dimensions_m[:2]) / 2
            + 0.01
        ):
            raise RuntimeError(
                "Held piece's return space is occupied; another correction is needed"
            )
    state = env.task_truth(index)
    jaw_forces = [
        math.hypot(*state["robot_contact_forces_n"][i])
        for i, path in enumerate(env.contact_robot_paths)
        if path.rsplit("/", 1)[-1]
        in (f"{side}_WSG32_NEGATIVE_JAW_LINK", f"{side}_WSG32_POSITIVE_JAW_LINK")
    ]
    if len(jaw_forces) != 2 or min(jaw_forces) <= CONTACT_ZERO_N or sum(jaw_forces) <= 0.5:
        raise RuntimeError("Returning a held piece requires actual bilateral jaw contact")
    tool = reference.ctrl.pose(side)[0].tolist()
    orientation = tool[3:]
    obstacle_top = max(
        c[2]
        for i, placement in enumerate(task.placements)
        if i != index
        for c in _corners(placement.spec.dimensions_m, env.task_truth(i)["block_pose_w"])
    )
    waypoints = _held_return_waypoints(
        tool, state["block_pose_w"], spec.dimensions_m, xy, obstacle_top
    )
    for stage, xyz in waypoints:
        reference.move(
            stage_prefix + "_" + stage,
            side,
            xyz,
            0,
            quaternion_xyzw=orientation,
            linear=True,
            linear_step_m=0.02,
        )
    reference.grip(stage_prefix + "_release", side, None)
    reference.move(
        stage_prefix + "_withdraw",
        side,
        waypoints[1][1],
        0,
        quaternion_xyzw=orientation,
        linear=True,
    )
    reference.wait(10, stage_prefix + "_support")
    returned = env.task_truth(index)
    actual_bottom = min(c[2] for c in _corners(spec.dimensions_m, returned["block_pose_w"]))
    event["returned_table_support"] = (
        returned["nonrobot_contact_force_n"] > CONTACT_ZERO_N
        and returned["robot_contact_force_n"] < CONTACT_ZERO_N
        and abs(actual_bottom - TABLE_TOP) < POSITION_TOLERANCE_M
    )
    if not event["returned_table_support"]:
        raise RuntimeError("Actual held piece was not physically returned to table support")


def recover_held_placement(reference, task, index, side, yaw, bottom) -> float:
    """Restore clearance and the intended grasp orientation before physical load transfer."""
    from vla_lego.isaac.evaluation import _corners

    env = reference.env
    pose = env.task_truth(index)["block_pose_w"]
    tool = reference.ctrl.pose(side)[0].tolist()
    lowest = min(c[2] for c in _corners(task.placements[index].spec.dimensions_m, pose))
    clear_z = tool[2] + max(0.025, bottom + 0.025 - lowest)
    reference.move(
        "placement_correction_raise",
        side,
        [*tool[:2], clear_z],
        quaternion_xyzw=tool[3:],
        linear=True,
    )
    reference.move(
        "placement_correction_align",
        side,
        [*task.assembly_xy_m, clear_z],
        yaw,
        linear=True,
    )
    held_z = env.task_truth(index)["block_pose_w"][2]
    tool_z = float(reference.ctrl.pose(side)[0, 2])
    return float(bottom + task.placements[index].spec.dimensions_m[2] / 2 + tool_z - held_z)


def verify_stop(task, case, env) -> dict:
    """Offline justification; this information is never a deployed decision input."""
    return stop_evidence(task, case, [env.task_truth(i) for i in range(len(env.blocks))])


def stop_evidence(task, case, states: list[dict]) -> dict:
    """Current physical justification, independent of the teacher's stopping time."""
    reason = case.row["stop_reason"]
    if reason == "unreachable":
        target = task.targets[case.error_level]
        index = next(i for i, p in enumerate(task.placements) if p.spec == target)
        xyz = states[index]["block_pose_w"][:3]
        justified = xyz[0] >= 0.80 and abs(xyz[1]) >= 0.35
    else:
        justified = not task.requested_pieces_available
    safe = all(
        s["robot_contact_force_n"] < 1e-6
        and (
            s["nonrobot_contact_force_n"] > 1e-6
            or any(force[2] > 1e-6 for force in s["block_contact_forces_n"])
        )
        and all(math.isfinite(v) for v in s["block_pose_w"])
        for s in states
    )
    return {
        "reason": reason,
        "observable_justification": justified,
        "safe_released_hold": safe,
        "passed": bool(justified and safe),
        "target_level": case.error_level + 1,
        "qualification": False,
    }
