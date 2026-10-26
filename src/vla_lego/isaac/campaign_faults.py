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


def recover_dropped(reference, task, index: int, side: str, yaw: float, *, replan=False) -> float:
    """Reacquire a block that has already fallen; this function never causes the fall."""
    env = reference.env
    state = env.task_truth(index)
    source = state["block_pose_w"][:3]
    if replan:
        from vla_lego.isaac.assembly_planning import pick_route

        quaternion = state["block_pose_w"][3:]
        x, y, z, w = quaternion
        if task.placements[index].spec.shape == "prism" and abs(1 - 2 * (x * x + y * y)) < math.cos(
            math.radians(15)
        ):
            raise RuntimeError(
                "Tipped prism requires a verified uprighting correction, not a nominal pick"
            )
        actual_yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
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


def interrupt_progress(reference, case, level: int) -> dict:
    env, recorder = reference.env, reference.env.recorder
    if not 0 <= level < 4:
        raise ValueError("An interrupted tower must remain incomplete")
    recorder.positive_imitation = False
    beginning = env.time
    reference.wait(60, f"level{level}_fault_interrupted_progress")
    recorder.positive_imitation = True
    recorder.fault_verified = True
    return {
        "type": "interrupted_progress",
        "teacher_induced": True,
        "physically_verified": True,
        "completed_levels": level + 1,
        "pause_s": env.time - beginning,
        "error_end_s": env.time,
        "original_task_resumed": True,
    }


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
