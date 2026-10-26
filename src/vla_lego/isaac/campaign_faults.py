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
    recorder.fault_verified = True
    return event


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
    reason = case.row["stop_reason"]
    if reason == "unreachable":
        target = task.targets[case.error_level]
        index = next(i for i, p in enumerate(task.placements) if p.spec == target)
        xyz = env.task_truth(index)["block_pose_w"][:3]
        justified = xyz[0] >= 0.80 and abs(xyz[1]) >= 0.35
    else:
        justified = not task.requested_pieces_available
    states = [env.task_truth(i) for i in range(len(env.blocks))]
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
