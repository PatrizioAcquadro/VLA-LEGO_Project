"""Teacher-induced faults produced and corrected only through native robot commands."""

from __future__ import annotations

from vla_lego.isaac.pilot import PilotCase, piece
from vla_lego.isaac.task import TowerTask


def pick_fault(reference, case: PilotCase, task: TowerTask, target_index: int) -> dict:
    env = reference.env
    recorder = env.recorder
    wrong = case.row["error"] == "wrong distractor picked"
    index = (
        next(i for i, p in enumerate(task.placements) if p.spec == piece(case.row["wrong_piece"]))
        if wrong
        else target_index
    )
    placement = task.placements[index]
    side = "left" if placement.xy_m[1] > 0 else "right"
    yaw = (-90 if side == "left" else 90) + placement.yaw_deg
    source = env.task_truth(index)["block_pose_w"][:3]
    y = (
        source[1]
        if wrong
        else source[1] + (-0.09 if abs(source[1]) > 0.3 else 0.09) * (1 if source[1] > 0 else -1)
    )
    yaw = reference.grasp_yaw(
        side,
        yaw,
        [
            [0.40, 0.40 if side == "left" else -0.40, 1.05],
            [min(source[0], 0.46), y, 1.02],
            [source[0], y, source[2] + 0.10],
            [source[0], y, source[2] + 0.06],
            [source[0], y, source[2] + 0.16],
            [source[0], y, 1.02],
            [0.40, 0.40 if side == "left" else -0.40, 1.05],
        ],
        object_index=index,
    )
    prefix = f"level{case.error_level}_fault_{'wrong_piece' if wrong else 'missed_grasp'}"
    recorder.positive_imitation = False
    for phase, xyz in (
        ("edge", [0.40, 0.40 if side == "left" else -0.40, 1.05]),
        ("source_high", [min(source[0], 0.46), y, 1.02]),
        ("approach", [source[0], y, source[2] + 0.10]),
        ("descend", [source[0], y, source[2] + 0.06]),
    ):
        reference.move(f"{prefix}_{phase}", side, xyz, yaw, linear=phase == "descend")
    reference.grip(prefix + "_close", side, placement.spec.dimensions_m[1])
    reference.move(
        prefix + "_lift",
        side,
        [source[0], y, source[2] + 0.16],
        yaw,
        linear=True,
        linear_step_m=0.02,
    )
    after = env.task_truth(index)
    delta = after["block_pose_w"][2] - source[2]
    verified = (
        delta > 0.07 if wrong else abs(delta) < 0.01 and after["robot_contact_force_n"] < 1e-6
    )
    event = {
        "type": case.row["error"],
        "target_level": case.error_level + 1,
        "teacher_induced": True,
        "object_index": index,
        "lift_delta_m": delta,
        "physically_verified": verified,
        "error_end_s": env.time,
    }
    if not verified:
        raise RuntimeError(f"Intended physical fault did not occur: {event}")
    recorder.positive_imitation = True
    if wrong:
        reference.move(
            prefix + "_correct_return",
            side,
            [source[0], source[1], source[2] + 0.06],
            yaw,
            linear=True,
            linear_step_m=0.02,
        )
    reference.grip(prefix + "_correct_open", side, None)
    reference.move(prefix + "_correct_retreat", side, [source[0], y, 1.02], yaw)
    reference.move(
        prefix + "_correct_exit",
        side,
        [0.40, 0.40 if side == "left" else -0.40, 1.05],
        yaw,
        linear=True,
    )
    if wrong:
        from vla_lego.isaac.environment import TABLE_TOP
        from vla_lego.isaac.evaluation import _corners

        returned = env.task_truth(index)
        bottom = min(p[2] for p in _corners(placement.spec.dimensions_m, returned["block_pose_w"]))
        event["returned_table_support"] = (
            returned["nonrobot_contact_force_n"] > 1e-6
            and returned["robot_contact_force_n"] < 1e-6
            and abs(bottom - TABLE_TOP) < 0.003
        )
        if not event["returned_table_support"]:
            raise RuntimeError("Wrong piece was not returned to table support.")
    recorder.fault_verified = True
    return event


def placement_fault(
    reference,
    case: PilotCase,
    task: TowerTask,
    indices: list[int],
    side: str,
    yaw: float,
    release_z: float,
) -> dict:
    env = reference.env
    index = indices[case.error_level]
    prefix = f"level{case.error_level}_fault_incomplete_placement"
    env.recorder.positive_imitation = False
    # Stop the physical placement 18 mm above support; retain the block for correction.
    # A harmless horizontal offset is never treated as an error.
    reference.move(
        prefix,
        side,
        [task.assembly_xy_m[0] + 0.015, task.assembly_xy_m[1], release_z + 0.018],
        yaw,
        linear=True,
    )
    reference.wait(4, prefix + "_failed_load_transfer")
    truth = env.task_truth(index)
    support = truth["block_contact_forces_n"][indices[case.error_level - 1]][2]
    from vla_lego.isaac.environment import TABLE_TOP

    expected_z = (
        TABLE_TOP
        + sum(target.dimensions_m[2] for target in task.targets[: case.error_level])
        + task.targets[case.error_level].dimensions_m[2] / 2
    )
    gap = truth["block_pose_w"][2] - expected_z
    verified = support <= 1e-6 and gap > 0.01 and truth["robot_contact_force_n"] > 0.5
    event = {
        "type": case.row["error"],
        "target_level": case.error_level + 1,
        "teacher_induced": True,
        "support_n": support,
        "height_gap_m": gap,
        "physically_verified": verified,
        "error_end_s": env.time,
    }
    if not verified:
        raise RuntimeError(f"Incomplete placement was not physically consequential: {event}")
    env.recorder.positive_imitation = True
    env.recorder.fault_verified = True
    return event
