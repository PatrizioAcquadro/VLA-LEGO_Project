"""Shared teacher stabilization planning for prospective and actual supported towers."""

from __future__ import annotations

import math

from vla_lego.isaac.environment import TABLE_TOP
from vla_lego.isaac.evaluation import _corners


def stabilized_level(task, cooperative_level):
    """Brace a broader supporting layer instead of tipping a narrower upper piece."""
    level = cooperative_level - 1
    while (
        level > 0 and task.targets[level - 1].dimensions_m[1] > task.targets[level].dimensions_m[1]
    ):
        level -= 1
    return level


def plan_stabilizer(reference, task, indices, cooperative_level, stabilizer_side, *, states=None):
    """No commands or state writes; actual contact execution remains the final authority."""
    from isaaclab.utils.math import quat_apply, quat_mul

    env = reference.env
    truth = env.task_truth if states is None else states.__getitem__
    stabilizer_level = stabilized_level(task, cooperative_level)
    state = truth(indices[stabilizer_level])
    lower = state["block_pose_w"][:3]
    lower_height = task.targets[stabilizer_level].dimensions_m[2]
    stabilizer_y = 0.20 if stabilizer_side == "left" else -0.20
    reference.admit_stabilizer_clearance(stabilizer_side)
    _, joint_ids, _ = reference.ctrl.frames[stabilizer_side]
    selected = None
    selected_cost = math.inf
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
    outline = _corners(task.targets[stabilizer_level].dimensions_m, state["block_pose_w"])

    def projected_gap(angle):
        radians = math.radians(angle)
        coordinates = [
            -math.sin(radians) * corner[0] + math.cos(radians) * corner[1] for corner in outline
        ]
        return max(coordinates) - min(coordinates)

    angles = tuple(sorted(angles, key=lambda angle: round(projected_gap(angle), 3)))
    stack_top = max(
        corner[2]
        for prior in range(cooperative_level)
        for corner in _corners(
            task.targets[prior].dimensions_m,
            truth(indices[prior])["block_pose_w"],
        )
    )
    approach_high_z = max(1.02, stack_top + 0.06)
    profiles = ((0, 0.045),) if lower_height < 0.05 else ((0, 0.045), (10, 0.060))
    for yaw_deg in angles:
        for pitch_deg, reach in profiles:
            yaw, pitch = math.radians(yaw_deg) / 2, math.radians(pitch_deg) / 2
            candidate = quat_mul(
                env.target.new_tensor([0, 0, math.sin(yaw), math.cos(yaw)]),
                env.target.new_tensor([0, math.sin(pitch), 0, math.cos(pitch)]),
            )
            finger = quat_apply(candidate, env.target.new_tensor([1.0, 0, 0])).tolist()
            axis = quat_apply(candidate, env.target.new_tensor([0.0, 1, 0])).tolist()
            gap = projected_gap(yaw_deg)
            # Preserve the existing 3 mm TCP tracking allowance
            # on either side of the open 70 mm jaw aperture.
            if gap + 0.006 > 0.070:
                continue
            pose = [v - reach * d for v, d in zip(lower, finger, strict=True)]
            # The 40 mm distal pads need clearance below the next piece.
            pose[2] -= max(0.0, 0.025 - lower_height / 2)
            orientation = candidate.tolist()
            path = (
                ("high", [0.40, stabilizer_y, approach_high_z]),
                ("above", [pose[0], pose[1], max(pose[2] + 0.06, approach_high_z)]),
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
                # A pose admitted exactly at several stops can drift under
                # load. Rank feasible grasps by available joint travel;
                # this changes teacher choice, never physical acceptance.
                bounds = reference.solvers[stabilizer_side].bounds
                travel = (seed - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
                grasp_cost = float(-(travel * (1 - travel) + 1e-6).log().sum())
                phase = "retract"
                upper = truth(indices[cooperative_level])
                upper_projection = [
                    sum(v * a for v, a in zip(corner, axis, strict=True))
                    for corner in _corners(
                        task.targets[cooperative_level].dimensions_m,
                        upper["block_pose_w"],
                    )
                ]
                # Do not lift open fingers around a wider upper piece.
                # Retain the controller's 3 mm tracking allowance per side.
                wide = max(upper_projection) - min(upper_projection) + 0.006 > 0.070
                retract = (
                    [
                        pose[0] - 0.08 * finger[0],
                        pose[1] - 0.08 * finger[1],
                        pose[2],
                    ]
                    if wide
                    else [pose[0], pose[1], pose[2] + 0.06]
                )
                for fraction in (0.25, 0.5, 0.75, 1.0):
                    xyz = [a + (b - a) * fraction for a, b in zip(pose, retract, strict=True)]
                    seed = reference.solvers[stabilizer_side].solve(
                        env.target.new_tensor([*xyz, *orientation]), seed
                    )
                clear_z = (
                    TABLE_TOP
                    + sum(t.dimensions_m[2] for t in task.targets[: cooperative_level + 1])
                    + 0.06
                )
                retreat = None
                for clear_pitch in (0, -20, 20):
                    half = math.radians(clear_pitch) / 2
                    rotation = quat_mul(
                        candidate,
                        env.target.new_tensor([0, math.sin(half), 0, math.cos(half)]),
                    ).tolist()
                    branch = seed
                    try:
                        for xyz in (
                            [retract[0], retract[1], clear_z],
                            [retract[0], stabilizer_y, clear_z],
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
                if grasp_cost < selected_cost:
                    selected_cost = grasp_cost
                    selected = (
                        path,
                        orientation,
                        pose,
                        gap,
                        retreat,
                        clear_z,
                        retract,
                    )
            except RuntimeError as error:
                failures.append(
                    {
                        "yaw_deg": yaw_deg,
                        "pitch_deg": pitch_deg,
                        "phase": phase,
                        "reason": str(error),
                    }
                )
    return selected, selected_cost, failures
