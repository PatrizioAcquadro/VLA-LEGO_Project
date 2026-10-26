"""Teacher build-site selection against the complete tower's arm routes."""

from __future__ import annotations

import math

from vla_lego.isaac.environment import TABLE_TOP


def pick_route(task, level, block, build_xy, side, cooperative_level):
    bottom = TABLE_TOP + sum(target.dimensions_m[2] for target in task.targets[:level])
    height = task.targets[level].dimensions_m[2]
    release = bottom + height / 2 + 0.06
    carry = max(block[2] + 0.16, release + 0.030)
    return [
        [0.40, 0.40 if side == "left" else -0.40, 1.05],
        [min(block[0], 0.46), block[1], 1.02],
        [block[0], block[1], block[2] + 0.10],
        [block[0], block[1], block[2] + 0.06],
        [block[0], block[1], block[2] + 0.16],
        [block[0], block[1], carry],
        *(
            [
                [build_xy[0], build_xy[1] + (y if side == "left" else -y), carry]
                for y in (0.20, 0.10)
            ]
            if level >= cooperative_level
            else []
        ),
        *([[*build_xy, block[2] + 0.16]] if release + 0.025 < block[2] + 0.16 else []),
        [*build_xy, release + 0.025],
        [*build_xy, release],
        [*build_xy, release + height / 2 + 0.025],
        [
            build_xy[0],
            build_xy[1] + (0.15 if side == "left" else -0.15),
            release + height / 2 + 0.025,
        ],
    ]


def build_site_candidates(task):
    """Prefer the declared site; alternatives remain free and within the same workspace."""
    proposed = [task.assembly_xy_m, (0.40, 0.0), (0.42, 0.0), (0.40, -0.06), (0.40, 0.06)]
    x0, x1, y0, y1 = task.build_region_xy_m
    result = []
    for xy in proposed:
        if xy in result or not (x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1):
            continue
        if all(
            math.dist(xy, p.xy_m) > math.hypot(*p.spec.dimensions_m[:2]) / 2 + 0.065
            for p in task.placements
        ):
            result.append(xy)
    return result


def choose_build_site(reference, task, indices, roles, cooperative_level):
    """Plan before commands; the task evaluator still owns physical success.

    No source scene, instruction or goal criterion changes. A free site is a
    teacher choice, not a hidden destination that the policy must reproduce.
    """
    from vla_lego.isaac.stabilizer_planning import plan_stabilizer

    rejected = []
    for xy in build_site_candidates(task):
        try:
            for level, (index, (side, yaw)) in enumerate(zip(indices, roles, strict=True)):
                if index < 0:
                    raise ValueError("Full-tower site planning requires the requested inventory")
                block = reference.env.task_truth(index)["block_pose_w"][:3]
                reference.grasp_yaw(
                    side,
                    yaw + task.placements[index].yaw_deg,
                    pick_route(task, level, block, xy, side, cooperative_level),
                )
            # Predict only nominal prefix geometry; never write those poses into simulation.
            states = [reference.env.task_truth(i) for i in range(len(task.placements))]
            bottom = TABLE_TOP
            for level in range(cooperative_level):
                index = indices[level]
                height = task.targets[level].dimensions_m[2]
                states[index] = {
                    **states[index],
                    "block_pose_w": [*xy, bottom + height / 2, *states[index]["block_pose_w"][3:]],
                }
                bottom += height
            selected, _, failures = plan_stabilizer(
                reference,
                task,
                indices,
                cooperative_level,
                roles[cooperative_level - 1][0],
                states=states,
            )
            if selected is None:
                rejected.append({"xy_m": xy, "stage": "stabilizer", "reasons": failures})
                continue
            return xy, {
                "declared_preference_xy_m": task.assembly_xy_m,
                "selected_xy_m": xy,
                "all_level_arm_routes_admitted": True,
                "stabilizer_approach_and_withdrawal_admitted": True,
                "rejected": rejected,
                "physical_success": False,
            }
        except RuntimeError as error:
            rejected.append({"xy_m": xy, "level": level + 1, "reason": str(error)})
    raise RuntimeError(f"No clear build site supports all tower arm routes: {rejected}")
