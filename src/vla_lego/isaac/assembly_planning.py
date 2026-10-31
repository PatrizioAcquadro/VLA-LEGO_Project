"""Teacher build-site selection against the complete tower's arm routes."""

from __future__ import annotations

import math

from vla_lego.isaac.environment import TABLE_TOP


def lift_route(block, build_xy, side, *, grasp_height=0.06, displaced=False):
    """Extract a displaced piece away from the prefix before a high cross-body lift."""
    lift = block[2] + grasp_height + 0.10
    if not displaced:
        return [("lift", [block[0], block[1], lift])]
    extract = block[2] + grasp_height + 0.04
    forward = max(block[0], 0.49)
    outward = build_xy[1] + (0.20 if side == "left" else -0.20)
    return [
        ("lift", [block[0], block[1], extract]),
        ("extract_forward", [forward, block[1], extract]),
        ("extract_outward", [forward, outward, extract]),
        ("lift_clear", [forward, outward, lift]),
    ]


def source_route(block, build_xy, side, *, grasp_height=0.06, displaced=False):
    """Approach a displaced source through the same low corridor used for extraction."""
    edge = ("edge", [0.40, 0.40 if side == "left" else -0.40, 1.05])
    approach = ("approach", [block[0], block[1], block[2] + grasp_height + 0.04])
    descend = ("descend", [block[0], block[1], block[2] + grasp_height])
    if not displaced:
        return [edge, ("source_high", [block[0], block[1], 1.02]), approach, descend]
    low = approach[1][2]
    forward, outward, _ = lift_route(
        block, build_xy, side, grasp_height=grasp_height, displaced=True
    )[2][1]
    return [
        edge,
        ("outward_high", [forward, outward, 1.02]),
        ("outward_low", [forward, outward, low]),
        ("source_front", [forward, block[1], low]),
        approach,
        descend,
    ]


def pick_route(
    task, level, block, build_xy, side, cooperative_level, *, grasp_height=0.06, displaced=False
):
    bottom = TABLE_TOP + sum(target.dimensions_m[2] for target in task.targets[:level])
    height = task.targets[level].dimensions_m[2]
    release = bottom + height / 2 + grasp_height
    lifting = lift_route(block, build_xy, side, grasp_height=grasp_height, displaced=displaced)
    raised = lifting[-1][1]
    lift = raised[2]
    carry = max(lift, release + 0.030)
    retreat = max(release + height / 2 + 0.025, 1.02)
    return [
        *(
            point
            for _, point in source_route(
                block, build_xy, side, grasp_height=grasp_height, displaced=displaced
            )
        ),
        *(point for _, point in lifting),
        [raised[0], raised[1], carry],
        *(
            [
                [build_xy[0], build_xy[1] + (y if side == "left" else -y), carry]
                for y in (0.20, 0.10)
            ]
            if level >= cooperative_level
            else []
        ),
        *([[*build_xy, lift]] if release + 0.025 < lift else []),
        [*build_xy, release + 0.025],
        [*build_xy, release],
        [
            build_xy[0],
            build_xy[1]
            + ((0.15 if side == "left" else -0.15) if level == len(task.targets) - 1 else 0),
            retreat,
        ],
        [
            build_xy[0],
            build_xy[1] + (0.15 if side == "left" else -0.15),
            retreat,
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
                    pick_route(
                        task,
                        level,
                        block,
                        xy,
                        side,
                        cooperative_level,
                        grasp_height=reference.grasp_height(
                            side, task.targets[level].dimensions_m[2], yaw
                        ),
                    ),
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
