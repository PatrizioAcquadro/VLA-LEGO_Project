"""Physical prerequisites and release guards; isolated checks do not prove settling."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from vla_lego.isaac import campaign_faults as faults
from vla_lego.isaac import clearance
from vla_lego.isaac.task import initial_tower_task


@pytest.mark.parametrize(
    "quaternion",
    [
        [0.583294332, 0.399707228, -0.583296001, 0.399709374],
        [2**-0.5, 0, 0, 2**-0.5],
        [0, 2**-0.5, 0, 2**-0.5],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ],
)
def test_uprighting_turn_stands_the_actual_prism_without_a_half_turn(quaternion):
    import math

    from vla_lego.isaac.campaign_faults import upright_turn

    q = np.asarray(quaternion) / np.linalg.norm(quaternion)
    x, y, z, w = q
    long_axis = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    axis, angle = upright_turn([0, 0, 0, *q])
    axis = np.asarray(axis)
    turned = (
        long_axis * math.cos(angle)
        + np.cross(axis, long_axis) * math.sin(angle)
        + axis * np.dot(axis, long_axis) * (1 - math.cos(angle))
    )
    assert np.linalg.norm(turned[:2]) < 1e-6
    assert abs(turned[2]) == pytest.approx(1)
    assert 0 <= angle <= math.pi / 2


@pytest.mark.parametrize(
    "table_force,piece_force,robot_force,speed,ready",
    [
        (0.4, [0, 0, 0], 0, 0, True),
        (0, [0, 0, 0.4], 0, 0.001, True),
        (0, [0, 0, 0], 0, 0, False),
        (0, [0.4, 0, 0], 0, 0, False),
        (0, [0, 0, -0.4], 0, 0, False),
        (0, [0, 0, 0.4], 0.1, 0, False),
        (0, [0, 0, 0.4], 0, 0.03, False),
    ],
)
def test_uprighting_requires_released_resting_support(
    table_force, piece_force, robot_force, speed, ready
):
    from types import SimpleNamespace

    from vla_lego.isaac.campaign_faults import upright_released_prism

    task = initial_tower_task()
    index = next(i for i, p in enumerate(task.placements) if p.spec.shape == "prism")
    states = [{"block_pose_w": [*p.xy_m, 0.85, 0, 0, 0, 1]} for p in task.placements]
    states[index].update(
        block_pose_w=[*task.placements[index].xy_m, 0.85, 2**-0.5, 0, 0, 2**-0.5],
        block_velocity_w=[speed, 0, 0, 0, 0, 0],
        robot_contact_force_n=robot_force,
        nonrobot_contact_force_n=table_force,
        block_contact_forces_n=[piece_force],
    )

    class PlanningReached(Exception):
        pass

    def plan(*args, **kwargs):
        raise PlanningReached

    reference = SimpleNamespace(env=SimpleNamespace(task_truth=states.__getitem__), grasp_yaw=plan)
    if ready:
        with pytest.raises(PlanningReached):
            upright_released_prism(reference, task, index, "left")
    else:
        with pytest.raises(RuntimeError, match="released, settled prism"):
            upright_released_prism(reference, task, index, "left")


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "held,returned,error",
    [
        (0, 0, None),
        (20, 0, None),
        (20, 20, "Tipped prism"),
        (20, 90, "Tipped prism"),
        (40, 0, "settling basin"),
        ("inside", 0, None),
        ("outside", 0, "settling basin"),
    ],
)
def test_measured_tilt_guards_release_and_final_upright_state(
    monkeypatch, side, held, returned, error
):
    task = initial_tower_task()
    index = next(i for i, p in enumerate(task.placements) if p.spec.shape == "prism")
    spec = task.placements[index].spec
    tipping_angle = math.degrees(math.atan(min(spec.dimensions_m[:2]) / spec.dimensions_m[2]))
    if isinstance(held, str):
        held = tipping_angle + (-0.01 if held == "inside" else 0.01)

    def orientation(degrees):
        half = math.radians(degrees) / 2
        return [math.sin(half), 0, 0, math.cos(half)]

    states = [
        {
            "block_pose_w": [*p.xy_m, 0.83, 0, 0, 0, 1],
            "robot_contact_force_n": 0,
            "nonrobot_contact_force_n": 0.3,
            "block_contact_forces_n": [[0, 0, 0] for _ in task.placements],
            "block_velocity_w": [0] * 6,
        }
        for p in task.placements
    ]
    states[index]["block_pose_w"][3:] = orientation(90)
    released = []

    class Pose(list):
        def clone(self):
            return self.copy()

    def move(stage, *args, **kwargs):
        if stage == "upright_lift":
            states[index]["block_pose_w"][2] += 0.1
        elif stage == "upright_turn_0":
            states[index]["block_pose_w"][3:] = orientation(held)

    def release(*args, **kwargs):
        released.append(index)
        states[index]["block_pose_w"][3:] = orientation(returned)

    reference = SimpleNamespace(
        env=SimpleNamespace(task_truth=states.__getitem__),
        ctrl=SimpleNamespace(pose=lambda side: [Pose([0.4, 0, 1, 0, 0, 0, 1])]),
        grasp_yaw=lambda *args, **kwargs: 0,
        grip=lambda *args, **kwargs: None,
        move=move,
    )
    monkeypatch.setattr(clearance, "source_grasp_span", lambda *args: spec.dimensions_m[0])
    monkeypatch.setattr(faults, "_upright_plan", lambda *args: [[0.4, 0, 1, 0, 0, 0, 1]])
    monkeypatch.setattr(faults, "return_held_piece", release)
    if error:
        with pytest.raises(RuntimeError, match=error):
            faults.upright_released_prism(reference, task, index, side)
    else:
        faults.upright_released_prism(reference, task, index, side)
    assert released == ([] if error == "settling basin" else [index])
