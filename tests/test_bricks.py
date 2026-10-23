"""Physical regressions for the maintained lightweight-brick task."""

from dataclasses import replace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.bricks.episode import EpisodeManager  # noqa: E402
from sim.bricks.scene import TABLE_TOP  # noqa: E402
from sim.bricks.specs import BrickSpec, load_bricks  # noqa: E402
from sim.bricks.task import evaluate_goal, make_goal  # noqa: E402

pytestmark = pytest.mark.mujoco


@pytest.fixture
def episode():
    return EpisodeManager(include_robot=False)


def test_configured_geometry_mass_and_native_contacts(episode):
    for spec, geom, body in zip(episode.bricks, episode.geom_ids, episode.body_ids, strict=True):
        np.testing.assert_allclose(episode.model.geom_size[geom], np.array(spec.size) / 2)
        assert episode.model.body_mass[body] == pytest.approx(spec.mass)
    assert episode.model.neq == 0
    assert episode.model.njnt == len(episode.bricks)
    assert episode.data.ncon > 0


@pytest.mark.parametrize("value", [0, -0.01, float("nan"), float("inf")])
def test_invalid_physical_dimensions_rejected(value):
    with pytest.raises(ValueError):
        BrickSpec("invalid", (value, 0.04, 0.04), 0.02)


def test_seeded_reset_clears_motion_forces_and_preserves_variation(episode):
    episode.reset(seed=13)
    first = episode.data.qpos.copy()
    episode.data.xfrc_applied[:] = 1
    episode.data.qvel[:] = 1
    episode.reset(seed=13)
    np.testing.assert_array_equal(episode.data.qpos, first)
    assert not episode.data.xfrc_applied.any()
    episode.reset(seed=14)
    assert not np.array_equal(episode.data.qpos, first)
    for spec, body in zip(episode.bricks, episode.body_ids, strict=True):
        assert episode.data.xpos[body, 2] == pytest.approx(TABLE_TOP + spec.size[2] / 2, abs=0.001)


@pytest.mark.parametrize("shape", ["tower", "triangle"])
def test_initialized_layout_remains_supported_and_stable(episode, shape):
    goal = make_goal(episode.bricks, shape)
    episode.reset(layout=goal)
    assert evaluate_goal(episode, goal).success


def test_wrong_position_is_not_success(episode):
    goal = make_goal(episode.bricks, "triangle")
    misplaced = replace(
        goal, targets=(replace(goal.targets[0], position=(0.65, 0.2, 0.79)), *goal.targets[1:])
    )
    episode.reset(layout=misplaced)
    result = evaluate_goal(episode, goal)
    assert not result.success
    assert not result.placements[0].success


def test_airborne_or_held_brick_has_no_support(episode):
    goal = make_goal(episode.bricks)
    episode.reset(layout=goal)
    address = episode.qpos_addresses[goal.targets[-1].brick_index]
    episode.data.qpos[address + 2] += 0.1
    mujoco.mj_forward(episode.model, episode.data)
    assert not evaluate_goal(episode, goal).success


def test_stability_window_detects_perturbed_tower(episode):
    goal = make_goal(episode.bricks)
    episode.reset(layout=goal)
    episode.data.xfrc_applied[episode.body_ids[goal.targets[-1].brick_index], 0] = 0.3
    assert not evaluate_goal(episode, goal).success


def test_equivalent_box_yaw_is_accepted(episode):
    goal = make_goal(episode.bricks, "triangle")
    equivalent = replace(goal, targets=tuple(replace(t, yaw=t.yaw + np.pi) for t in goal.targets))
    episode.reset(layout=equivalent)
    assert evaluate_goal(episode, goal).success


def test_missing_placement_and_invalid_task_are_rejected(episode):
    goal = make_goal(episode.bricks)
    with pytest.raises(ValueError, match="each brick"):
        evaluate_goal(episode, replace(goal, targets=goal.targets[:1]))
    with pytest.raises(ValueError, match="three"):
        make_goal(episode.bricks[:2], "triangle")
    with pytest.raises(ValueError, match="Unknown shape"):
        make_goal(episode.bricks, "unsupported")


def test_varied_geometry_and_mass_do_not_use_lego_dimensions():
    spec = BrickSpec("wide", (0.12, 0.05, 0.04), 0.06)
    episode = EpisodeManager([spec], include_robot=False)
    goal = make_goal(episode.bricks)
    episode.reset(layout=goal)
    assert evaluate_goal(episode, goal).success


def test_alex_control_contract_survives_free_brick_joints():
    from sim.sim_runner import SimRunner

    episode = EpisodeManager(load_bricks(), include_robot=True)
    runner = SimRunner(episode.model, episode.data)
    state = runner.step(np.zeros(17))
    assert runner.robot_state.normalize(state.to_flat_array()).shape == (52,)
    assert episode.model.nu == 17
    assert runner.substeps == 25
