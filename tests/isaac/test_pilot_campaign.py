"""The global learning schedule and stopping decisions survive allocation boundaries."""

import pytest

from vla_lego.pretrained.pilot_campaign import (
    learning_rate_factor,
    prune_checkpoints,
)


def test_global_schedule_has_one_warmup_and_finite_decay_floor():
    assert learning_rate_factor(0) == 0.001
    assert learning_rate_factor(999) == 1.0
    assert learning_rate_factor(1_000) == 1.0
    assert learning_rate_factor(30_000) > learning_rate_factor(60_000)
    assert learning_rate_factor(100_000) == pytest.approx(0.025)


def test_pruning_retains_the_loaded_resume_source_after_latest_moves(tmp_path):
    for step in (0, 3000, 4000, 5000, 6000, 7000):
        checkpoint = tmp_path / f"{step:06d}"
        checkpoint.mkdir()
        (checkpoint / "state.json").write_text("{}")
    latest = tmp_path / "latest"
    latest.symlink_to("005000")
    loaded = latest.resolve()
    latest.unlink()
    latest.symlink_to("007000")
    prune_checkpoints(tmp_path, 6000, resume_checkpoint=loaded)
    assert {p.name for p in tmp_path.iterdir()} == {
        "000000",
        "005000",
        "006000",
        "007000",
        "latest",
    }


def test_pruning_preserves_physical_best_when_later_offline_selection_differs(tmp_path):
    for step in (0, 20000, 35000, 45000, 50000, 55000, 60000):
        checkpoint = tmp_path / f"{step:06d}"
        checkpoint.mkdir()
        (checkpoint / "state.json").write_text("{}")
    (tmp_path / "physical-best").symlink_to("035000")
    prune_checkpoints(tmp_path, 45000, resume_checkpoint=tmp_path / "050000")
    assert (tmp_path / "035000/state.json").is_file()
    assert (tmp_path / "physical-best").resolve() == tmp_path / "035000"
    assert (tmp_path / "045000/state.json").is_file()
    assert (tmp_path / "050000/state.json").is_file()


def test_pruning_rejects_foreign_physical_pointer_before_deleting_anything(tmp_path):
    import pytest

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for step in (35000, 55000, 60000):
        checkpoint = checkpoints / str(step)
        checkpoint.mkdir()
        (checkpoint / "state.json").write_text("{}")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "state.json").write_text("{}")
    (checkpoints / "physical-best").symlink_to(foreign)
    with pytest.raises(ValueError, match="complete checkpoint"):
        prune_checkpoints(checkpoints, 55000)
    assert (checkpoints / "35000/state.json").is_file()
