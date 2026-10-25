"""Essential masked-window and privileged-input boundaries for later adaptation."""

import json

import numpy as np
import pytest
from PIL import Image

from vla_lego.pretrained.pilot_dataset import PilotEpisode


def recording(tmp_path, *, admitted=True):
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "head.png")
    metadata = {
        "verified_success": admitted,
        "supervision_role": "verified_correction" if admitted else "context_only",
        "frames": 3,
        "instruction": "Build the instructed tower.",
    }
    (tmp_path / "episode.json").write_text(json.dumps(metadata))
    rows = [
        {
            "head_rgb": "head.png",
            "joint_position": [0] * 18,
            "joint_velocity": [0] * 18,
            "action": [i + 1] * 16,
            "task_decision": "complete" if i == 2 else "continue",
        }
        for i in range(3)
    ]
    (tmp_path / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "annotations.jsonl").write_text(
        "".join(
            json.dumps({"positive_imitation": i != 1, "stage": "hidden teacher stage"}) + "\n"
            for i in range(3)
        )
    )


def test_future_fault_ends_positive_chunk_and_correction_uses_its_own_observation(tmp_path):
    recording(tmp_path)
    episode = PilotEpisode(tmp_path, horizon=4)
    assert len(episode) == 2
    before, correction = episode[0], episode[1]
    assert before["action_is_pad"].tolist() == [False, True, True, True]
    assert before["action"][0, 0] == 1
    assert correction["action"][0, 0] == 3
    assert correction["action"][0, -2:].tolist() == [1, 0]
    assert correction["observation.state"].shape == (32,)
    assert correction["observation.images.head"].shape == (3, 8, 8)
    assert not {"stage", "task_decision", "blocks", "positive_imitation"} & before.keys()


def test_failed_attempt_cannot_be_loaded_as_positive_data(tmp_path):
    recording(tmp_path, admitted=False)
    with pytest.raises(ValueError, match="Only admitted"):
        PilotEpisode(tmp_path)


def test_diagnostics_cover_startup_and_first_acquisition_not_only_later_blocks():
    from types import SimpleNamespace

    from vla_lego.pretrained.pilot_training_data import PilotTrainingData

    stages = ["left_park", "level0_approach", "level0_grip", "level0_lift"]
    stages += ["level2_grip"] * 6 + ["complete"]
    data = PilotTrainingData.__new__(PilotTrainingData)
    data.episodes = [
        SimpleNamespace(
            eligible_frames=list(range(len(stages))),
            annotations=[{"stage": stage} for stage in stages],
        )
    ]
    data.pools = [{"park": [0], "move": [1, 3], "grasp": [2, *range(4, 10)], "terminal": [10]}]
    samples = data.diagnostic_samples()
    indices = [sample for _, sample, _ in samples]
    assert {0, 1, 2, 3, 10} <= set(indices)
    assert len(indices) == len(set(indices))


def test_future_offsets_share_the_current_observation_anchor_and_stop_at_errors(tmp_path):
    recording(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    rows[0]["joint_position"] = [0.25] * 18
    rows[2]["joint_position"] = [0.5] * 18
    (tmp_path / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    episode = PilotEpisode(tmp_path, horizon=3, action_representation="observation_arm_offset")
    before, after = episode[0], episode[1]
    assert before["action"][0, 0] == 0.75
    assert before["action"][0, 14] == 1.0
    assert after["action"][0, 0] == 2.5
    assert before["action_is_pad"].tolist() == [False, True, True]

    from vla_lego.pretrained.pilot_training_data import PilotTrainingData

    data = PilotTrainingData.__new__(PilotTrainingData)
    data.split, data.action_representation = "train", "observation_arm_offset"
    data.episodes = [episode]
    stats = data.statistics()
    assert stats["action"]["mean"][0] == pytest.approx(1.625)
    assert stats["action"]["std"][0] == pytest.approx(0.875)
    assert stats["action"]["mean"][14] == pytest.approx(2.0)


def test_offset_chunk_uses_one_anchor_rather_than_each_future_measured_position(tmp_path):
    recording(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    for i, row in enumerate(rows):
        row["joint_position"] = [i * 0.5] * 18
    (tmp_path / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "annotations.jsonl").write_text(
        "".join(json.dumps({"positive_imitation": True, "stage": "move"}) + "\n" for _ in rows)
    )
    episode = PilotEpisode(tmp_path, horizon=2, action_representation="observation_arm_offset")
    # At t=1, q=0.5: targets 2 and3 are both relative to0.5, not future q=1.
    assert episode[1]["action"][:, 0].tolist() == [1.5, 2.5]
    assert episode[1]["action"][:, 14].tolist() == [2.0, 3.0]


def test_conditional_handoff_report_serializes_numpy_measurements(tmp_path):
    from vla_lego.isaac.pilot_scenario import replay_demonstration_handoff

    recording(tmp_path)
    (tmp_path / "annotations.jsonl").write_text(
        "".join(
            json.dumps({"positive_imitation": True, "stage": "move" if i < 2 else "grip"}) + "\n"
            for i in range(3)
        )
    )
    truth = {"block_pose_w": [0.0] * 7}
    (tmp_path / "privileged.jsonl").write_text((json.dumps({"blocks": [truth]}) + "\n") * 3)

    class Environment:
        blocks = [0]

        def capture(self):
            pass

        def set_joint_targets(self, targets):
            assert len(targets) == 16

        def step(self, count):
            assert count == 10

        def observe(self):
            return {"joint_position": np.zeros(18, dtype=np.float32)}

        def task_truth(self, index):
            return truth

    report = replay_demonstration_handoff(Environment(), tmp_path, "grip", lambda _: None)
    assert report["teacher_prefix_ticks"] == 2
    assert not report["autonomous_from_start"]
    assert json.loads(json.dumps(report))["prefix_joint_error_rad"] == 0.0


def correction_cohort(root):
    entries = []
    for index, category in enumerate(("nominal", "recovery", "stop", "policy", "policy")):
        directory = root / "attempts" / f"episode-{index}" / "recording"
        directory.mkdir(parents=True)
        recording(directory)
        stages = ["left_park"] * 39 + ["level0_approach"] * 229
        stages += ["level0_grip"] * 4 + ["level0_lift"] * 140 + ["verified_complete"]
        metadata = json.loads((directory / "episode.json").read_text())
        metadata.update(
            frames=len(stages),
            split="train",
            generation_family=f"T{index % 2 + 1}",
            collection_condition="marker_free_continuous_site_v1",
            collection_case={"category": "nominal" if category == "policy" else category},
            verified_success=category != "stop",
            verified_stop=category == "stop",
            supervision_role=(
                "verified_correction"
                if category in ("policy", "recovery")
                else "verified_stop" if category == "stop" else "verified_nominal"
            ),
        )
        if category == "policy":
            metadata["correction_source"] = {"exact_native_parity": True, "source_run": "parent"}
        (directory / "episode.json").write_text(json.dumps(metadata))
        rows = []
        annotations = []
        for frame, stage in enumerate(stages):
            rows.append(
                {
                    "head_rgb": "head.png",
                    "joint_position": [0] * 18,
                    "joint_velocity": [0] * 18,
                    "action": [0] * 16,
                    "task_decision": (
                        ("abstain" if category == "stop" else "complete")
                        if frame == len(stages) - 1
                        else "continue"
                    ),
                }
            )
            annotations.append(
                {"stage": stage, "positive_imitation": not (category == "recovery" and frame == 1)}
            )
        (directory / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (directory / "annotations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in annotations)
        )
        entries.append({"case": f"case-{index}", "path": f"attempts/episode-{index}/report.json"})
    (root / "train-episodes.json").write_text(json.dumps(entries))
    cohort = {
        "name": "correction-cohort",
        "episode_counts": {"train": 5, "development": 3},
        "policy_correction_probability": 0.2,
        "policy_acquisition_probability": 0.5,
    }
    (root / "cohort.json").write_text(json.dumps(cohort))


def test_declared_corrections_sample_real_acquisition_beyond_the_initial_five_seconds(tmp_path):
    from collections import Counter

    from vla_lego.pretrained.pilot_training_data import PilotTrainingData

    correction_cohort(tmp_path)
    data = PilotTrainingData(tmp_path, "train", seed=20261024)
    counts = Counter()
    buckets = Counter()
    for draw in range(10000):
        number, sample, bucket = data.select(draw)
        counts[data.membership[number]["sampling_category"]] += 1
        if bucket.startswith("policy_"):
            buckets[bucket] += 1
            if bucket in ("policy_grasp", "policy_lift"):
                assert data.episodes[number].eligible_frames[sample] >= 268
        assert data.select(draw) == (number, sample, bucket)
    assert 1800 < counts["policy_correction"] < 2200
    assert set(buckets) == {"policy_start", "policy_approach", "policy_grasp", "policy_lift"}
    assert all(value > 100 for value in buckets.values())
    assert set(counts) == {"nominal", "recovery", "stop", "policy_correction"}
    assert data.episodes[3].metadata["collection_case"]["category"] == "nominal"
    assert data.protocol["cohort"]["name"] == "correction-cohort"
    assert data.sampling_description["base_probability"] == 0.8
    assert not {"stage", "sampling_category", "correction_source"} & data[0].keys()


def test_new_episode_membership_requires_an_explicit_cohort_and_rejects_aliases(tmp_path):
    from vla_lego.pretrained.pilot_training_data import PilotTrainingData

    correction_cohort(tmp_path)
    path = tmp_path / "train-episodes.json"
    entries = json.loads(path.read_text())
    entries[-1]["path"] = entries[-2]["path"]
    path.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="membership"):
        PilotTrainingData(tmp_path, "train", seed=1)
    (tmp_path / "cohort.json").unlink()
    with pytest.raises(ValueError, match="membership"):
        PilotTrainingData(tmp_path, "train", seed=1)


def test_resume_protocol_records_changes_to_declared_sampling(tmp_path):
    from vla_lego.pretrained.pilot_training_data import PilotTrainingData

    correction_cohort(tmp_path)
    saved = PilotTrainingData(tmp_path, "train", seed=1).protocol
    path = tmp_path / "cohort.json"
    cohort = json.loads(path.read_text())
    cohort["policy_correction_probability"] = 0.3
    path.write_text(json.dumps(cohort))
    assert PilotTrainingData(tmp_path, "train", seed=1).protocol != saved


def test_extra_descendants_do_not_overweight_one_correction_family(tmp_path):
    import shutil
    from collections import Counter

    from vla_lego.pretrained.pilot_training_data import PilotTrainingData

    correction_cohort(tmp_path)
    shutil.copytree(tmp_path / "attempts/episode-4", tmp_path / "attempts/episode-5")
    path = tmp_path / "train-episodes.json"
    entries = json.loads(path.read_text())
    entries.append({"case": "case-5", "path": "attempts/episode-5/report.json"})
    path.write_text(json.dumps(entries))
    path = tmp_path / "cohort.json"
    cohort = json.loads(path.read_text())
    cohort["episode_counts"]["train"] = 6
    path.write_text(json.dumps(cohort))
    data = PilotTrainingData(tmp_path, "train", seed=20261024)
    counts = Counter()
    for draw in range(10000):
        number, _, _ = data.select(draw)
        if data.membership[number]["sampling_category"] == "policy_correction":
            counts[data.episodes[number].metadata["generation_family"]] += 1
    assert 0.45 < counts["T1"] / sum(counts.values()) < 0.55
