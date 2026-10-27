"""Prepare train-only normalization and the measured motor interface before learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS
from vla_lego.pretrained.campaign_data import CampaignData
from vla_lego.pretrained.pilot_campaign import write_json


def measured_interface(metadata: dict) -> dict:
    positions = [*ARM_JOINTS, "NECK_Z", "NECK_Y", *JAW_JOINTS]
    motors = [*ARM_JOINTS, *JAW_JOINTS]
    if (
        metadata["state_names"] != positions
        or metadata["action_names"] != motors
        or metadata["fps"] != 20
        or metadata["privileged_policy_inputs"]
    ):
        raise ValueError("Campaign observations/actions differ from the measured Alex interface")
    return {
        "position_names": positions,
        "state_names": [*positions, *(name + "_velocity" for name in ARM_JOINTS)],
        "action_joint_names": motors,
        "include_arm_velocity": True,
        "control_hz": 20,
        "action_representation": "observation_arm_offset",
    }


def prepare(corpus: Path, statistics_path: Path) -> dict:
    # This constructor enforces all620 admissions, splits and actual-correction coverage.
    training = CampaignData(corpus, "train", seed=20261026)
    interface = measured_interface(training.episodes[0].metadata)
    entries = json.loads((corpus / "admitted-episodes.json").read_text())["episodes"]
    for entry in entries:
        metadata = json.loads((corpus / entry["recording"] / "episode.json").read_text())
        if measured_interface(metadata) != interface:
            raise ValueError("Admitted episodes have incompatible measured interfaces")
    normalization = {
        "action_representation": "observation_arm_offset",
        "statistics": training.statistics(),
    }
    outputs = {statistics_path: normalization, corpus / "alex_interface.json": interface}
    for path, value in outputs.items():
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("Preserve existing incompatible learning input: " + str(path))
    for path, value in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    return {"admitted": len(entries), "normalization_episodes": len(training.entries)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.corpus, args.statistics)))


if __name__ == "__main__":
    main()
