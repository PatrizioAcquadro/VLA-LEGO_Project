"""Index usable full pilot episodes while retaining every collection attempt."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from vla_lego.isaac.environment import ARM_JOINTS, JAW_JOINTS
from vla_lego.isaac.pilot import PilotCase, load_cases
from vla_lego.pretrained.decisions import DECISION_NAMES


def audit_episode(directory: Path, case: PilotCase) -> dict:
    report = json.loads((directory / "report.json").read_text())
    metadata = json.loads((directory / "recording/episode.json").read_text())
    if (
        metadata["fps"] != 20
        or metadata["state_names"] != [*ARM_JOINTS, "NECK_Z", "NECK_Y", *JAW_JOINTS]
        or metadata["action_names"] != [*ARM_JOINTS, *JAW_JOINTS]
    ):
        raise ValueError("Episode differs from the recorded named Alex interface")
    if report.get("pilot_case") != case.row or metadata.get("collection_case") != case.row:
        raise ValueError("Recorded case differs from the approved case/split")
    if (metadata["generation_family"], metadata["split"]) != (
        case.row["family"],
        case.row["split"],
    ):
        raise ValueError("Episode family/split mismatch")
    if not report["passed"] or metadata["privileged_policy_inputs"]:
        raise ValueError("Physical case failed or policy boundary violated")
    if case.category == "stop":
        if not metadata["verified_stop"] or metadata["verified_success"]:
            raise ValueError("A stop must be distinct from tower success")
    else:
        events = report["evaluation"]["cooperation_events"]
        if (
            not metadata["verified_success"]
            or not report["evaluation"]["passed"]
            or not any(
                e["stabilizer"] == case.row["stabilizer"]
                and e["upper_level"] == case.cooperative_level
                for e in events
            )
        ):
            raise ValueError("Missing physical completion or specified cooperative event")
    if case.category == "recovery" and not (
        metadata.get("teacher_induced_fault") and report.get("fault", {}).get("physically_verified")
    ):
        raise ValueError("Corrective episode lacks its physical fault")
    stages: Counter = Counter()
    decisions: Counter = Counter()
    masked = count = 0
    previous = None
    allowed = {
        "frame",
        "observation_time_s",
        "image_time_s",
        "command_time_s",
        "head_rgb",
        "joint_position",
        "joint_velocity",
        "action",
        "requested_action",
        "task_decision",
    }
    with (
        (directory / "recording/frames.jsonl").open() as frames,
        (directory / "recording/annotations.jsonl").open() as annotations,
        (directory / "recording/privileged.jsonl").open() as evidence,
    ):
        for count, raw in enumerate(frames, 1):
            row, label, truth = (
                json.loads(raw),
                json.loads(next(annotations)),
                json.loads(next(evidence)),
            )
            t = row["observation_time_s"]
            if (
                set(row) != allowed
                or row["frame"] != count - 1
                or label["frame"] != count - 1
                or truth["frame"] != count - 1
            ):
                raise ValueError("Unexpected policy field or unsynchronized annotation")
            if (
                row["image_time_s"] > t + 1e-8
                or row["command_time_s"] > t + 1e-8
                or abs(row["image_time_s"] - t) > 1e-8
                or previous is not None
                and abs(t - previous - 0.05) > 1e-8
            ):
                raise ValueError("Noncausal or discontinuous 20 Hz sensor/action clock")
            for key, width in (
                ("joint_position", 18),
                ("joint_velocity", 18),
                ("action", 16),
                ("requested_action", 16),
            ):
                if len(row[key]) != width or not all(math.isfinite(v) for v in row[key]):
                    raise ValueError("Invalid recorded state/action")
            if (
                not (directory / "recording" / row["head_rgb"]).is_file()
                or len(truth["blocks"]) != 8
            ):
                raise ValueError("Missing image or incorrect physical inventory")
            stages[label["stage"]] += 1
            decisions[row["task_decision"]] += 1
            masked += not label["positive_imitation"]
            previous = t
        if next(annotations, None) or next(evidence, None) or count != metadata["frames"]:
            raise ValueError("Frame/annotation/evidence counts differ")
    if not count or (case.category == "recovery") != (masked > 0):
        raise ValueError("Empty episode or incorrect error-supervision mask")
    if not decisions["abstain" if case.category == "stop" else "complete"]:
        raise ValueError("Missing recorded terminal decision")
    return {
        "path": str(directory / "recording"),
        "frames": count,
        "masked_error_frames": masked,
        "positive_frames": count - masked,
        "stage_frames": dict(stages),
        "decision_frames": dict(decisions),
        "instruction": metadata["instruction"],
        "family": case.row["family"],
        "split": case.row["split"],
        "category": case.category,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    result: dict = {
        "complete": False,
        "episodes": {},
        "attempts": [],
        "missing_cases": [],
        "counts": {},
    }
    counts: Counter = Counter()
    target_coverage: Counter = Counter()
    nominal_roles: Counter = Counter()
    cooperative_levels: Counter = Counter()
    recovery_levels: Counter = Counter()
    for case in cases:
        for directory in sorted((args.root / "attempts").glob(case.name + "-*")):
            attempt = {"case": case.name, "path": str(directory), "usable": False}
            try:
                episode = audit_episode(directory, case)
                attempt["usable"] = True
                if case.name not in result["episodes"]:
                    result["episodes"][case.name] = episode
                    counts[(case.row["split"], case.category)] += 1
                    if case.category != "stop":
                        target_coverage.update(
                            f"{case.row['split']}/{t.description}" for t in case.task().targets
                        )
                        cooperative_levels[f"{case.row['split']}/{case.cooperative_level + 1}"] += 1
                    if case.category == "nominal":
                        nominal_roles[f"{case.row['split']}/{case.row['stabilizer']}"] += 1
                    elif case.category == "recovery":
                        recovery_levels[
                            f"{case.row['split']}/{case.row['error']}/level{case.error_level + 1}"
                        ] += 1
            except (ValueError, KeyError, OSError, StopIteration) as error:
                attempt["reason"] = str(error)
            result["attempts"].append(attempt)
        if case.name not in result["episodes"]:
            result["missing_cases"].append(case.name)
    result["counts"] = {f"{split}/{category}": count for (split, category), count in counts.items()}
    result["complete"] = len(result["episodes"]) == 60
    result["coverage"] = {
        "target_combinations": dict(target_coverage),
        "nominal_stabilizers": dict(nominal_roles),
        "cooperative_placement_levels": dict(cooperative_levels),
        "recovery_levels": dict(recovery_levels),
    }
    result["collection_yield"] = {
        "usable_cases": len(result["episodes"]),
        "attempts": len(result["attempts"]),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["episodes"]:
        positions = [*ARM_JOINTS, "NECK_Z", "NECK_Y", *JAW_JOINTS]
        interface = {
            "action_representation": "absolute",
            "action_joint_names": [*ARM_JOINTS, *JAW_JOINTS],
            "position_names": positions,
            "state_names": positions + [f"{name}_VELOCITY" for name in ARM_JOINTS],
            "include_arm_velocity": True,
            "control_hz": 20,
            "task_decision_names": list(DECISION_NAMES),
        }
        args.output.with_name("alex_interface.json").write_text(
            json.dumps(interface, indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("complete", "counts", "collection_yield", "missing_cases")
            }
        )
    )


if __name__ == "__main__":
    main()
