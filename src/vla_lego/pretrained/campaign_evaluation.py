"""Bounded physical panels with persistent failed and conditional outcomes."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from vla_lego.pretrained.decisions import DECISION_NAMES

PANEL_COUNTS = {
    "single_block_control": {"single_block": 10},
    "intermediate_calibration": {"single_block": 2, "tower": 2, "recovery": 1, "stop": 1},
    "final_confirmation": {"single_block": 6, "tower": 8, "recovery": 3, "stop": 3},
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_cases(protocol: dict, panel: str, declaration: list[dict]) -> list[dict]:
    """Use the cases and budgets selected before any learned result."""
    cases: list[dict] = protocol["panels"][panel]
    definitions = {case["row"]["case"]: case for case in declaration}
    if (
        not protocol["selected_before_campaign_learning"]
        or Counter(row["category"] for row in cases) != PANEL_COUNTS[panel]
        or len({row["case"] for row in cases}) != len(cases)
    ):
        raise ValueError("Physical panel selection differs from the frozen protocol")
    for row in cases:
        partition = row["partition"]
        if (
            definitions[row["case"]] != row["case_definition"]
            or row["conditional_recovery"] != (row["category"] == "recovery")
            or not row["budget_frozen_before_any_candidate_result"]
            or not isinstance(row["autonomous_tick_budget"], int)
            or row["autonomous_tick_budget"] < 1
            or row["native_wall_bound_s"] != 5400
            or (panel == "intermediate_calibration" and partition != "calibration")
            or (panel == "final_confirmation" and partition != "confirmation")
        ):
            raise ValueError("Changed declaration, partition, initialization or rollout budget")
    if panel == "single_block_control" and (
        Counter(row["start"] for row in cases) != {"familiar": 5, "held_out": 5}
        or {row["active_arm"] for row in cases} != {"left", "right"}
        or any(
            row["partition"] != ("train" if row["start"] == "familiar" else "calibration")
            for row in cases
        )
    ):
        raise ValueError("Control requires five familiar and five held-out starts across both arms")
    return cases


def total_probe_ticks(case: dict) -> int:
    # tower_rollout subtracts its replayed prefix and forty terminal hold ticks.
    return int(case["autonomous_tick_budget"] + case["teacher_recovery_prefix_ticks"] + 40)


def execution_binding(root: Path, checkpoint: Path, decision: Path | None, protocol: Path) -> dict:
    """A cached reference is reusable only with identical model, head and runtime."""
    weights = (
        checkpoint / "pretrained_model"
        if (checkpoint / "pretrained_model").is_dir()
        else checkpoint
    )
    for name in ("config.json", "model.safetensors", "alex_interface.json"):
        if not (weights / name).is_file():
            raise FileNotFoundError(weights / name)
    files = {
        "motor/" + str(path.relative_to(weights)): file_digest(path)
        for path in sorted(weights.rglob("*"))
        if path.is_file()
    }
    files["decision"] = file_digest(decision) if decision else "historical_embedded_decisions"
    files["panel_protocol"] = file_digest(protocol)
    for path in sorted((root / "src/vla_lego").rglob("*.py")):
        files[str(path.relative_to(root))] = file_digest(path)
    for name in ("run_policy_probe.py", "evaluate_campaign_panel.py", "isaac-run.sh"):
        files["scripts/" + name] = file_digest(root / "scripts" / name)
    return files


def retain_binding(directory: Path, binding: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "execution-binding.json"
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError("Preserve cached outcomes: motor/head/panel/runtime identity changed")
    else:
        if any(directory.iterdir()):
            raise ValueError("Existing results lack a verifiable execution identity")
        path.write_text(json.dumps(binding, indent=2) + "\n")


def run_case_once(directory: Path, command: list[str]) -> None:
    if directory.exists():
        return
    error = "Probe ended before output creation"
    try:
        subprocess.run(command, check=False)
    except OSError as exception:
        error = str(exception)
    finally:
        if not directory.exists():
            directory.mkdir(parents=True)
            (directory / "report.json").write_text(json.dumps({"error": error}) + "\n")


def trace_progress(directory: Path, task: dict, *, historical_reference=False) -> dict:
    """Evaluator-only acquisition/release events; these never replace task scoring."""
    from vla_lego.isaac.catalog import BlockSpec
    from vla_lego.isaac.environment import TABLE_TOP
    from vla_lego.isaac.evaluation import CONTACT_ZERO_N, POSITION_TOLERANCE_M

    trace = directory / "rollout/trace.jsonl"
    if not trace.exists() or not task:
        return {"available": False}
    targets = [
        next(i for i, p in enumerate(task["placements"]) if p["spec"] == target)
        for target in task["targets"]
        if any(p["spec"] == target for p in task["placements"])
    ]
    heights = [BlockSpec(**p["spec"]).dimensions_m[2] for p in task["placements"]]
    expected = {}
    bottom = TABLE_TOP
    for index in targets:
        expected[index] = bottom + heights[index] / 2
        bottom += heights[index]
    acquisitions: list[dict] = []
    grasps: list[dict] = []
    releases: list[dict] = []
    previous_grasps: set[int] | None = None
    previous_lifted: set[int] | None = None
    released_indices = set()
    initial_released = set()
    trace_rows = command_rows = 0
    finite_motor_trace = True
    max_head = 0.0
    with trace.open() as stream:
        for line in stream:
            row = json.loads(line)
            trace_rows += 1
            command_rows += "tick" in row
            if "tick" in row:
                action = row.get("learned_action", [])
                width = 18 if historical_reference else 16
                finite_motor_trace &= len(action) == width and all(math.isfinite(v) for v in action)
            blocks, robot = row["blocks"], row["robot"]
            paths = robot["robot_contact_paths"]
            if sum(path.rsplit("/", 1)[-1] == "HEAD_LINK" for path in paths) != 1:
                raise ValueError("Physical progress requires the named head contact channel")
            jaws = [
                [
                    i
                    for i, path in enumerate(paths)
                    if path.rsplit("/", 1)[-1]
                    in (f"{side}_WSG32_NEGATIVE_JAW_LINK", f"{side}_WSG32_POSITIVE_JAW_LINK")
                ]
                for side in ("left", "right")
            ]
            if any(len(indices) != 2 for indices in jaws):
                raise ValueError("Physical progress requires both named jaw pairs")
            max_head = max(
                [
                    max_head,
                    *(
                        math.hypot(*force)
                        for path, force in zip(paths, robot["robot_contact_forces_n"], strict=True)
                        if path.rsplit("/", 1)[-1] == "HEAD_LINK"
                    ),
                ]
            )
            held = {
                i
                for i, block in enumerate(blocks)
                if any(
                    min(math.hypot(*block["robot_contact_forces_n"][j]) for j in pair)
                    > CONTACT_ZERO_N
                    and sum(math.hypot(*block["robot_contact_forces_n"][j]) for j in pair) > 0.5
                    for pair in jaws
                )
            }
            lifted = {
                i for i in held if blocks[i]["block_pose_w"][2] - TABLE_TOP - heights[i] / 2 >= 0.07
            }
            if "tick" in row:
                for collection, current, previous in (
                    (grasps, held, previous_grasps),
                    (acquisitions, lifted, previous_lifted),
                ):
                    if previous is not None:
                        collection.extend(
                            {
                                "tick": row["tick"],
                                "object_index": i,
                                "requested_piece": i in targets,
                            }
                            for i in sorted(current - previous)
                        )
            previous_grasps, previous_lifted = held, lifted
            region = task["build_region_xy_m"]
            for level, index in enumerate(targets):
                block = blocks[index]
                x, y, z = block["block_pose_w"][:3]
                support = (
                    block["nonrobot_contact_force_n"]
                    if level == 0
                    else block["block_contact_forces_n"][targets[level - 1]][2]
                )
                if (
                    abs(z - expected[index]) >= POSITION_TOLERANCE_M
                    or support <= CONTACT_ZERO_N
                    or (
                        level == 0
                        and not (region[0] <= x <= region[1] and region[2] <= y <= region[3])
                    )
                ):
                    break
                if (
                    block["robot_contact_force_n"] < CONTACT_ZERO_N
                    and index not in released_indices
                ):
                    released_indices.add(index)
                    if trace_rows == 1:
                        initial_released.add(index)
                    else:
                        releases.append(
                            {
                                "time_s": row["time_s"],
                                "target_level": level + 1,
                                "object_index": index,
                            }
                        )
    return {
        "available": previous_grasps is not None,
        "trace_rows": trace_rows,
        "command_rows": command_rows,
        "finite_sixteen_motor_trace": finite_motor_trace,
        "bilateral_grasp_events": grasps,
        "correct_requested_piece_grasps": sum(event["requested_piece"] for event in grasps),
        "lifted_acquisition_events": acquisitions,
        "correct_requested_piece_acquisitions": sum(
            event["requested_piece"] for event in acquisitions
        ),
        "wrong_piece_acquisitions": sum(not event["requested_piece"] for event in acquisitions),
        "released_ordered_target_events": releases,
        "unique_new_released_ordered_targets": len(released_indices - initial_released),
        "initial_released_ordered_targets": len(initial_released),
        "max_head_contact_n": max_head,
        "initial_contacts_excluded_from_new_acquisitions": True,
        "release_events_are_not_stable_full_task_success": True,
    }


def _read_report(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON report object")
        return value
    except (OSError, ValueError) as error:
        return {"error": str(error)}


def outcome(case: dict, directory: Path, *, historical_reference=False) -> dict:
    """A terminal word or teacher prefix alone never establishes learned success."""
    report_path = directory / "rollout/report.json"
    result = _read_report(report_path)
    probe_path = directory / "report.json"
    probe = _read_report(probe_path)
    started = result.get("autonomy_started") is True
    conditional = (
        result.get("teacher_commands") is True or result.get("teacher_prefix_ticks", 0) > 0
    )
    evaluation = result.get("evaluation", {})
    terminal = result.get("terminal_decision")
    physical = evaluation.get("passed") is True and terminal == "complete"
    if case["category"] == "stop":
        physical = result.get("justified_stop") is True and terminal == "abstain"
    valid_start = (
        conditional == case["conditional_recovery"]
        and result.get("autonomous_from_start") is (not conditional)
        and (not conditional or result.get("recovery_fault_present") is True)
        and result.get("teacher_prefix_ticks", 0) == case["teacher_recovery_prefix_ticks"]
    )
    separate_decisions = result.get("separate_categorical_decisions") is True
    valid_interface = separate_decisions or (
        historical_reference
        and result.get("separate_categorical_decisions") is False
        and result.get("task_decision_names") == list(DECISION_NAMES)
    )
    try:
        progress = trace_progress(
            directory, result.get("task", {}), historical_reference=historical_reference
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        progress = {"available": False, "error": str(error)}
    executed = result.get("executed_ticks", 0)
    full_trace = (
        progress["available"]
        and progress["finite_sixteen_motor_trace"]
        and isinstance(executed, int)
        and executed >= 0
        and progress["max_head_contact_n"] <= 1e-6
        and executed <= progress["command_rows"] <= executed + 1
        and progress["trace_rows"] - progress["command_rows"] == 40
    )
    success = bool(
        started
        and valid_start
        and valid_interface
        and full_trace
        and result.get("case_passed") is True
        and physical
    )
    return {
        "case": case["case"],
        "category": case["category"],
        "partition": case["partition"],
        "start": case["start"],
        "active_arm": case["active_arm"],
        "availability": "executed" if started else "unavailable_or_interrupted",
        "physical_success": success,
        "fresh_start_task_success": success and not conditional and case["category"] != "stop",
        "justified_stop_success": success and case["category"] == "stop",
        "conditional_recovery_success": success and conditional,
        "teacher_assisted": conditional,
        "teacher_prefix_ticks": result.get("teacher_prefix_ticks"),
        "initialization_matches_protocol": valid_start,
        "separate_categorical_decisions": separate_decisions,
        "historical_reference": historical_reference,
        "decision_interface_valid": valid_interface,
        "terminal": terminal,
        "false_completion": terminal == "complete" and not physical,
        "false_abstention": terminal == "abstain" and case["category"] != "stop",
        "autonomous_tick_budget": case["autonomous_tick_budget"],
        "executed_ticks": result.get("executed_ticks"),
        "ordered_levels": evaluation.get("max_ordered_supported_levels"),
        "ordered_levels_at_handoff": result.get("ordered_levels_at_handoff", 0),
        "cooperation_events": evaluation.get("cooperation_events", []),
        "physical_checks": evaluation.get("checks", {}),
        "physical_progress": progress,
        "complete_physical_trace": full_trace,
        "wall_seconds": probe.get("wall_seconds"),
        "error": result.get("error", probe.get("error")),
        "report": str(report_path),
        "qualification": False,
    }


def panel_summary(rows: list[dict]) -> dict:
    summary = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        summary[category] = {
            "cases": len(selected),
            "physical_successes": sum(row["physical_success"] for row in selected),
            "fresh_start_successes": sum(row["fresh_start_task_success"] for row in selected),
            "conditional_recovery_successes": sum(
                row["conditional_recovery_success"] for row in selected
            ),
            "unavailable_or_interrupted": sum(
                row["availability"] != "executed" for row in selected
            ),
            "false_completions": sum(row["false_completion"] for row in selected),
            "false_abstentions": sum(row["false_abstention"] for row in selected),
        }
    return summary


def control_gate(rows: list[dict], checkpoint: Path) -> dict:
    if (
        len(rows) != 10
        or len({row["case"] for row in rows}) != 10
        or any(
            row["category"] != "single_block"
            or row["teacher_assisted"]
            or row.get("historical_reference", False)
            for row in rows
        )
        or Counter(row["start"] for row in rows) != {"familiar": 5, "held_out": 5}
        or {row["active_arm"] for row in rows} != {"left", "right"}
        or checkpoint.resolve().name != "020000"
        or json.loads((checkpoint / "state.json").read_text())["step"] != 20_000
    ):
        raise ValueError("The20k control requires the complete unassisted ten-case panel")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "completed_control_updates": 20_000,
        "teacher_assisted": False,
        "passed": sum(row["physical_success"] for row in rows) >= 8,
        "cases": rows,
    }
