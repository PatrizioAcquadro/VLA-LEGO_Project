"""Preserve physical evidence and budgets across cached campaign comparisons."""

import json
import sys
from dataclasses import asdict

import pytest

from vla_lego.pretrained.campaign_evaluation import (
    control_gate,
    outcome,
    panel_summary,
    retain_binding,
    run_case_once,
    total_probe_ticks,
)


def case(category="single_block"):
    return {
        "case": "example",
        "category": category,
        "partition": "calibration",
        "start": "held_out",
        "active_arm": "left",
        "autonomous_tick_budget": 3000,
        "teacher_recovery_prefix_ticks": 1200 if category == "recovery" else 0,
        "conditional_recovery": category == "recovery",
    }


def report(directory, *, conditional=False, passed=True):
    from vla_lego.isaac.catalog import BlockSpec
    from vla_lego.isaac.environment import TABLE_TOP

    path = directory / "rollout"
    path.mkdir(parents=True)
    spec = BlockSpec("cube", "small", "red")
    robot = {
        "robot_contact_paths": [
            *(
                f"/{side}_WSG32_{jaw}_JAW_LINK"
                for side in ("left", "right")
                for jaw in ("NEGATIVE", "POSITIVE")
            ),
            "/HEAD_LINK",
        ],
        "robot_contact_forces_n": [[0.0, 0.0, 0.0]] * 5,
    }
    rows = []
    for tick in range(43):
        held = tick == 1 or (conditional and tick == 0)
        block = {
            "block_pose_w": [
                0.7 if tick < 2 else 0.4,
                0.0,
                TABLE_TOP + spec.dimensions_m[2] / 2 + (0.08 if held else 0.0),
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "robot_contact_forces_n": [
                [0.0, 0.0, 0.6 if held and j < 2 else 0.0] for j in range(5)
            ],
            "robot_contact_force_n": 1.2 if held else 0.0,
            "nonrobot_contact_force_n": 0.0 if held else 1.0,
        }
        rows.append(
            {
                "time_s": tick / 20,
                "blocks": [block],
                "robot": robot,
                **(
                    {"tick": tick, "learned_action": [0.0] * 16}
                    if tick < 3
                    else {"phase": "terminal_hold"}
                ),
            }
        )
    (path / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    value = {
        "task": {
            "targets": [asdict(spec)],
            "placements": [{"spec": asdict(spec)}],
            "build_region_xy_m": [0.3, 0.6, -0.1, 0.1],
        },
        "autonomy_started": True,
        "autonomous_from_start": not conditional,
        "teacher_commands": conditional,
        "teacher_prefix_ticks": 1200 if conditional else 0,
        "recovery_fault_present": conditional,
        "separate_categorical_decisions": True,
        "case_passed": passed,
        "evaluation": {"passed": passed},
        "terminal_decision": "complete",
        "executed_ticks": 3,
    }
    (path / "report.json").write_text(json.dumps(value))
    return value


def test_progress_excludes_existing_teacher_acquisition_and_requires_full_trace(tmp_path):
    from vla_lego.pretrained.campaign_evaluation import trace_progress

    value = report(tmp_path, conditional=True)
    progress = trace_progress(tmp_path, value["task"])
    assert progress["correct_requested_piece_acquisitions"] == 0
    assert progress["unique_new_released_ordered_targets"] == 1
    trace = tmp_path / "rollout/trace.jsonl"
    full_trace = trace.read_text()
    trace.write_text("\n".join(trace.read_text().splitlines()[:-1]) + "\n")
    assert not outcome(case("recovery"), tmp_path)["physical_success"]
    trace.write_text(trace.read_text() + '{"partial":')
    row = outcome(case("recovery"), tmp_path)
    assert not row["physical_success"] and row["physical_progress"]["error"]
    rows = [json.loads(line) for line in full_trace.splitlines()]
    rows[1]["learned_action"] = [0.0] * 18
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert not outcome(case("recovery"), tmp_path)["physical_success"]


def test_recovery_and_terminal_hold_do_not_consume_autonomous_budget():
    assert total_probe_ticks(case()) == 3040
    assert total_probe_ticks(case("recovery")) == 4240


def test_failed_or_unavailable_original_is_never_rerun(tmp_path):
    directory = tmp_path / "failure"
    command = [sys.executable, "-c", "raise SystemExit(1)"]
    run_case_once(directory, command)
    original = (directory / "report.json").read_bytes()
    run_case_once(directory, ["this-command-must-never-run"])
    assert (directory / "report.json").read_bytes() == original
    missing = outcome(case(), directory)
    assert not missing["physical_success"]
    assert panel_summary([missing])["single_block"]["cases"] == 1
    assert panel_summary([missing])["single_block"]["unavailable_or_interrupted"] == 1


def test_changed_head_motor_or_runtime_cannot_reuse_reference(tmp_path):
    original = {"motor": "a", "decision": "b", "runtime": "c"}
    retain_binding(tmp_path, original)
    retain_binding(tmp_path, original)
    for key in original:
        with pytest.raises(ValueError, match="identity changed"):
            retain_binding(tmp_path, {**original, key: "changed"})
    assert json.loads((tmp_path / "execution-binding.json").read_text()) == original


def test_teacher_recovery_is_conditional_and_terminal_alone_is_not_success(tmp_path):
    value = report(tmp_path, conditional=True)
    row = outcome(case("recovery"), tmp_path)
    assert row["physical_success"] and row["conditional_recovery_success"]
    assert not row["fresh_start_task_success"]
    assert not outcome(case(), tmp_path)["physical_success"]
    value["evaluation"]["passed"] = False
    (tmp_path / "rollout/report.json").write_text(json.dumps(value))
    row = outcome(case("recovery"), tmp_path)
    assert not row["physical_success"] and row["false_completion"]
    value["evaluation"]["passed"] = True
    value["separate_categorical_decisions"] = False
    (tmp_path / "rollout/report.json").write_text(json.dumps(value))
    assert not outcome(case("recovery"), tmp_path)["physical_success"]


def test_control_gate_keeps_all_ten_denominators_and_actual_checkpoint_step(tmp_path):
    checkpoint = tmp_path / "020000"
    checkpoint.mkdir()
    (checkpoint / "state.json").write_text(json.dumps({"step": 20000}))
    report(tmp_path)
    original = outcome(case(), tmp_path)
    rows = [
        {
            **original,
            "case": str(i),
            "active_arm": "left" if i % 2 else "right",
            "start": "familiar" if i < 5 else "held_out",
            "physical_success": i < 8,
        }
        for i in range(10)
    ]
    assert control_gate(rows, checkpoint)["passed"]
    rows[-1]["availability"] = "unavailable_or_interrupted"
    rows[0]["physical_success"] = False
    assert not control_gate(rows, checkpoint)["passed"]
    assert len(control_gate(rows, checkpoint)["cases"]) == 10
    (checkpoint / "state.json").write_text(json.dumps({"step": 10}))
    with pytest.raises(ValueError, match="20k"):
        control_gate(rows, checkpoint)
