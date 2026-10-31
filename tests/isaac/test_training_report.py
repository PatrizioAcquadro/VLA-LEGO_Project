"""Offline reporting supports the actual motor-only and historical metric formats."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

REPORT = Path(__file__).resolve().parents[2] / "scripts" / "pilot_training_report.py"


@pytest.mark.parametrize("historical", [False, True])
def test_report_preserves_recorded_metrics_and_leaves_missing_values_blank(tmp_path, historical):
    run = tmp_path / "run"
    run.mkdir()
    metrics = {
        "motor_flow": 0.3,
        "motor_mae_normalized": 0.1,
        "arm_mae_rad": 0.01,
        "jaw_mae_m": 0.002,
    }
    if historical:
        metrics.update(decision_flow=0.02, decision_correct=0.7)
    else:
        metrics.update(executed_motor_mae_normalized=0.04, hold_executed_motor_mae_normalized=0.08)
    history = [
        {"step": step, "train": dict(metrics), "development": dict(metrics)} for step in (0, 5000)
    ]
    del history[1]["development"]["jaw_mae_m"]
    (run / "history.json").write_text(json.dumps(history))
    (run / "status.json").write_text(
        json.dumps({"status": "allocation_complete", "step": 5000, "best_step": 0})
    )
    (run / "progress.jsonl").write_text(json.dumps({"step": 10, "loss": 0.3}) + "\n")
    output = tmp_path / "report"
    subprocess.run(
        [sys.executable, str(REPORT), "--run", str(run), "--output", str(output)], check=True
    )
    with (output / "offline-curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["development_jaw_mae_m"] == "0.002"
    assert rows[1]["development_jaw_mae_m"] == ""
    assert ("train_decision_correct" in rows[0]) == historical
    if not historical:
        assert rows[0]["train_executed_motor_mae_normalized"] == "0.04"
        assert rows[0]["train_hold_executed_motor_mae_normalized"] == "0.08"
    with Image.open(output / "learning-curves.png") as image:
        assert image.width > 1000 and image.height > 500
    assert "physically qualified" in (output / "training-report.md").read_text()


def test_empty_history_is_rejected_without_a_fabricated_report(tmp_path):
    (tmp_path / "history.json").write_text("[]")
    (tmp_path / "status.json").write_text("{}")
    output = tmp_path / "report"
    result = subprocess.run(
        [sys.executable, str(REPORT), "--run", str(tmp_path), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "no diagnostics" in result.stderr
    assert not output.exists()
