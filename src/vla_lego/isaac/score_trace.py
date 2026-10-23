"""Score a completed native full-task trace without altering its original outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lego.isaac.catalog import BlockPlacement, BlockSpec
from vla_lego.isaac.evaluation import TowerEvaluator
from vla_lego.isaac.task import TowerTask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.run_dir / "report.json").read_text())
    raw = source["task"]
    task = TowerTask(
        tuple(
            BlockPlacement(BlockSpec(**p["spec"]), (p["xy_m"][0], p["xy_m"][1]))
            for p in raw["placements"]
        ),
        tuple(BlockSpec(**spec) for spec in raw["targets"]),
        (raw["assembly_xy_m"][0], raw["assembly_xy_m"][1]),
    )
    evaluator = None
    contact_paths = None
    with (args.run_dir / "trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            paths = row["robot"]["robot_contact_paths"]
            if evaluator is None:
                contact_paths = paths
                evaluator = TowerEvaluator(task, paths)
            if paths != contact_paths:
                raise ValueError("Contact-body ordering changed inside the source episode.")
            evaluator.update(row["time_s"], row["blocks"])
    if evaluator is None:
        raise ValueError("Cannot score an empty trace.")
    report = evaluator.result()
    report.update(
        kind="full_task_reference_score" if not source["learned"] else "full_task_learned_score",
        source_run=str(args.run_dir.resolve()),
        source_outcome_preserved=True,
        source_report_passed=source["passed"],
        qualification=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("kind", "passed", "checks")}), flush=True)


if __name__ == "__main__":
    main()
