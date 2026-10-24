"""Run the frozen before/after development panel once per model/case."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-status", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--lerobot-python", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.training_status.read_text())
    if state["status"] != "training_complete":
        raise ValueError("Full native panels are allowed only after training has concluded")
    selected = state["best_step"]
    for step in (0, selected):
        checkpoint = args.weights_root / f"{step:06d}"
        for name in ("config.json", "model.safetensors", "alex_interface.json"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
    cases = json.loads((args.corpus / "development-episodes.json").read_text())
    if len(cases) != 10:
        raise ValueError("The frozen panel requires all ten development cases")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".panel.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = args.gpu_lock.open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = Path(__file__).resolve().parent / "run_policy_probe.py"
    outputs = []
    for label, step in (("before", 0), ("after", selected)):
        checkpoint = args.weights_root / f"{step:06d}"
        for ordinal, entry in enumerate(cases):
            metadata = json.loads(
                (
                    args.corpus
                    / "attempts"
                    / Path(entry["path"]).parent.name
                    / "recording/episode.json"
                ).read_text()
            )
            category = metadata["collection_case"]["category"]
            directory = args.output_dir / label / entry["case"]
            directory.parent.mkdir(exist_ok=True)
            if not directory.exists():
                command = [
                    sys.executable,
                    str(source),
                    "--task",
                    "tower",
                    "--checkpoint",
                    str(checkpoint.resolve()),
                    "--lerobot-python",
                    str(args.lerobot_python.absolute()),
                    "--cache-dir",
                    str(args.cache_dir.resolve()),
                    "--output-dir",
                    str(directory.resolve()),
                    "--pilot-corpus",
                    str(args.corpus.resolve()),
                    "--pilot-case",
                    entry["case"],
                    "--ticks",
                    "12000",
                    "--wall-seconds",
                    "3600",
                    "--execution-horizon",
                    "10",
                    "--seed",
                    str(20261024 + ordinal),
                ]
                # Nonzero task outcomes are evidence, not an invitation to retry.
                subprocess.run(command, check=False)
            report = directory / "rollout/report.json"
            if report.exists():
                result = json.loads(report.read_text())
                outputs.append(
                    {
                        "model": label,
                        "step": step,
                        "case": entry["case"],
                        "category": category,
                        "availability": (
                            "executed"
                            if result.get("autonomy_started", False)
                            else "setup_unavailable"
                        ),
                        "case_passed": result.get("case_passed", False),
                        "autonomy_started": result.get("autonomy_started", False),
                        "ordered_levels": result.get("evaluation", {}).get(
                            "max_ordered_supported_levels"
                        ),
                        "terminal": result.get("terminal_decision"),
                        "physical_tower": result.get("tower_constructed", False),
                        "teacher_prefix_ticks": result.get("teacher_prefix_ticks", 0),
                        "ordered_levels_at_handoff": result.get("ordered_levels_at_handoff", 0),
                        "error": result.get("error"),
                    }
                )
            else:
                outputs.append(
                    {
                        "model": label,
                        "step": step,
                        "case": entry["case"],
                        "category": category,
                        "case_passed": False,
                        "availability": "unavailable_or_interrupted",
                    }
                )
            (args.output_dir / "panel-results.json").write_text(
                json.dumps(outputs, indent=2) + "\n"
            )
    summary = {}
    for label in ("before", "after"):
        summary[label] = {}
        for category in ("nominal", "recovery", "stop"):
            rows = [r for r in outputs if r["model"] == label and r["category"] == category]
            summary[label][category] = {
                "cases": len(rows),
                "successes": sum(r["case_passed"] for r in rows),
                "unavailable": sum(r.get("availability") != "executed" for r in rows),
                "false_abstentions": (
                    sum(r.get("terminal") == "abstain" for r in rows) if category != "stop" else 0
                ),
                "conflicting_decisions": sum(r.get("terminal") == "conflict" for r in rows),
            }
    (args.output_dir / "panel-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model_cases": len(outputs),
                "results": str(args.output_dir / "panel-results.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
