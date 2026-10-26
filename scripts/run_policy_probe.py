"""Launch one native policy/simulator diagnostic together, without automatic retries."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def stop_owned(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lerobot-python", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=1200)
    parser.add_argument("--execution-horizon", type=int)
    parser.add_argument("--jaw-overlap", action="store_true")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--task", choices=("single_block", "tower"), default="single_block")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--missing-target", type=int, choices=range(5))
    parser.add_argument("--pilot-corpus", type=Path)
    parser.add_argument("--pilot-case")
    parser.add_argument("--pilot-split", choices=("train", "development"), default="development")
    parser.add_argument("--teacher-start-stage")
    parser.add_argument("--campaign-cases", type=Path)
    parser.add_argument("--campaign-case")
    parser.add_argument("--campaign-partition", choices=("train", "calibration", "confirmation"))
    parser.add_argument("--campaign-corpus", type=Path)
    parser.add_argument("--decision-checkpoint", type=Path)
    parser.add_argument("--instruction-capacity", type=int, choices=range(1, 97))
    parser.add_argument("--collect-campaign-correction", action="store_true")
    parser.add_argument("--wall-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.task != "tower" and (args.mirror or args.missing_target is not None):
        parser.error("Scene variants require --task tower")
    if bool(args.pilot_corpus) != bool(args.pilot_case) or (
        args.pilot_case and args.task != "tower"
    ):
        parser.error("Recorded pilot evaluation requires --task tower, corpus and case")
    if bool(args.campaign_cases) != bool(args.campaign_case) or bool(args.campaign_case) != bool(
        args.campaign_partition
    ):
        parser.error("Campaign declaration, case and partition are required together")
    if args.campaign_case and (
        args.task != "tower" or args.pilot_case or args.mirror or args.missing_target is not None
    ):
        parser.error("Campaign evaluation uses the declared task with the tower scenario runner")
    if args.campaign_corpus and not args.campaign_case:
        parser.error("Campaign corpus requires a declared case")
    if args.decision_checkpoint and not args.campaign_case:
        parser.error("The separate learned head requires the campaign scenario runner")
    if args.collect_campaign_correction and (
        not args.campaign_case
        or args.campaign_partition != "train"
        or args.decision_checkpoint
        or args.teacher_start_stage
        or not args.jaw_overlap
        or args.execution_horizon != 10
    ):
        parser.error(
            "Correction collection requires the fixed TRAIN component and its ten-step jaw kernel"
        )
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    worker = client = None
    weights = args.checkpoint.resolve()
    if (weights / "pretrained_model").is_dir():
        weights /= "pretrained_model"
    started = time.monotonic()
    report: dict = {
        "kind": "single_policy_probe",
        "automatic_retries": 0,
        "passed": False,
        "availability": "setup_unavailable",
        "autonomy_started": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "weights": str(weights),
        "decision_checkpoint": (
            None if args.decision_checkpoint is None else str(args.decision_checkpoint.resolve())
        ),
        "instruction_capacity": args.instruction_capacity,
        "campaign_case": args.campaign_case,
        "campaign_partition": args.campaign_partition,
        "collection_only": args.collect_campaign_correction,
        "ticks": args.ticks,
        "model_seed": args.seed,
        "task": args.task,
        "jaw_overlap": args.jaw_overlap,
    }
    code = 1
    try:
        for name in ("ISAAC_SIM_DIR", "ISAAC_LAB_DIR", "ALEX_ASSETS_ROOT"):
            if not os.environ.get(name):
                raise ValueError(f"Required native environment variable: {name}")
        worker_env = dict(os.environ, PYTHONPATH=str(root / "src"))
        with tempfile.TemporaryDirectory(prefix="vla-lego-policy-") as temporary:
            socket_path = Path(temporary) / "policy.sock"
            worker_command = [
                str(args.lerobot_python.absolute()),
                "-m",
                "vla_lego.pretrained.worker",
                "--checkpoint",
                str(weights),
                "--socket",
                str(socket_path),
                "--output-dir",
                str(output / "worker"),
                "--seed",
                str(args.seed),
            ]
            if args.execution_horizon is not None:
                worker_command += ["--execution-horizon", str(args.execution_horizon)]
            if args.jaw_overlap:
                worker_command += ["--jaw-overlap"]
            if args.instruction_capacity is not None:
                worker_command += ["--instruction-capacity", str(args.instruction_capacity)]
            if args.decision_checkpoint:
                worker_command += ["--decision-checkpoint", str(args.decision_checkpoint.resolve())]
            if args.pilot_case or args.campaign_case:
                worker_command += ["--record-video"]
            with (
                (output / "worker.log").open("w") as worker_log,
                (output / "rollout.log").open("w") as client_log,
            ):
                worker = subprocess.Popen(
                    worker_command,
                    cwd=root,
                    env=worker_env,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 180
                while (
                    not socket_path.exists()
                    and worker.poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
                if not socket_path.exists():
                    report["availability"] = "setup_unavailable"
                    raise RuntimeError("Policy did not become ready; inspect worker.log.")
                print(json.dumps({"policy_ready": True, "output": str(output)}), flush=True)
                if args.collect_campaign_correction:
                    from vla_lego.isaac.campaign import load_campaign

                    case = next(
                        case
                        for case in load_campaign(args.campaign_cases)
                        if case.name == args.campaign_case
                    )
                    client_command = [
                        "bash",
                        str(root / "scripts/isaac-run.sh"),
                        "-m",
                        "vla_lego.isaac.tower_reference",
                        "--output-dir",
                        str(output / "rollout"),
                        "--cache-dir",
                        str(args.cache_dir.resolve()),
                        "--campaign-cases",
                        str(args.campaign_cases.resolve()),
                        "--case",
                        case.name,
                        "--family",
                        case.row["family"],
                        "--split",
                        "train",
                        "--record",
                        "--rgb-storage",
                        "ffv1",
                        "--event-settling",
                        "--policy-socket",
                        str(socket_path),
                        "--policy-reference",
                        str(weights),
                        "--policy-window-ticks",
                        str(args.ticks),
                    ]
                else:
                    client_command = (
                        [
                            "bash",
                            str(root / "scripts/isaac-run.sh"),
                            "-m",
                            (
                                "vla_lego.isaac.tower_rollout"
                                if args.task == "tower"
                                else "vla_lego.isaac.rollout"
                            ),
                            "--socket",
                            str(socket_path),
                            "--output-dir",
                            str(output / "rollout"),
                            "--cache-dir",
                            str(args.cache_dir.resolve()),
                            "--ticks",
                            str(args.ticks),
                        ]
                        + (["--mirror"] if args.mirror else [])
                        + (
                            ["--teacher-start-stage", args.teacher_start_stage]
                            if args.teacher_start_stage
                            else []
                        )
                        + (
                            ["--missing-target", str(args.missing_target)]
                            if args.missing_target is not None
                            else []
                        )
                        + (
                            [
                                "--pilot-corpus",
                                str(args.pilot_corpus.resolve()),
                                "--pilot-case",
                                args.pilot_case,
                                "--pilot-split",
                                args.pilot_split,
                            ]
                            if args.pilot_case
                            else []
                        )
                        + (
                            [
                                "--campaign-cases",
                                str(args.campaign_cases.resolve()),
                                "--campaign-case",
                                args.campaign_case,
                                "--campaign-partition",
                                args.campaign_partition,
                            ]
                            if args.campaign_case
                            else []
                        )
                        + (
                            ["--campaign-corpus", str(args.campaign_corpus.resolve())]
                            if args.campaign_corpus
                            else []
                        )
                    )
                client = subprocess.Popen(
                    client_command,
                    cwd=root,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = client.wait(timeout=args.wall_seconds)
                result_path = output / "rollout" / "report.json"
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    report.update(
                        passed=result["passed"],
                        autonomy_started=(
                            False
                            if args.collect_campaign_correction
                            else result.get("autonomy_started")
                        ),
                        availability=(
                            "started"
                            if result.get("autonomy_started") or result.get("policy_window_started")
                            else "setup_unavailable"
                        ),
                        rollout_report=str(result_path),
                        case_passed=result.get("case_passed", result["passed"]),
                    )
                else:
                    report["availability"] = "unscorable"
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stop_owned(worker)
    except KeyboardInterrupt:
        report["interrupted"] = True
        code = 130
    except Exception as error:
        report["error"] = str(error)
    finally:
        stop_owned(client)
        stop_owned(worker)
        report.update(
            wall_seconds=time.monotonic() - started,
            client_exit_code=code,
            worker_exit_code=worker.returncode if worker else None,
        )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
