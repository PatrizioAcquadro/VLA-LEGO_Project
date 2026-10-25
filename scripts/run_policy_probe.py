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
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--task", choices=("single_block", "tower"), default="single_block")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--missing-target", type=int, choices=range(5))
    parser.add_argument("--pilot-corpus", type=Path)
    parser.add_argument("--pilot-case")
    parser.add_argument("--pilot-split", choices=("train", "development"), default="development")
    parser.add_argument("--wall-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.task != "tower" and (args.mirror or args.missing_target is not None):
        parser.error("Scene variants require --task tower")
    if bool(args.pilot_corpus) != bool(args.pilot_case) or (
        args.pilot_case and args.task != "tower"
    ):
        parser.error("Recorded pilot evaluation requires --task tower, corpus and case")
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    worker = client = None
    started = time.monotonic()
    report: dict = {
        "kind": "single_policy_probe",
        "automatic_retries": 0,
        "passed": False,
        "availability": "setup_unavailable",
        "autonomy_started": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "ticks": args.ticks,
        "model_seed": args.seed,
        "task": args.task,
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
                str(args.checkpoint.resolve()),
                "--socket",
                str(socket_path),
                "--output-dir",
                str(output / "worker"),
                "--seed",
                str(args.seed),
            ]
            if args.execution_horizon is not None:
                worker_command += ["--execution-horizon", str(args.execution_horizon)]
            if args.pilot_case:
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
                client = subprocess.Popen(
                    [
                        "bash",
                        str(root / "scripts" / "isaac-run.sh"),
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
                    ),
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
                        autonomy_started=result.get("autonomy_started"),
                        availability=(
                            "started" if result.get("autonomy_started") else "setup_unavailable"
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
