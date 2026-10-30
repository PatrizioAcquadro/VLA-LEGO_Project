"""Execute a frozen campaign panel once per motor/head/runtime combination."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from vla_lego.pretrained.campaign_data import admitted_entries
from vla_lego.pretrained.campaign_evaluation import (
    PANEL_COUNTS,
    control_gate,
    execution_binding,
    file_digest,
    outcome,
    panel_cases,
    panel_summary,
    retain_binding,
    run_case_once,
    total_probe_ticks,
)
from vla_lego.pretrained.pilot_campaign import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=tuple(PANEL_COUNTS), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--campaign-cases", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decision-checkpoint", type=Path, required=True)
    parser.add_argument("--lerobot-python", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--results-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--global-gpu-lock", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-reservation", type=Path, required=True)
    parser.add_argument("--owner-thread-id", required=True)
    parser.add_argument("--gpu-slot")
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    parser.add_argument("--write-control-gate", type=Path)
    args = parser.parse_args()
    if args.write_control_gate and args.panel != "single_block_control":
        parser.error("Only the single-block control can write its advancement gate")
    entries = admitted_entries(args.corpus)
    protocol = json.loads(args.protocol.read_text())
    cases = panel_cases(protocol, args.panel, json.loads(args.campaign_cases.read_text())["cases"])
    admitted = {entry["case"]: entry for entry in entries}
    for row in cases:
        entry = admitted[row["case"]]
        if any(entry[key] != row[key] for key in ("partition", "category", "recording")):
            raise ValueError("The physical panel differs from the admitted source")
    root = Path(__file__).resolve().parents[1]
    binding = execution_binding(
        root, args.checkpoint.resolve(), args.decision_checkpoint, args.protocol
    )
    native = {}
    for name in ("ISAAC_SIM_DIR", "ISAAC_LAB_DIR", "ALEX_ASSETS_ROOT"):
        native[name] = str(Path(os.environ[name]).resolve(strict=True))
    binding.update(native_environment=native, runtime_evidence=file_digest(args.runtime_evidence))
    identity = (
        file_digest(args.decision_checkpoint)[:12] + "-" + binding["motor/model.safetensors"][:12]
    )
    cache = args.results_cache.resolve() / identity
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (
        (args.output_dir / ".panel.lock").open("a") as panel_lock,
        args.global_gpu_lock.open("a") as global_lock,
        args.gpu_lock.open("a") as gpu_lock,
    ):
        fcntl.flock(panel_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(global_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reservation = json.loads(args.gpu_reservation.read_text())
        if (
            reservation["device_uuid"] != args.gpu_uuid
            or args.owner_thread_id not in reservation["allowed_owner_thread_ids"]
        ):
            raise ValueError("The physical GPU is not reserved for this owner")
        if "slots" in reservation:
            if (
                reservation["slots"].get(args.gpu_slot, {}).get("owner_thread_id")
                != args.owner_thread_id
            ):
                raise ValueError("The shared GPU slot belongs to another owner")
            expected_lock = args.global_gpu_lock.with_name(
                args.global_gpu_lock.stem + "-slot" + args.gpu_slot + ".lock"
            )
        else:
            expected_lock = Path(reservation["lock_path"])
        if args.gpu_lock.resolve() != expected_lock.resolve():
            raise ValueError("The exclusive lock differs from the owned GPU reservation")
        cache.mkdir(parents=True, exist_ok=True)
        # Different panel invocations may share the same frozen reference cache.
        with (cache / ".cache.lock").open("a") as cache_lock:
            fcntl.flock(cache_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            retain_binding(cache / "binding", binding)
            if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
                raise ValueError("Bind CUDA_VISIBLE_DEVICES to the owned physical GPU UUID")
            processes = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            if any(line.split(",")[0].strip() == args.gpu_uuid for line in processes.splitlines()):
                raise RuntimeError("The owned GPU has an existing workload; preserve it")
            rows = []
            try:
                for case in cases:
                    directory = cache / case["case"]
                    if not directory.exists():
                        if shutil.disk_usage(cache).free < 8 * 1024**3:
                            raise RuntimeError("Preserve the eight-GiB free storage floor")
                        command = [
                            sys.executable,
                            str(root / "scripts/run_policy_probe.py"),
                            "--task",
                            "tower",
                            "--checkpoint",
                            str(args.checkpoint.resolve()),
                            "--decision-checkpoint",
                            str(args.decision_checkpoint.resolve()),
                            "--lerobot-python",
                            str(args.lerobot_python.absolute()),
                            "--cache-dir",
                            str(args.cache_dir.resolve()),
                            "--output-dir",
                            str(directory),
                            "--campaign-cases",
                            str(args.campaign_cases.resolve()),
                            "--campaign-case",
                            case["case"],
                            "--campaign-partition",
                            case["partition"],
                            "--campaign-corpus",
                            str(args.corpus.resolve()),
                            "--ticks",
                            str(total_probe_ticks(case)),
                            "--wall-seconds",
                            str(case["native_wall_bound_s"]),
                            "--execution-horizon",
                            "10",
                            "--jaw-overlap",
                            "--instruction-capacity",
                            "96",
                            "--seed",
                            str(case["seed"]),
                        ]
                        # Never retry an existing directory, even after setup or process failure.
                        run_case_once(directory, command)
                    rows.append(outcome(case, directory))
                    write_json(args.output_dir / "panel-results.json", rows)
            finally:
                write_json(
                    args.output_dir / "panel-summary.json",
                    {
                        "panel": args.panel,
                        "planned_cases": len(cases),
                        "reviewed_outcomes": len(rows),
                        "not_yet_started": len(cases) - len(rows),
                        "results_cache": str(cache),
                        "automatic_retries": 0,
                        "categories": panel_summary(rows),
                        "qualification": False,
                    },
                )
            if args.write_control_gate:
                gate = control_gate(rows, args.checkpoint)
                if (
                    args.write_control_gate.exists()
                    and json.loads(args.write_control_gate.read_text()) != gate
                ):
                    raise ValueError("Preserve the existing control review")
                write_json(args.write_control_gate, gate)


if __name__ == "__main__":
    main()
