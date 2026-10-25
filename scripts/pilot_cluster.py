"""Compact health checks and sequential Slurm allocations for one pilot campaign."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

REMOTE_STATUS = """import getpass,json,sys,time,subprocess
from pathlib import Path
root=Path(sys.argv[1]);job=sys.argv[2];name=sys.argv[3]
path=root/'run/status.json'
state=json.loads(path.read_text()) if path.exists() else {}
active=subprocess.run(['squeue','-h','-u',getpass.getuser(),'-n',name,'-o','%i %T'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True).stdout.strip()
latest=root/'run/checkpoints/latest'
out={k:state.get(k) for k in ('status','step','device','best_step','stopping_reason','error','allocation_until','last_progress')}
out.update(active_jobs=active,checkpoint=str(latest.resolve()) if latest.exists() else None,source_age_s=round(time.time()-state.get('epoch_s',time.time()),1),data_ready=(root/'corpus-ready.json').exists())
if job:
 out['accounting']=subprocess.run(['sacct','-n','-X','-j',job,'--format=JobID,State,Elapsed,AllocTRES','-P'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True).stdout.strip()
print(json.dumps(out))
"""


def remote(state: dict, command: str, *, data: str | None = None) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", state["host"], command],
        input=data,
        text=True,
        capture_output=True,
        timeout=45,
        check=True,
    )
    return result.stdout.strip()


def summary(state: dict) -> dict:
    root = state["remote_root"]
    command = f"{shlex.quote(root + '/lerobot/.venv/bin/python')} - {shlex.quote(root)} "
    command += shlex.quote(str(state.get("job_id", "")))
    command += " " + shlex.quote(state.get("job_name", "vla-pilot-20261024"))
    return dict(json.loads(remote(state, command, data=REMOTE_STATUS)))


def batch_script(state: dict, endpoint: int, *, resume: bool) -> str:
    root = state["remote_root"]
    quote = shlex.quote
    command = [
        root + "/lerobot/.venv/bin/python",
        "-m",
        "vla_lego.pretrained.train_pilot",
        "--corpus",
        root + "/corpus",
        "--output",
        root + "/run",
        "--statistics",
        root + "/statistics.json",
        "--until-step",
        str(endpoint),
        "--microbatch",
        "64",
        "--maximum-step",
        str(state.get("maximum_step", 100_000)),
        "--seed",
        str(state["seed"]),
        "--action-representation",
        state.get("action_representation", "absolute"),
    ]
    if resume:
        command += ["--resume", root + "/run/checkpoints/latest"]
    return f"""#!/bin/bash
#SBATCH --job-name={state.get('job_name', 'vla-pilot-20261024')}
#SBATCH --account={state['account']}
#SBATCH --partition={state['partition']}
#SBATCH --qos={state['qos']}
#SBATCH --gres={state['gres']}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@300
#SBATCH --output={root}/logs/main-%j.log
set -euo pipefail
cd {quote(root)}
export PYTHONPATH={quote(root + '/source')}
export HF_HOME={quote(root + '/hf-cache')}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
srun {shlex.join(command)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "advance"))
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    status = summary(state)
    if args.action == "advance":
        if status["active_jobs"]:
            status["action"] = "already_active"
        elif status["status"] == "training_complete":
            status["action"] = "ready_for_final_evaluation"
        elif status["status"] == "failed":
            status["action"] = "diagnosis_required"
        elif not status["data_ready"]:
            status["action"] = "awaiting_corpus_transfer"
        elif (status.get("step") or 0) >= 100_000:
            status["action"] = "maximum_step_reached_requires_finalization"
        elif status["status"] not in (None, "allocation_complete", "allocation_interrupted"):
            status["action"] = "inspect_terminated_allocation_before_resume"
        else:
            step = int(status.get("step") or 0)
            reviews = [n for n in state.get("review_steps", [30_000]) if n > step]
            endpoint = (
                min(reviews)
                if reviews
                else min(state.get("maximum_step", 100_000), (step // 10_000 + 1) * 10_000)
            )
            script = batch_script(state, endpoint, resume=bool(status["checkpoint"]))
            job = remote(state, "sbatch --parsable", data=script).split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"Unconfirmed scheduler submission: {job}")
            submitted_at = time.time()
            state.update(job_id=job, submitted_at=submitted_at, allocation_until=endpoint)
            state.setdefault("allocations", []).append(
                {"job_id": job, "until_step": endpoint, "submitted_at": submitted_at}
            )
            args.state.write_text(json.dumps(state, indent=2) + "\n")
            allocation = args.state.parent / f"main-{job}.sbatch"
            allocation.write_text(script)
            status.update(action="submitted", job_id=job, allocation_until=endpoint)
    print(json.dumps(status, separators=(",", ":")))


if __name__ == "__main__":
    main()
