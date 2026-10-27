# Compute and containers

## Expanded campaign storage — October 26, 2026

A verified **209-episode storage snapshot** is on the campaign corpus in Gilbreth scratch: 14,393 files and 161,833,406,321 bytes, with file sizes and SHA-256 checked against Blackwell. Admissions are appended progressively with separate verification. Original recordings and failed attempts remain on Blackwell. The mechanism-coverage revision is canonical in the transferred corpus; original and coordination declarations remain preserved. Storage preparation does not establish the complete 620-episode corpus or training readiness. [Data](data.md#expanded-alex003-campaign--october-26-2026) owns admission and learning gates.

After the initial data gate, two native Blackwell collectors completed four additional single-block episodes that pass physical, reader and visual admission. Observed aggregate throughput was 6.70 recorded frames/s versus 4.67 frames/s in earlier single-collector runs, with about 13.2 GiB sampled peak GPU memory. This cross-case engineering measurement supports two bounded collectors; it is not a controlled hardware speed benchmark or policy qualification. Actual-policy correction collection uses the same bounded slots and leases, preserving separate setup, physical and behavioral outcomes.

## Shared workstation admission

The local RTX 4090 and Blackwell are shared with AlexDoor-XAS and SquadBot-AV. A persistent reservation and the same device lease govern admission for every queued successor and restart; free memory or an idle device does not cancel another project's window. Transfer ownership explicitly after the active episode and cleanup finish. Time limits cannot interrupt a healthy episode or silently reclaim its reservation.

Blackwell collection additionally respects its reserved slots under the shared device lease. Existing workloads drain before a slot changes owner, and a waiting successor cannot capture a slot reserved for another project. Keep unrelated workloads protected. Current windows and handoffs belong to the local execution records, not a fixed schedule in this wiki. These resource agreements change neither the scientific scope nor dataset, optimization or allocation budgets.

## Planned resource roles

| Resource | Intended role | Boundary |
|---|---|---|
| Local RTX 4090, 24 GB | Interactive development, visual simulation, small experiments and profiling | Larger VLA adaptation depends on measured fit; do not assume full fine-tuning fits. |
| Blackwell workstation, about 96 GB | Broader rendered collection, training and larger offline teachers | RTX PRO 6000 Blackwell Max-Q; current rendered two-collector measurement is reported above. Availability and new workloads still require measurement. |
| Gautschi | H100 training; RTX-capable L40-class nodes for rendering when available | Partition visibility is not a guaranteed allocation; select the actual available GPU and compatible runtime. |
| Gilbreth | A100/H100 training; compatible A10 nodes for rendering | Access, account limits, memory and allocation must match the job. |
| Robot Orin | All final VLA inference together with sensing, logging and robot communication | Qualify concurrent load on the real system; the NUC retains low-level control. |

The planning inspection found the local 4090 and approximately 96 GB Blackwell device, Gautschi H100/L40 partitions and Gilbreth A10/A100/H100 partitions. Hardware/access observations alone do not qualify a workload; the campaign measurement above covers current rendered collection. [Gautschi](https://docs.rcac.purdue.edu/userguides/gautschi/overview/) and [Gilbreth](https://docs.rcac.purdue.edu/userguides/gilbreth/overview/) documentation describe system resources; actual allocations govern use.

[Isaac Sim's RTX renderer requires supported RT hardware](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html). A100/H100 are training destinations, not supported visual-rendering destinations. Separate rendered collection from training rather than treating every CUDA GPU as interchangeable.

Maintain separate compatible environments for Isaac/Alex, model training and Orin deployment. The shared Alex package and selected VLA may require different Python/CUDA stacks. Choose collection and distributed-training scale from measured useful-data throughput, memory, storage and queue availability; no eight-GPU allocation or mandatory distributed framework is assumed.

The [model page](models.md#onboard-feasibility) owns Orin compatibility and memory constraints.

## Active pilot training

The October 24, 2026 SmolVLA pilot campaign used **one Gilbreth A100 80 GB**, account `euge`, partition `a100-80gb`, QoS `normal`, eight CPU cores and 64 GB host memory. The actual preflight job `12013348` verified batch-64 optimization, checkpoint reload and resume on an A100 80 GB PCIe device, using about 9.73 GiB peak allocated CUDA memory. Initial job **12013446** and four sequential continuations completed 70,000 updates in **8 h 11 min 20 s of allocated main-run GPU time**; the preflight used another 89 s. Main-run peak allocated CUDA memory was about 9.36 GiB. All five main allocations completed without errors; no training allocation remains active. Gautschi was not needed. Other projects' allocations were not interrupted.

Remote campaign root is `/depot/euge/data/pacquadr/vla-lego/pilot-training-20261024`: isolated LeRobot/Python 3.12.15, PyTorch 2.11.0+cu128, offline model cache, admitted corpus, train-only statistics and resumable run. The 60 selected episodes and 333,373 RGB images were verified present before launch. Isaac simulation/evaluation uses the separate Blackwell RTX environment; A100 is not the rendering destination. The [model page](models.md#active-marker-free-pilot-adaptation--october-24-2026) owns the frozen training recipe and stopping criteria.

Local `outputs/pilot-training-20261024/cluster.json` owns the job identity and allocation history. The compact status command is:

```bash
.venv/bin/python scripts/pilot_cluster.py status --state outputs/pilot-training-20261024/cluster.json
```

The `advance` action submits only when no campaign job is active, resumes the latest complete checkpoint, and requires inspection after an unexpected termination. It extends the same run through the declared endpoints rather than restarting learning. Training and the paired Blackwell panel are now complete; do not submit another allocation or repeat model-case directories. The twenty native-case wall times sum to 7.99 hours, excluding worker initialization and final application teardown. Policy workers peaked at 1.183 GiB allocated CUDA memory, excluding Isaac; no Orin measurement is implied. [Validation](validation.md#smolvla-pilot-before-and-after--october-25-2026) owns physical outcomes.

The thread heartbeat `continua-pilot-smolvla-alex003` checked every 15 minutes and ended healthy waiting turns promptly. It was disabled after delivery on October 25, 2026; its saved status is `PAUSED`. All owned training/evaluation work has ended. Scheduler accounting, panel logs, original outcomes and failed/incomplete component evidence remain in the ignored campaign directory. A new learning or collection iteration requires a new assignment.

## Existing MuJoCo and Qwen operations

All commands below run from a source checkout and operate the retained legacy stack. They do not install Isaac or qualify the new VLA. Image names and scheduler directives are templates, not evidence of current remote availability or allocation permission.

## Local Docker

```bash
docker build -f infra/containers/Dockerfile -t vla-lego:local .
VLA_DOCKER_IMAGE=vla-lego:local scripts/docker-run.sh python -m vla_lego.sim.demo --output-dir outputs/docker-demo-001
```

The image contains dependencies, not project source. The wrapper mounts the current checkout at `/workspace` and requests the GPU. Use module commands because the source is mounted rather than installed: `python -m vla_lego.sim.demo`, `python -m vla_lego.models.validate`, `python scripts/validate_configs.py`.

Docker and Apptainer definitions share dependency extraction from `pyproject.toml`. The base is PyTorch 2.10/CUDA 12.8; dependencies are installed in a venv that reuses base PyTorch. The unused inherited `spin` build tool is removed because its Click constraint conflicts with the current Hugging Face stack. Builds run `pip check`. No source checkout, credentials or model weights are baked into the image; the Docker build context is allowlisted by `infra/containers/Dockerfile.dockerignore`, using Docker’s [Dockerfile-specific ignore convention](https://docs.docker.com/build/concepts/context/#filename-and-location). Always build from the repository root with the explicit `-f` path above.

The entrypoint requires the mount, exposes only `/workspace/src` on `PYTHONPATH`, prints runtime/revision information and executes the requested command. Host GPU support requires a functioning NVIDIA container runtime. The default registry tag may be older than this branch; build locally to test this source/dependency pair. W&B variables passed through the wrapper affect only explicitly requested tracker initialization.

## Apptainer

```bash
apptainer build vla-lego.sif infra/containers/apptainer.def
VLA_APPTAINER_SIF=vla-lego.sif scripts/apptainer-run.sh python -m vla_lego.sim.demo --output-dir outputs/apptainer-demo-001
```

Build on a machine where Apptainer and the required build permissions are available. The wrapper uses `--nv`, mounts the checkout and optionally binds `VLA_SCRATCH_ROOT`. The CI conversion path builds a SIF from the published Docker digest; edits only to `infra/containers/apptainer.def` do not trigger that Docker path. Neither local SIF building nor cluster execution was validated in this audit.

## Gilbreth Python environment

On the cluster, first inspect available modules, partitions, quotas and disk space with `bash infra/gilbreth/scripts/inspect_cluster.sh`. Select a compatible Python/CUDA stack from the actual environment; old module names and hardcoded NCCL interface settings have been retired.

```bash
export VLA_SCRATCH_ROOT=/scratch/gilbreth/$USER/vla-lego
export VLA_ENV_DIR=$VLA_SCRATCH_ROOT/venv
export HF_HOME=$VLA_SCRATCH_ROOT/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
bash infra/gilbreth/scripts/setup_env.sh
mkdir -p logs
```

`VLA_PYTHON` can select an interpreter for environment creation. The setup installs the maintained simulation/model/tracking extras; it does not install a trainer or DeepSpeed. Set `VLA_SCRATCH_ROOT` explicitly on the cluster. Without it, config inspection falls back to the checkout rather than assuming a username or creating a scratch path. Cluster configs describe location metadata only. They do not configure devices, request SLURM resources or launch distributed jobs.

Inspect the `#SBATCH` account/partition/QoS/resource lines before submitting these from the checkout:

| Template | Workload |
|---|---|
| `download_model.sh` | CPU download only; never loads or runs model weights |
| `sim_smoke.sh` | One GPU, EGL diagnostic demo |
| `validate_model.sh` | One GPU, native Qwen/VLA smoke |

The simulation/model jobs write unique `outputs/*-${SLURM_JOB_ID}` directories and scheduler logs. Ensure these directories live on appropriate scratch storage for your checkout. Keep failed runs and scheduler exit status. No distributed training template is maintained. Future training belongs to [Phase 5](implementation_phases/phase-5-training.md), using the selected pretrained VLA; old toy-job results cannot qualify it.

## Shared multi-GPU practice

October 26, 2026. Guidance shared with SquadBot and the other robot-learning
projects; this does not launch work or change this project's approved recipe.

Use the existing PyTorch **DDP/torchrun/NCCL** stack and isolated application
environment. Before a long run, measure the actual stage on one/two GPUs and
consider four when batch, memory and timings justify it. Preserve effective global
batch, episode/chunk semantics, masks, optimizer/sampler/RNG state and update
budget; do not copy another project's batch or learning rates. Prefer a suitable
single-node allocation and compare Gautschi H100 versus Gilbreth A100 using queue,
staging and end-to-end throughput. Parallel independent approved runs can be
better than extra ranks on a small-batch job. Report wall time and GPU-hours.

For a separately authorized performance task, reuse PyTorch Profiler and existing
logs; install TensorBoard only if a viewer is needed, in the project environment.
Use Nsight Systems only for an identified unresolved bottleneck. Preserve the
shared NVIDIA image and pinned CUDA/PyTorch stack. Do not add DeepSpeed, FSDP,
Ray, compilation or new kernels without a measured requirement/benefit. Retain
bounded distributed-gradient/resume validation after consequential changes.

Respect workload ownership and GPU leases; do not stop other projects' jobs.
GPU training does not qualify the native Isaac rendering path. No dependencies
were installed, jobs submitted or trainer support claimed by this documentation
change. The project's existing training implementation remains authoritative.

References: [PyTorch DDP](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html),
[torchrun](https://docs.pytorch.org/docs/2.11/elastic/run.html),
[Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
