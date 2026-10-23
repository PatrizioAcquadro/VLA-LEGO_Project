# Compute and containers

## Planned resource roles

| Resource | Intended role | Boundary |
|---|---|---|
| Local RTX 4090, 24 GB | Interactive development, visual simulation, small experiments and profiling | Larger VLA adaptation depends on measured fit; do not assume full fine-tuning fits. |
| Blackwell workstation, about 96 GB | Broader rendered collection, training and larger offline teachers | Planning inspection identified RTX PRO 6000 Blackwell Max-Q. Shared availability and throughput still need workload-specific measurement. |
| Gautschi | H100 training; RTX-capable L40-class nodes for rendering when available | Partition visibility is not a guaranteed allocation; select the actual available GPU and compatible runtime. |
| Gilbreth | A100/H100 training; compatible A10 nodes for rendering | Access, account limits, memory and allocation must match the job. |
| Robot Orin | All final VLA inference together with sensing, logging and robot communication | Qualify concurrent load on the real system; the NUC retains low-level control. |

The planning inspection found the local 4090 and approximately 96 GB Blackwell device, Gautschi H100/L40 partitions and Gilbreth A10/A100/H100 partitions. These are hardware/access observations, not evidence of qualified Isaac or VLA jobs. [Gautschi](https://docs.rcac.purdue.edu/userguides/gautschi/overview/) and [Gilbreth](https://docs.rcac.purdue.edu/userguides/gilbreth/overview/) documentation describe system resources; actual allocations govern use.

[Isaac Sim's RTX renderer requires supported RT hardware](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html). A100/H100 are training destinations, not supported visual-rendering destinations. Separate rendered collection from training rather than treating every CUDA GPU as interchangeable.

Maintain separate compatible environments for Isaac/Alex, model training and Orin deployment. The shared Alex package and selected VLA may require different Python/CUDA stacks. Choose collection and distributed-training scale from measured useful-data throughput, memory, storage and queue availability; no eight-GPU allocation or mandatory distributed framework is assumed.

The [model page](models.md#onboard-feasibility) owns Orin compatibility and memory constraints. No new simulator, scheduler, training or deployment workload was run for this documentation revision.

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
