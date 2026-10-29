# Compute and containers

## Historical pilot recordings — October 27, 2026

Both historical pilot `attempts/` trees now reside on **Blackwell**. Their local
copies were removed one batch at a time after complete file/path, size, SHA-256,
permission, nanosecond modification-time, named-ownership and extended-attribute
comparison, unchanged-source checks and fresh process audits. All failed and
interrupted attempts are preserved; scientific outcomes and admission are unchanged.

| Collection | Blackwell destination | Preserved files |
|---|---|---:|
| Marker-free pilot | `/home/pacquadr/projects/vla-lego/outputs/pilot-markerfree-20261023/attempts` | 529,151 |
| Previous marked corpus | `/home/pacquadr/projects/vla-lego/outputs/pilot-collection-20261023/attempts` | 89,773 |

Existing remote copies supplied 270,586 identical files; only 348,338 missing
files were transferred. No content conflicts or remote-only entries were found.
All 618,924 destination files were read completely for the final hashes. The
native reader also accessed all 60 admitted marker-free episodes (333,373 frames),
and metadata/image reads passed for all 35 previous marked attempts. No simulation
or training was started, and healthy AlexDoor/local and VLA/Blackwell jobs were
preserved. Removed allocation was **333.82 GiB**; filesystem measurements immediately
around the two removals showed **333.79 GiB** reclaimed while other jobs continued writing.

Code, indices, sources, reports, visual-review copies, checkpoints and all other
collections remain local. The five marker-free reader indices were also copied
to the remote collection root and SHA-256 verified against the retained originals.
For historical `PilotTrainingData`, `recorded_scenario` or `pilot_audit` access, run
on Blackwell using its collection root. These filesystem readers cannot consume
an SSH URL; restore required attempts explicitly before a future local read.
Original absolute paths in historical reports remain provenance. Do not restart
historical local collection/sync scripts against the retired local trees.

Local manifests, process audits, per-batch removal receipts and restore commands
are retained in `outputs/pilot-storage-migration-20261027/`; each collection root
has an `ATTEMPTS-ON-BLACKWELL.md` pointer. No external dependent symlinks or active
readers needed relocation. Named owner `pacquadr` maps local UID/GID 1002 to
Blackwell UID/GID 1001; permission bits are unchanged.

## Expanded campaign storage — October 26, 2026

A verified **224-episode storage snapshot** is on the campaign corpus in Gilbreth scratch: 14,483 files and 181,230,681,851 bytes, with file sizes and SHA-256 checked against Blackwell. This is the preserved archive from before the Blackwell-only instruction; subsequent admitted episodes remain on Blackwell. Original recordings and failed attempts remain on Blackwell. The mechanism-coverage revision is canonical in the transferred corpus; original and coordination declarations remain preserved. Storage preparation does not establish the complete 620-episode corpus or training readiness. [Data](data.md#expanded-alex003-campaign--october-26-2026) owns admission and learning gates.

After the initial data gate, two native Blackwell collectors completed four additional single-block episodes that pass physical, reader and visual admission. Observed aggregate throughput was 6.70 recorded frames/s versus 4.67 frames/s in earlier single-collector runs, with about 13.2 GiB sampled peak GPU memory. This cross-case engineering measurement supports two bounded collectors; it is not a controlled hardware speed benchmark or policy qualification. Actual-policy correction collection uses the same bounded slots and leases, preserving separate setup, physical and behavioral outcomes.

## Parallel workstation collection — October 28, 2026

The latest user instruction authorizes collection on **Blackwell, the local RTX4090, elab2 RTX4090, and compatible elab3/elab4 RTX3090**. It supersedes October27 Blackwell-only collection. Preserve healthy workloads and serialize ownership with the actual device locks. Use one bounded native episode on each added host before finite, disjoint never-started shards; installed Isaac and an idle GPU do not establish compatible collection. Keep the existing two Blackwell lanes bounded.

All four added hosts have executed native CUDA under verified device UUID and exclusive episode leases. Elab3 and elab4 now pass full original cooperative-tower validation: 5,378 and 5,710 frames, complete physical/data review and five individual images each, admitted after verified Blackwell transfer. They run two fresh disjoint six-case shards, each with four TRAIN, one calibration and one confirmation case, a sixteen-GiB staging bound and stop on the first failure. The local green-source repair remains unproved after a second original-task failure; all original recordings and failures are preserved. No unchanged failed case is retried.

The user-requested Desktop installer has completed elab2's global migration to the same Isaac6.0.1 rc.7 build and compatible pinned Lab revision. Native CUDA, PhysX GPU, RTX rendering, GUI/ROS launchers and a bounded Cartpole installation check pass. Only the twenty-four Isaac6.2 distributions were removed; all 268 unrelated legacy distributions retain their versions in the preserved environment. Canonical terminal, menu and Desktop launchers are coherent. The corpus agent independently verifies the handoff paths, original native proof, reservation and free GPU before starting one fresh bounded original-task control. Frozen generator457 is unchanged, adapter497 is absent, and the old511 launch remains held. A complete Alex003 physical/data result is still required before elab2 bulk collection. Historical6.2 failure and all thirteen zero-frame deployment refusals remain preserved.

Keep the canonical corpus and learning checkpoints on Blackwell. Additional hosts may use owned bounded staging, with complete verified transfer before deletion of only that staging. Preserve AlexDoor and historical data, existing cluster archives, all failures and denominator accounting. No new Purdue GPU jobs or corpus mirrors, timed cross-project turns, peer returns or ordinary scheduling messages. The old local collector remains retired; each new batch has fresh ownership.

## Planned resource roles

| Resource | Intended role | Boundary |
|---|---|---|
| Local RTX 4090, 24 GB | Authorized parallel collection | Respect active workloads and lock ownership; bounded staging. |
| Elab2 RTX 4090, 24 GB | Authorized parallel collection | Isaac6.0.1 migration verified; one original Alex003 control before scaling. |
| Elab3 and elab4 RTX 3090, 24 GB each | Authorized parallel collection | Full native host validation passed; finite disjoint shards and verified transfer. |
| Blackwell workstation, about 96 GB | Broader rendered collection, training and larger offline teachers | RTX PRO 6000 Blackwell Max-Q; current rendered two-collector measurement is reported above. Availability and new workloads still require measurement. |
| Gautschi | Potential future resource, outside the current assignment | Further use requires a changed resource instruction. |
| Gilbreth | Preserve the existing verified archive and historical runs | No new campaign jobs or corpus mirrors under this assignment. |
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
