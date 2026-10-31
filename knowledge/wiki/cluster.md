# Compute and storage

## Native environments

Keep simulation, training and onboard environments separate. The executed Blackwell campaign used:

| Component | Validated environment/revision |
|---|---|
| Isaac | 6.0.1 rc.7; native Python 3.12.13, PyTorch 2.10.0+cu128 |
| Isaac Lab | `ffff603eafc6b74264a5261cc0183d6a65390d78` |
| Shared Alex assets | `9ac4bd45c779d3f73d61f5b144cd1bb76dc8607b` |
| LeRobot | `f9e9d739ea7055b282ea1b165e8785412a674693`; separate Python 3.12.13, PyTorch 2.11.0+cu128 |
| Native launch inputs | `ISAAC_SIM_DIR`, `ISAAC_LAB_DIR`, `ALEX_ASSETS_ROOT`; optional verified `VLA_ISAAC_PYTHON` and `VLA_VIDEO_PYTHON` |

The environment record is `outputs/data-expansion-20261026/physical-native-environment-1765.json`; it identifies exact installations and frozen source. These are executed-revision facts, not permission to upgrade a host or a guarantee for another installed version. Core package installation does not provision Isaac or LeRobot. Existing model weights and upstream source are reused without Hub uploads.

## Planned resource roles

| Resource | Established use and boundary |
|---|---|
| Blackwell | Canonical corpus/checkpoints, native collection, motor/decision learning and physical simulation panels |
| Local RTX 4090 and elab2–elab5 | Bounded compatible collection/diagnostics when explicitly allocated; each added host required its own native original-task proof |
| Purdue/Gilbreth/Gautschi | Historical pilot training and archive storage; future jobs require actual allocation and stage-specific justification |
| Jetson AGX Orin | Required destination for all online learned inference; not yet qualified |

The completed campaign leaves no automatic successor. Check actual UUID, persistent reservation metadata, exclusive/shared lock protocol, healthy processes, disk capacity and ownership before a new workload. An idle GPU does not revoke another project's reservation. Use native GPU access for supported model/rendering checks, resolving sandbox access rather than silently using CPU. A100 training evidence does not establish RTX rendering support.

## Canonical campaign storage

The complete expanded corpus and learning artifacts live on Blackwell under `/home/pacquadr/projects/vla-lego/outputs/data-expansion-20261026`. Local controls/reports refer to this authority; the historical 224-episode Gilbreth snapshot is a partial archive, not the complete corpus. Do not create another corpus authority or rewrite original report paths. [Data](data.md#expanded-alex003-campaign--october-26-2026) owns membership and admission.

Transfers preserve complete recordings, producer sources and failures. Verify content and reader access before deleting only an explicitly authorized staging copy. Storage availability or a successful copy is not task/data qualification.

## Historical pilot recordings — October 27, 2026

Both complete pilot `attempts/` trees were verified on Blackwell before their local copies were removed:

| Collection | Blackwell root | Preserved files |
|---|---|---:|
| Marker-free pilot | `/home/pacquadr/projects/vla-lego/outputs/pilot-markerfree-20261023/attempts` | 529,151 |
| Earlier marked corpus | `/home/pacquadr/projects/vla-lego/outputs/pilot-collection-20261023/attempts` | 89,773 |

Indices, reports, source snapshots, review copies and checkpoints remain local. Use the Blackwell collection root for historical filesystem readers, or explicitly restore the required attempts before a local read; readers cannot consume an SSH URL. Original absolute report paths remain provenance. Migration evidence and restore commands are retained in `outputs/pilot-storage-migration-20261027/`, with `ATTEMPTS-ON-BLACKWELL.md` pointers in each collection root. Do not restart old collection/sync drivers against retired local trees.

## Shared multi-GPU practice

The maintained trainer is single-GPU. Distributed execution is a future optimization that requires demonstrated benefit and its own integration/resume validation.

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
GPU training does not qualify the native Isaac rendering path. Preserve this project's actual training implementation and recipe when evaluating additional resources.

References: [PyTorch DDP](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html),
[torchrun](https://docs.pytorch.org/docs/2.11/elastic/run.html),
[Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
