# Phase 0 — Foundation and setup

Scope: original 0.1, 0.2 and 0.3. Establish a usable engineering/research environment before collecting data or training. The website's original one-week calendar is historical. Current operating commands live in [development](../development.md), [simulation](../simulation.md) and [cluster](../cluster.md).

## 0.1 Development environment

**State:** local GPU runtime, package/container workflows and optional tracking components exist. Real VLA multi-GPU training, cluster checkpoint/resume and hosted tracking are unqualified. Original toy jobs were retired.

### Implementation and remaining steps

1. **Scheduler access:** inspect the actual account, partitions, limits, GPU topology and scratch quota. Maintain the one-GPU simulation/model templates. Verify a queued job starts, writes logs and exits correctly before adding resource scale.
2. **CUDA/PyTorch:** use the declared dependency stack and record actual driver/runtime/device versions. Verify real model forward/backward on the GPU. A dummy checkpoint is no substitute for a VLA optimizer/resume check; that gate belongs to 4.2.
3. **Distributed setup:** the original plan requested DeepSpeed ZeRO-1 and eight A100 GPUs. Treat this as a future candidate architecture, not installed capability. First establish a correct real single-GPU path; then justify DDP/ZeRO from measured memory and throughput. Do not retain unused configurations in anticipation.
4. **Tracking:** integrate total/AR/FM loss, learning rate, gradient norm, throughput, seed/config/revision and checkpoint location through the existing adapter. Check rank ownership and offline status before making dashboard claims.
5. **Repository/CI:** preserve one coherent package, packaged assets/configuration, small local commits and required software checks. Validate installed code outside the checkout. Container publication and releases remain separate authorized actions.

### Completion gate

Local environment completion requires package imports, maintained config composition, native simulation and real GPU model checks. Cluster completion additionally requires a successful scheduler run with preserved logs and a real reloadable training state from 4.2. Report local, single-node and multi-node qualification separately. Current local evidence is in [validation](../validation.md); none establishes eight-GPU readiness.

## 0.2 MuJoCo stack and visualization

**State:** implemented and locally validated native/offscreen workflows; GUI and remote cluster operation were not exercised during this audit.

### Maintained substeps

- **0.2.1 Runtime:** keep MuJoCo loading/stepping deterministic within the supported environment, with finite state and physical limits. Native dynamics use CPU; GPU-capable rendering uses the actual GPU.
- **0.2.2 Viewer:** maintain one scene-selectable desktop entrypoint for geometry, contacts, cameras and joint inspection. Viewer import tests do not replace a displayed session when GUI changes are made.
- **0.2.3 Offscreen:** maintain RGB, depth, segmentation, fixed camera selection, timestamps and bounded image/video export. Inspect real outputs rather than accepting file creation alone.
- **0.2.4 Smoke:** keep deterministic stepping, finite-state, alignment and export regressions. One operator smoke plus the maintained test suite replaces duplicate generic validator scripts.
- **0.2.5 Assets:** resolve includes/meshes from the package, lint missing references, and verify scenes from an installed wheel outside the source directory. Preserve source provenance for commissioned robot assets.
- **0.2.6 Cluster compatibility:** run EGL without a window on an allocated GPU; preserve renderer/driver/version metadata and scheduler exit status. Use OSMesa explicitly only for the GPU-less CI path.

### Completion gate

The asset linter, native physics, synchronized modalities and diagnostic exporter must pass with inspectable nonempty images. Installed-package access must work from another working directory. Re-run desktop or cluster checks only when changes affect those paths. The original website's scripted assembly success milestone is a separate Phase 2.1 task gate, not proof of runtime installation.

## 0.3 Literature and technical design

**State:** targeted EO-1 reference notes and local model decisions exist; a complete comparative research synthesis has not been demonstrated.

### Work to complete

1. Read EO-1 architecture, data mixture and optimization methodology against the exact reference code. Capture assumptions that affect this implementation rather than copying a second codebase.
2. Compare pi0/pi0.5 and EO-1 on action representation, flow time direction, embodiment conditioning, observations, data requirements and compute. State which comparison is relevant to lightweight-brick manipulation.
3. Derive the local conditional-flow path, target velocity, masking and integration direction; connect the mathematics to existing code/tests. Avoid confusing beta time-sampling conventions between reversed paths.
4. Review Qwen3.5 processor expansion, positional encoding, freeze behavior and context/memory tradeoffs. The website's older Qwen2.5 notes are background, not the current implementation contract.
5. Explain what interleaved reasoning is expected to improve and how Phase 6.3 could falsify that hypothesis against a simpler observation-to-action baseline.

### Completion gate and handoff

Produce a concise source-linked design rationale, unresolved implementation questions and testable comparisons. Keep [references](../references.md) canonical; put code-specific decisions in the relevant model/data plan. Distinguish published external results from local measurements. Handoff to Phases 2–4 requires clear assumptions and failure tests, not a bibliography alone.

## Source trace

[Original roadmap](https://websitepresentation.vercel.app/roadmap) and its 0.1/0.2 detail pages; the removed 0.2 draft is recoverable at `31bffb75:docs/roadmap/phase0.2.md`. See [history](../history.md) for exact revision pointers.
