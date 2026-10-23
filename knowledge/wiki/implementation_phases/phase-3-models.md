# Phase 3 — Model architecture

Scope: original 3.1 and 3.2. Deliver tested multimodal/action components before task training. Current checkpoint/configuration, APIs and limitations are canonical in [models](../models.md); tensor/time conventions are in [contracts](../contracts.md).

## 3.1 VLM backbone integration

**State:** implemented with Qwen3.5-4B and locally validated on the RTX 4090. The website's Qwen2.5 reference is superseded. A100 profiling and production training characterization remain pending.

### Implementation areas

- **3.1.0 Dependencies/configuration:** maintain one shared backbone preset and inherited dev/full VLA presets. Use supported Transformers/PyTorch and native SDPA without an unnecessary attention dependency. Device, freeze policy, cache location and context planning budget must be explicit.
- **3.1.1 Loading:** load pretrained weights lazily through `VLMBackbone`/`get_model`; expose hidden size, LM head, dtype and processor. Preserve safe behavior with automatic device maps. Validate actual placement before a GPU workload; do not accept silent CPU offload as a GPU result.
- **3.1.2 Processor:** preserve the native chat template, image expansion/grid/token-type metadata, tokenizer IDs and ordered four views. Measure token counts from the processor rather than a guessed pixels-to-token formula. Enforce dataset sequence budgets in Phase 2.3; a configuration value alone does not truncate input.
- **3.1.3 Inference sanity:** test text, one image, multiple views, hidden states, freeze behavior and finite native outputs. A dummy recorder is useful for metadata-transport regression only; native GPU checks establish actual model compatibility.
- **3.1.4a Local memory:** retain bounded profiling and distinguish frozen inference, head training and full-backbone training. Preserve OOM as an outcome rather than dropping its configuration.
- **3.1.4b A100:** measure actual production sequence/batch/freeze combinations on an allocated A100. Record runtime, peak allocated/reserved memory, throughput and failed cells. Local inference memory is not an A100 training feasibility measurement.

### Completion gate

The local integration gate requires real four-view forward/hidden states, correct metadata, finite tensors and intended parameter/device/freeze state. This gate passes. Production characterization requires independent A100 evidence and must remain open until run. No inference smoke establishes task understanding or trained action quality.

### Handoff

Expose one processor/backbone contract to the VLA and data pipelines. Resolve any positional or attention mismatch in 3.2 before training a large dataset. Keep exact measured revisions/environment in experiments without creating a second dependency manifest for routine development.

## 3.2 Flow-matching action head

**State:** projectors, flow-matching math, assembled VLA forward and inference implemented. Native GPU component/gradient checks pass. Arbitrary temporal interleaving, native multimodal positional equivalence and learned control are unqualified.

### Implementation areas

- **3.2.0 Chunk/sequence contract:** retain 16×17 action chunks, 52-D state, explicit valid-step masks and modality boundaries. The current assembly groups text/vision, then state, then actions. Define causal/block visibility and validate it with perturbation tests before claiming support for the Phase 2.3 temporal layout. The existence of a mask utility alone does not prove it is used by native attention.
- **3.2.1 Flow matching:** maintain the declared noise-to-data path, matching target sign, bounded time sampling and consistent Euler/midpoint/RK4 denoising. Keep tests for interpolation endpoints, masked tails, constant-velocity solutions and perturbations. Remove no-op parameters instead of pretending they tune the path.
- **3.2.2 State projector:** map normalized state to backbone-width tokens with the existing small MLP. Verify shape, dtype and gradients using actual state conventions; arbitrary simulator object truth is not part of this projector.
- **3.2.3 Noisy-action/output projectors:** fuse action and timestep embeddings, return backbone-width action tokens and predict 17-D velocity. Maintain float32 head arithmetic by default and explicit dtype casts around backbone injection.
- **3.2.4 Model assembly:** scatter native vision features into matching placeholders, append state/action tokens and compute text/action losses on their intended positions. Current sequential positional fallback must be evaluated against correct native multimodal position handling; freezing weights does not resolve it. Integrate the modular losses only through Phase 4.1's verified contract.
- **3.2.5 End-to-end validation:** use real MuJoCo views and Qwen on CUDA. Check finite logits/actions, correct output shapes, intended trainable gradients and absence of frozen-backbone gradients. Include real multimodal training/backward when qualifying that path, rather than extrapolating from a text-only probe. Profile inference latency and memory before selecting a closed-loop execution schedule.

### Completion gates and remaining decisions

Component inference/gradient routing passes locally. Before training, validate full-sequence padding, all-ignored text handling, masked actions, native attention semantics and temporal/modal positional treatment using one real minibatch. Before control, implement a runner that maps predicted normalized actions through the existing controller and explicitly chooses how many of 16 steps to execute before replanning. The historical eight-step proposal is not a current executable setting.

Before claiming a useful policy, evaluate actual completed/released structures under Phase 6.1. Finite untrained actions, a small synthetic loss or prearranged scenes are not success evidence. Keep a simple single-observation-to-action baseline for the later interleaving ablation.

## Source trace

Original roadmap and baseline `docs/roadmap/phase3.1.md` / `phase3.2.md`. Historical validation reports and their limitations are summarized in [history](../history.md). Earlier drafts' Phase 3.3/3.4 training/inference labels are reconciled here into 3.2, 4.1 and 4.2 rather than creating unlisted phases.
