# Models

## Implemented path

`get_model(cfg)` routes `model.architecture.type` to a lazy `VLMBackbone` or `VLAModel`. The maintained checkpoint is `Qwen/Qwen3.5-4B`: hidden size 2560, vocabulary 248320, bf16 backbone with PyTorch SDPA by default. Weight downloads and model loading happen only when requested. No trained brick-action checkpoint is included.

```python
from hydra import compose, initialize_config_module
from vla_lego.models import get_model

with initialize_config_module(config_module="vla_lego.configs", version_base=None):
    cfg = compose(config_name="config", overrides=["model=vla_dev"])
model = get_model(cfg)  # Downloads/loads pretrained weights; use a CUDA machine.
```

`vlm`/`vla` are backbone/full-assembly presets. Their `*_dev` variants declare a smaller context planning budget (4096 rather than 8192). `max_seq_length` is planning metadata: the wrapper does not automatically truncate inputs to it. Dataset construction must count native processor tokens and enforce the complete sequence budget explicitly. Shared values live in `vlm.yaml` and `action_head.yaml`; do not duplicate them in presets.

`preprocess_images()` accepts RGB uint8 arrays or PIL images and preserves native processor tensors. `VLMBackbone.forward()` returns logits; `get_hidden_states()` returns final representations. Image-only legacy callers that pass a grid without token types receive derived `mm_token_type_ids`; explicit metadata is preserved. `freeze_backbone()`, `freeze_vision()` and `unfreeze_backbone()` control trainable parameters. `device_map="auto"` can offload when memory is insufficient; native validation deliberately refuses CPU-resident parameters.

`VLAModel` scatters visual features into token embeddings, appends projected state and noisy action tokens, runs the shared backbone and predicts action velocity. The MLP head stays float32 by default. `predict_actions()` denoises 16 actions with ten Euler steps by default; midpoint and RK4 math are available. The component forward returns weighted text/action losses, while standalone modular losses live in `src/vla_lego/models/losses.py`. Their production combination is [unfinished](implementation_phases/phase-4-training-infrastructure.md#41-loss-function-implementation).

## Known limits

- Injected embedding sequences use sequential position IDs rather than verified native multimodal spatial positions. Frozen weights do not establish equivalent behavior.
- Current assembly groups text/vision, state and action tokens; arbitrary temporally interleaved episodes and their attention semantics are not implemented by a production data pipeline.
- Native smoke proves finite outputs, shapes and a text-only synthetic gradient probe. It does not prove multimodal training convergence, semantic reasoning quality, action accuracy or safe control.
- All action-head weights are untrained for this task. Never interpret a finite `(1,16,17)` tensor as a useful policy.

## Validate and profile

```bash
MUJOCO_GL=egl vla-validate-model --model-config vla_dev --output-dir outputs/model-check-001
python scripts/profile_vlm_memory.py --quick --model-config vlm_dev --output-dir outputs/profile-001
```

Both require CUDA. The smoke uses real four-view MuJoCo images, checks native logits and a VLA action chunk, and records a bounded synthetic gradient-routing check without executing predictions. Output directories must be new. The profiler intentionally explores memory limits and can encounter OOM; it is a diagnostic, not a qualified training recipe. Do not repeat a large sweep without a concrete need. [Validation](validation.md) separates current results from historical estimates and untested A100 configurations.
