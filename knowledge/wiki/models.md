# Models

## Planned pretrained VLA selection

Use a model with existing robotic pretraining, then adapt it to Alex003. The decision is conditional on task behavior and onboard execution, not a mandatory comparison of every architecture. No candidate below is integrated or qualified in this repository yet.

| Candidate | Why consider it | Decision boundary |
|---|---|---|
| [GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) | Leading candidate: released 3B VLA, cross-embodiment robotic pretraining and documented Jetson deployment. | Demonstrate Alex adaptation, complete latency, memory and host compatibility on the actual Orin. |
| [SmolVLA](https://huggingface.co/blog/smolvla) | Compact alternative: 450M parameters, robotic pretraining and an asynchronous execution approach. | Prefer it if it meets task requirements with a workable Orin stack where GR00T does not. Compact size alone is not qualification. |
| [π0.5 / openpi](https://github.com/Physical-Intelligence/openpi) | Released pretrained manipulation model and established adaptation tooling. | Technical comparator; integration only if it offers a concrete advantage within the onboard budget. |
| [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) | Released 6B model with broad robot/human pretraining and cross-embodiment action representation. | Larger comparison candidate; desktop latency is not evidence of Orin feasibility. |

Compare the same Alex observation boundary, pilot task distribution and physical controller. Measure block selection, full completion, cooperation and recovery, plus memory with concurrent services and observation-to-applied-command latency. Choose one primary model; additional integrations must earn their maintenance cost. Checkpoint and code terms must be compatible with intended training and deployment.

## Onboard feasibility

The [Alex guide](https://docs.google.com/document/d/17QtexPK_RqmfRammA7CsvEUuhFZFkicrJfONCdlWkEg/edit?tab=t.sfoy44dcf7kc) describes a Jetson AGX Orin 64 GB on the robot, a Forge carrier and JetPack 6.2.1. As reviewed October 23, 2026, the [GR00T deployment guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md) specifies JetPack 7.2 / Jetson Linux 39.2 / CUDA 13.2 for its current Orin path and excludes JetPack 6.x. That is an unresolved integration constraint, not an instruction to upgrade the robot.

Establish compatibility with the carrier/BSP, ZED drivers, communication and control services before selecting the runtime. A container cannot supply missing host support. If a compatible GR00T path cannot be demonstrated, qualify SmolVLA or another justified smaller path; workstation VLA inference is not an accepted final fallback.

NVIDIA's [hardware guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md) reports approximately 6.6 Hz TensorRT inference on Orin for one camera and four denoising steps. This is external evidence for a particular configuration, not an Alex measurement, actuator rate or guaranteed task responsiveness. Include image acquisition/preprocessing, state transport, policy inference, command delivery, observation age and missed deadlines in the local budget.

The same guide recommends at least 40 GB for its default fine-tuning path and 80 GB for full fine-tuning. The local 24 GB GPU is therefore for inference/profiling or explicitly supported smaller adaptations, not assumed default GR00T training. Use the [compute plan](cluster.md#planned-resource-roles) for larger runs. Select precision, adaptation scope, predicted chunk and executed prefix from measured behavior; no inherited 16-step requirement applies.

## Existing Qwen path

This retained component path is a migration baseline, not the selected future policy. It remains operational until an appropriate replacement exists; completing its custom interleaving or reproducing EO-1 is not required.

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

`VLAModel` scatters visual features into token embeddings, appends projected state and noisy action tokens, runs the shared backbone and predicts action velocity. The MLP head stays float32 by default. `predict_actions()` denoises 16 actions with ten Euler steps by default; midpoint and RK4 math are available. The component forward returns weighted text/action losses, while standalone modular losses live in `src/vla_lego/models/losses.py`. Their production combination remains unfinished and is no longer a required roadmap milestone.

## Existing Qwen limitations

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
