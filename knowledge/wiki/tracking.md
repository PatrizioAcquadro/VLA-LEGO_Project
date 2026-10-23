# Experiment tracking

`src/vla_lego/tracking/` is an existing optional integration library; there is no active trainer. [Phase 5](implementation_phases/phase-5-training.md) may reuse it where it fits the selected VLA's training path, without duplicating upstream logging. Metrics helpers do not require W&B; install the `tracking` extra to use its backend. No external dashboard or upload was created by this documentation revision.

Future reporting distinguishes attempted collection cases, usable nominal/corrective segments, autonomous rollout outcomes and assisted interventions. Include source/case mixture, learning curves, memory and full inference latency. The [data strategy](data.md) owns dataset accounting; [validation](validation.md#assembly-qualification-targets) owns qualification denominators. A lower loss or successful upload is not learned assembly success.

`ExperimentTracker` is the public adapter: initialize a run with config/name/tags, log training metrics, save configuration/checkpoint artifacts when explicitly requested, then finish. It initializes on the main distributed rank only and supports online, offline and disabled operation. Online initialization may fall back to offline; inspect the actual run state before claiming publication. Use `mode="disabled"` or `enabled=False` for code paths that must not contact W&B.

```python
from vla_lego.tracking.metrics import extract_loss_components

metrics = extract_loss_components({
    "total_loss": 3.0, "text_loss": 1.0, "action_loss": 2.0,
})
assert metrics == {"loss/total": 3.0, "loss/ar": 1.0, "loss/fm": 2.0}
```

The total is authoritative; it is not recomputed by summing the total and its components. Gradient norms, learning rate, throughput and optional GPU statistics have separate helpers. GPU telemetry can be unavailable and must not be represented as zero utilization or evidence of CPU-only execution. Metadata gathers revision/config/seeds/environment for actual experiments; it does not establish dataset provenance or scientific reproducibility by itself.

When a trainer is added, explicitly define logging cadence, sample/token denominators, gradient accumulation boundaries and checkpoint ownership. Keep logging outside autograd and save from the correct rank. Checkpoint resumability must be tested by the trainer rather than inferred from a successful artifact upload.
