# Phase 7 — Fine-tuning and adaptation

Scope: original 7.1 and 7.2. **State: planned.** Requires a declared Phase 5 model, Phase 6 fixed evaluation protocol and newly collected adaptation data. No task-specific/few-shot result exists. The website's LEGO-kit wording is replaced by lightweight-brick sets and arrangements.

## 7.1 Task-specific fine-tuning

### Experiment sequence

1. Choose one concrete adaptation target, such as a new brick dimension distribution, tower height or arrangement family. Define its training demonstrations and held-out evaluation groups without leaking the final test layouts.
2. Evaluate the starting checkpoint before adaptation. Fix data exposure, optimization budget, trainable modules and checkpoint-selection rule. Prefer the existing head/freeze interfaces; add LoRA or another adaptation dependency only if a measured memory/quality need justifies it.
3. Fine-tune on the GPU using the qualified trainer/resume path. Record adaptation updates, examples, compute and any failed runs. Preserve the starting model for a real before/after comparison.
4. Evaluate both the target task and the original task distribution under the same controller/evaluator. Quantify gains, unchanged behavior, regressions and catastrophic forgetting. Include a from-scratch or head-only baseline where it answers the actual adaptation hypothesis within the available budget.
5. Derive a concise recipe from observed results: which modules change, data required, effective batch/steps, resource limits and situations where adaptation did not help.

### Completion gate

Report paired baseline/adapted outcomes, total attempted tasks, uncertainty, adaptation data/compute and original-task retention. A lower target-training loss alone is not improved task performance. Stop at the preregistered budget or a defined numerical/data failure; do not change the test set to rescue an adaptation claim.

### Handoff

Deliver reloadable adapted and baseline checkpoints, a clear target distribution and an evidence-based recipe. Keep physical calibration changes separate from learned adaptation so their contribution remains interpretable.

## 7.2 Few-shot adaptation

### Experiment sequence

1. Retain the original proposed demonstration counts of 10, 50 and 100 as experiment candidates, adjusted only before the study if available real episodes or the scientific question require it. Count complete usable trajectories, not windows sampled from the same few episodes.
2. Draw nested, grouped adaptation sets from a fixed eligible pool, with independent sampling seeds and a held-out evaluation set. Declare the number of repeats and compute budgets in advance. Preserve failed collection attempts separately from usable demonstration count.
3. Compare adaptation from the declared pretrained checkpoint against training from scratch with matched data, task inputs and explicitly reported optimization/compute. Do not equate equal updates with equal total pretraining exposure.
4. Evaluate fixed tasks and generalization axes with the Phase 6 runner. Report every count/seed cell, confidence/variability and failures. Do not choose the best subset or repeatedly resample until the curve becomes monotonic.
5. Estimate the smallest demonstrated data quantity that meets a predefined task criterion, with uncertainty. If no count reaches the criterion, report that result; do not extrapolate a guaranteed sample requirement.

### Completion gate and handoff

Deliver learning curves by actual demonstration count, all repeat outcomes, matched-baseline definitions, data/compute exposure and failure taxonomy. This characterizes data efficiency only within the tested distribution. It does not establish zero-shot transfer, new geometry classes or hardware adaptation. Pass concrete collection and calibration requirements to Phase 8 rather than an unsupported universal fine-tuning recipe.
