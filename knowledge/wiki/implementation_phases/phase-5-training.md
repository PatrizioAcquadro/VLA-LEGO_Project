# Phase 5 — Training execution

Scope: original 5.1 and 5.2. **State: planned; no task-trained checkpoint or learning curve exists.** Start only after Phases 2–4 deliver a qualified dataset, correct multimodal/loss path, single-GPU resume and a measured resource budget. The original multiweek eight-A100 schedule is historical.

## 5.1 Initial training run

### Run design

Define the primary hypothesis before using a large allocation: can the maintained observation/state/action model learn real lightweight-brick pick-and-place and generalize to held-out episode groups? Begin with the simplest validated action-conditioning layout. Interleaved reasoning is an experimental factor, not an assumed improvement.

Fix training/validation splits, data mixture, normalization, model revision, freeze policy, action horizon, optimizer/scheduler, global batch and compute budget. Declare validation cadence, checkpoint selection, maximum steps/wall time and stopping conditions. Stop for nonfinite losses/gradients, data leakage, inconsistent action alignment, repeated unrecoverable runtime failures or exhausted budget. Do not extend a run simply until a favorable checkpoint appears.

### Execution

1. Run a bounded pilot using actual recorded episodes and the GPU. Confirm data loading, updates, memory, logging and restart with the production configuration reduced only in scale.
2. Launch the initial training run at the qualified device/node scale. Track total/text/action objectives, valid sample/token counts, gradient norm, learning rate, throughput and memory.
3. Evaluate fixed held-out losses and periodic closed-loop task trials through the actual controller. Keep teacher/reference results separate from learned commands.
4. Diagnose issues from concrete evidence. Change one causal factor at a time where practical; retain the baseline run and earlier failures. Do not alter the test set or task tolerances in response to a disappointing score.

### Completion gate and artifacts

Deliver a reloadable checkpoint, complete run configuration, denominator-aware learning curves, stop reason, resource measurements and fixed held-out task results. A finite or decreasing loss meets an optimization diagnostic only. The original >50% simple-task aspiration requires a predefined brick task set and actual released/stable assemblies before it can become a task result.

## 5.2 Continued training with adjustments

### Decision sequence

1. Review the initial run against fixed validation and failure taxonomy. Distinguish perception/grounding errors, grasp/contact failures, placement errors and horizon/controller problems before modifying the model.
2. Continue unchanged when evidence supports it. If changing hyperparameters, freeze policy, data mixture or curriculum, state the reason and keep an interpretable comparison to the prior checkpoint/budget.
3. Introduce curriculum only for a demonstrated bottleneck: single placements → short towers → varied sizes/planar layouts → later bimanual tasks. Preserve held-out difficulty and avoid treating a simpler curriculum stage as improved final-task success.
4. Save a bounded set of checkpoints at fixed intervals and milestone decisions. Select by the preregistered validation rule, never the final test score. Keep necessary optimizer state for continuation and avoid redundant artifact copies without a retention purpose.
5. Record consequential training decisions, resource use and negative results. Ordinary source changes belong in Git, not a growing experiment diary in the wiki.

### Completion gate and handoff

Provide the selected model and earlier baseline, checkpoint-selection rule, total compute/data exposure, held-out results with uncertainty and remaining failure modes. “Training complete” means the declared budget/stopping condition was reached; it does not imply the target quality was achieved. The historical >70% target and >85% stretch aspiration must be interpreted through the actual Phase 6 benchmark, with failures retained.

Phase 6 receives reloadable candidates and an explicit baseline. Phase 7 must start from a declared checkpoint without retrospectively changing pretraining exposure. No physical-transfer claim is permitted from simulation training alone.

## Source trace

Both subphases come from the website's training execution stage: initial launch/monitoring/early fixes, followed by justified adjustments, possible curriculum, checkpoint comparison and a concise decision record. Implementation details now depend on the real-data trainer rather than the retired placeholder stack.
