# Phase 5 — Iterative learning

## Objective

Turn the selected pretrained VLA and initial Phase 4 corpus into an autonomous assembly candidate, then improve it through actual rollout errors and targeted corrections.

The expanded 620-episode corpus and 30 actual-policy corrections are admitted. The separate temporal classifier completes 20k updates but fails calibration. The independent motor positive control completes 20k updates on native Blackwell CUDA; its ten-case unassisted physical panel finishes 0/10 integrated successes, failing the 8/10 advancement target. Both learning targets remain failed. Bounded diagnosis finds no demonstrated integration defect justifying an unchanged rerun, so motor learning stops at 20k without a full-task extension and leaves 80k of the ceiling unused. The final paired comparison is complete and interpreted: historical reference 0/20 and current policy 0/20, with all forty attempts executed once and no unavailable outcomes. Confirmation is exposed only for frozen final validation, not further model or threshold selection. The negative campaign is delivered, without a qualified physical-best candidate or Phase 5 advancement; [Models](../models.md#expanded-corpus-learning-assignment--october-26-2026) owns the results and limits. Phase 6 qualification is not claimed.

## Design choices

Use the selected model's supported adaptation path. Start with the least expensive adaptation scope that can learn the task; widen capacity only for an observed limitation. Use deliberate source/case weights, preserving useful nominal behavior while adding recovery and real adaptation later.

The policy owns piece selection, sequence, arm roles and corrections. The controller realizes bounded motions; no teacher or task-state machine completes autonomous rollouts.

## Conceptual activities

- Establish real training that can continue from saved state and produce deployable policy weights with the matching input/action normalization.
- Fine-tune on positive nominal and corrective targets, treating failed/context/stop segments according to their declared learning role.
- Measure full-task behavior on development episodes, separately from teacher performance.
- Collect interventions on actual policy errors and return them to Phase 4; repeat the collection–learning loop.
- Attribute failures to coverage, perception, control, timing or capacity before increasing model or data scale.
- Select checkpoints using held-out development outcomes and verify that deployment optimization preserves the relevant behavior.

## Expected result

One selected, reloadable candidate with its observation/action contract, preprocessing, normalization and Orin execution settings; an account of training exposure, autonomous development outcomes and remaining failures.

## Advancement conditions

The candidate autonomously completes full cooperative towers on development episodes and has demonstrated recovery on the admitted error curriculum. Its development results justify attempting all three simulation qualification targets, under the intended execution timing. This is readiness to attempt qualification, not a substitute for Phase 6. Final qualification episodes never select checkpoints. If progress stalls, make a bounded data/control/model intervention or report the unresolved limitation rather than training indefinitely.
