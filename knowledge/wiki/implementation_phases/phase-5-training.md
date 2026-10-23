# Phase 5 — Iterative learning

## Objective

Adapt the selected pretrained VLA into a policy that chooses the correct pieces and coordinates assembly and recovery from deployable observations.

## Design choices

Follow the selected model's supported adaptation path. Begin with the least expensive adaptation scope that can learn the task; widen trainable capacity only for an observed limitation. Preserve pretrained knowledge with a deliberate, source-aware mixture.

Task sequence, piece choice and correction decisions belong to the policy. Kinematics and physical command limits belong to the controller. No hidden teacher completes autonomous rollouts.

## Conceptual activities

- Fine-tune on the admitted corpus with appropriate successful and corrective action targets.
- Measure full-task rollout behavior on development episodes, including language grounding, coordination and recovery.
- Collect corrections to actual policy errors, then repeat training; use the practical intervention principle from IntervenGen and RECAP.
- Diagnose whether failures arise from coverage, perception, control, timing or model capacity before increasing data or model size.
- Retain useful earlier behavior while adapting, and select checkpoints by task outcomes rather than training loss alone.
- Scale compute only when it removes a measured bottleneck.

## Expected result

A reloadable candidate policy and an evidence-based collection/adaptation recipe, with observed improvements and remaining failure modes.

## Advancement conditions

Autonomous development rollouts demonstrate the intended behavior without privileged observations or external task selection. The candidate has a feasible Orin execution configuration and enough evidence to justify the fixed [qualification campaigns](../validation.md#assembly-qualification-targets). Final qualification data is not used for routine tuning.
