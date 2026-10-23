# Phase 6 — Simulation qualification

## Objective

Determine whether the learned policy is ready for controlled physical transfer, including nominal assembly, generalization and recovery.

## Design choices

Use the canonical [task definition](../bricks.md#target-task) and [qualification targets](../validation.md#assembly-qualification-targets). Evaluate autonomous decisions with the same observation boundary, command limits and representative latency as onboard execution.

Success requires correct block identities and order, meaningful cooperation, complete release and calibrated dwell stability. Two arms in the scene do not establish bimanual behavior.

## Conceptual activities

- Fix campaign conditions, timeouts, allowed recoveries, tolerances and unseen groups before collecting outcomes.
- Measure nominal five-block assembly, dedicated generalization cases and recoverable perturbations separately.
- Include realistic camera delay, inference/replanning delay and command execution timing from Orin.
- Examine wrong-piece choices, interference, instability, unnecessary refusal and unsuccessful recovery, retaining all attempted episodes.
- Report counts, uncertainty, completion times, interventions and representative failures as well as success.

## Expected result

A bounded assessment of what the candidate can do, where it fails and which simulation-to-real gaps remain.

## Advancement conditions

Meet the simulation criteria in the canonical qualification table without retuning thresholds or substituting favorable retries. If a campaign drives further learning, keep its results and reserve fresh independent qualification episodes. Hardware readiness is an additional transfer condition; simulation performance does not establish physical success.
