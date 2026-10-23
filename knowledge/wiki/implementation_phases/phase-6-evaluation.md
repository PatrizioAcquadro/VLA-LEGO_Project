# Phase 6 — Simulation qualification

## Objective

Decide whether one fixed learned candidate is ready for controlled physical transfer. Entry requires the Phase 5 candidate and the declared task, held-out groups and calibrated evaluation protocol.

## Design choices

Apply the canonical [task definition](../bricks.md#target-task) and all three [simulation qualification targets](../validation.md#assembly-qualification-targets). The same checkpoint, observation boundary and execution configuration must satisfy the separate nominal, generalization and recovery campaigns.

Use measured Orin-equivalent observation age, replanning latency and command timing, including deployment precision/horizon choices. Desktop speed alone is insufficient.

## Conceptual activities

- Fix episode sets, distribution, timeout, allowed recoveries, perturbations and evaluator tolerances before the campaign.
- Evaluate nominal five-block assembly, held-out generalization and recovery-to-full-completion separately.
- Verify actual cooperative contribution and final release/stability; do not score a corrected grasp alone as a recovered task.
- Report counts, confidence intervals, timing, interventions and failures, including per-axis generalization and per-error recovery results.
- Assess appropriate stops separately from assembly completion and report unnecessary refusals on feasible cases.

## Expected result

A qualification decision for the fixed candidate, complete campaign outcomes and a ranked account of remaining physical-transfer gaps.

## Advancement conditions

The candidate meets all three simulation targets on their full declared denominators, without hidden assistance, favorable retries or adjusted acceptance ranges. A passing aggregate cannot hide a missing required campaign. Failure returns to targeted development; retain the result and reserve fresh qualification episodes if these episodes inform learning. Passing this phase allows controlled transfer work, not a claim of real-robot success.
