# Phase 3 — Data sources and teachers

## Objective

Establish the data foundation: relevant public knowledge, physically valid Alex demonstrations and useful recovery supervision.

## Design choices

Reuse robotic pretraining, selected public robot episodes and an Alex-specific simulated corpus. Human egocentric data is optional and must provide measured benefit under compatible terms; hand trajectories are not Alex actuator labels.

Start with a geometric teacher combining motion planning, feedback and bimanual coordination. Use human demonstrations or corrections for behavior it cannot cover. Larger models may propose programs or annotations offline, but physically executed trajectories remain the authority. The [data strategy](../data.md) owns source admission, case distribution and supervision boundaries.

## Conceptual activities

- Select public sources by actual availability, terms, embodiment, modalities, action semantics and task relevance; remove aggregate overlap.
- Produce nominal grasp–transport–place–release sequences and recoveries from consequential errors, including coordination failures.
- Preserve context, failed actions, corrective interventions and subsequent outcomes without teaching failed actions as successful targets.
- Expand object-relative subsequences using Mimic/DexMimicGen principles while preserving dependencies between hands.
- Establish a small pilot spanning nominal, recoverable and non-completable cases, with grouped splits before windowing.

## Expected result

An admitted source shortlist, executable teachers and a pilot demonstrating useful behavior and corrections, with all attempted episodes accounted for.

## Advancement conditions

Pilot actions are physically grounded, observations match the deployable interface, and recovery examples include verified outcomes. Source terms, overlap and action mappings are resolved for admitted data. Scaling waits for Phase 2; public corpus size or fluent annotations alone do not establish readiness.
