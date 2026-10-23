# Development roadmap

Build an **ordered five-block tower with Alex003**, selecting the requested colors and sizes among distractors and coordinating both arms. Establish the capability in Isaac Sim/Lab, then on the Purdue robot with VLA inference entirely on its Jetson AGX Orin. The canonical [task](../bricks.md), [interfaces](../contracts.md), [data strategy](../data.md) and [qualification criteria](../validation.md#assembly-qualification-targets) define the shared requirements.

This roadmap replaces the original 22 subphases. Phase 0 now has a verified native GPU foundation; the minimum grasp/release scene and Phases 1–6 remain in development or pending, as recorded in [status](../status.md). Existing MuJoCo/Qwen components are a migration baseline. There is no fixed calendar or guaranteed cluster allocation.

| Phase | Intended result | Depends on |
|---|---|---|
| [0 — Cleanup and foundations](phase-0-foundation.md) | Essential repository, clear ownership and separate simulation, training and onboard environments | Current baseline |
| [1 — Alex003 and blocks in Isaac](phase-1-simulation.md) | Hardware-faithful scene and physically executed cooperative stacking | 0 |
| [2 — VLA and onboard feasibility](phase-2-vla-feasibility.md) | Pretrained model choice, deployable interfaces and demonstrated Orin execution path | Minimum usable environment from 1 |
| [3 — Data sources and teachers](phase-3-data-teachers.md) | Selected public sources and an Alex pilot with nominal behavior and recovery | Minimum usable environment from 1 |
| [4 — Collection and coverage](phase-4-collection.md) | Corpus covering task variation, errors and corrections | 1, 2 and 3 |
| [5 — Iterative learning](phase-5-training.md) | Fine-tuned policy improved through autonomous rollouts and targeted corrections | Initial corpus from 4 |
| [6 — Simulation qualification](phase-6-evaluation.md) | Measured assembly, generalization and recovery at Orin-equivalent timing | 5 |
| [7 — Physical transfer](phase-7-transfer.md) | Measured transfer, then targeted real adaptation and onboard qualification | 6 and hardware readiness |
| [8 — Consolidation and optional extensions](phase-8-consolidation.md) | Repeatable qualified tower system and completed retirement of obsolete paths | 7 |

## Dependencies and handoffs

1. Phase 0 provides a usable simulation foundation. Phase 1 first provides a minimum scene with both arms/grippers, a table, one block, deployable observations and physical grasp/release. This intermediate result unlocks Phases 2 and 3; full Phase 1 completion additionally requires calibrated cooperative five-block reference execution.
2. Phase 2 provides a provisional observation/action contract. Phase 3 records a small pilot under it; Phase 2 uses that pilot for bounded adaptation and model selection. Their joint final handoff is a selected deployable model, finalized contract and compatible pilot. Prototype work can overlap; mass collection cannot bypass this handoff.
3. Phase 4 delivers the initial training corpus, which starts Phase 5. Later collection uses the policy's measured failures and learning curves. Thus an initial corpus does not depend on a policy that cannot yet be trained.
4. Phase 5 selects a candidate on development data. Phase 6 qualifies that fixed candidate on separate campaigns. Failed qualification returns to development, preserving the result and reserving fresh qualification episodes if the old ones inform learning.
5. Phase 7 measures physical transfer before real adaptation, then adapts and evaluates separately. Phase 8 consolidates the qualified system and completes cleanup.

## What completes the current project

Completion means the five-block task meets all simulation targets and the post-adaptation real-robot target, with Orin inference, a reported pre-adaptation baseline or explicit unavailable outcome, and the repeatable handoff/cleanup from Phase 8. A teacher demonstration, a trained checkpoint or one successful video alone is insufficient.

Walls, bridges, insertion blocks and LEGO are optional follow-on scope. A paper, external benchmark, autonomous scene reset and public release are not prerequisites for completing the current project. Human preparation between episodes is allowed; assistance during an autonomous attempt is counted as intervention.

## Decisions that must be resolved during development

These are measured implementation decisions, not permission to leave the design unspecified indefinitely. The responsible phase records the chosen value or method and its evidence in the canonical topic before the dependent work begins.

| Decision | Responsible phase and deadline |
|---|---|
| Admitted block geometries/scales/colors, instruction vocabulary and feasible workspace | Phase 1, before the initial training corpus |
| Print/contact properties, cooperation scoring and physical acceptance ranges | Phase 1 initial calibration; Phase 7 physical confirmation before its campaigns |
| Model, sensors, action/state meaning, frames, normalization, stop mechanism and timing budget | Phase 2, finalized with the pilot before mass collection |
| Public-source use, supervision roles, case coverage and grouped split definitions | Phase 3 before scaling; Phase 4 makes the initial corpus ready |
| Corpus scale, sampling weights and next collection priorities | Phases 4–5, decided from coverage and development outcomes for each expansion |
| Candidate and deployment configuration; exact campaign composition, timeouts and recovery budget | Phase 5 selection, frozen before Phase 6 qualification; Phase 7 freezes its physical campaigns separately |
| Actual usable resources and environments | Phase 0 establishes the minimum; qualify each additional resource before using it, without requiring all clusters |

An unresolved decision keeps the affected handoff open. Do not invent a physical tolerance or latency limit merely to make the plan look numerically complete.

Prioritize a working task and reusable existing technology. LIBERO, custom temporal interleaving, EO-1 replication, mandatory multi-model integration and publication-oriented ablations are outside the required path. Walls and bridges precede tolerant insertion parts; LEGO remains a later extension.

The [topic pages](../index.md) own design details and operating guidance. Each phase describes only its objective, choices, conceptual work, expected result and advancement conditions. Ordinary earlier plans remain recoverable through [Git history](../history.md).
