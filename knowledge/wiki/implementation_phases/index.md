# Development roadmap

Build an **ordered five-block tower with Alex003**, selecting the requested colors and sizes among distractors and coordinating both arms. Establish the capability in Isaac Sim/Lab, then on the Purdue robot with VLA inference entirely on its Jetson AGX Orin. The canonical [task](../bricks.md), [interfaces](../contracts.md), [data strategy](../data.md) and [qualification criteria](../validation.md#assembly-qualification-targets) define the shared requirements.

This roadmap replaces the original 22 subphases. **All nine phases below are planned.** Existing MuJoCo/Qwen components are a migration baseline, not completion of these phases; [status](../status.md) distinguishes the two. There is no fixed calendar or guaranteed cluster allocation.

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
| [8 — Consolidation and extensions](phase-8-consolidation.md) | Repeatable system, completed retirement of obsolete paths and justified extensions | 7 |

Phases 2 and 3 can progress together once the minimum Isaac environment exists. Large-scale collection waits for both: the model/hardware contract and a useful teacher pilot. Phases 4 and 5 then form an iterative collection–learning loop; final qualification uses separate, fixed campaigns.

Prioritize a working task and reusable existing technology. LIBERO, custom temporal interleaving, EO-1 replication, mandatory multi-model integration and publication-oriented ablations are outside the required path. Walls and bridges precede tolerant insertion parts; LEGO remains a later extension.

The [topic pages](../index.md) own design details and operating guidance. Each phase describes only its objective, choices, conceptual work, expected result and advancement conditions. Ordinary earlier plans remain recoverable through [Git history](../history.md).
