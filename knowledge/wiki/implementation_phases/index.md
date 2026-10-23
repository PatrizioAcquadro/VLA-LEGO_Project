# Implementation plans

Nine canonical files follow website Phases 0–8. Their sections preserve all 22 original subphases, including work already implemented and work still to do. No separate subphase files are needed. These are development specifications; a local ignored `TODO.md` can hold the active implementation checklist.

The [website roadmap](https://websitepresentation.vercel.app/roadmap) and historical repository plans are the sources. The original January 16–April 4, 2026 schedule is historical. There is no new delivery calendar or assumed 8×A100 allocation. Status comes from actual source and [evidence](../validation.md), not the passage of scheduled dates.

| Phase file | Included subphases | Current state |
|---|---|---|
| [0 — Foundation](phase-0-foundation.md) | 0.1 Environment, 0.2 MuJoCo, 0.3 Literature | Local runtime established; cluster qualification and literature synthesis incomplete |
| [1 — Simulation](phase-1-simulation.md) | 1.1 Alex robot, 1.2 Lightweight bricks | Maintained components and local diagnostics; physical calibration/demo collection pending |
| [2 — Data](phase-2-data.md) | 2.1 Trajectories, 2.2 Annotation, 2.3 Interleaving | Planned for lightweight bricks; no qualified dataset |
| [3 — Models](phase-3-models.md) | 3.1 Qwen backbone, 3.2 Action head | Implemented components, local native GPU validation; integration/scale limits remain |
| [4 — Training infrastructure](phase-4-training-infrastructure.md) | 4.1 Losses, 4.2 Distributed training | Standalone loss components partial; real training/distributed path pending |
| [5 — Training](phase-5-training.md) | 5.1 Initial run, 5.2 Continued training | Planned, blocked on real data and trainer |
| [6 — Evaluation](phase-6-evaluation.md) | 6.1 Brick tasks, 6.2 LIBERO, 6.3 Ablations | Task evaluator exists; learned benchmarks and all experiments pending |
| [7 — Adaptation](phase-7-adaptation.md) | 7.1 Task fine-tuning, 7.2 Few-shot | Planned |
| [8 — Transfer and handoff](phase-8-transfer-handoff.md) | 8.1 Sim-to-real, 8.2 Alex mapping, 8.3 Handoff | Documentation baseline delivered; physical transfer and research artifact handoff pending |

## Scope reconciliation

- Phase 1.2 now models lightweight cuboids and gravity-supported structures. The old studs/tubes, insertion, baseplate and retention plans are retired, with important negative results preserved in [history](../history.md).
- Phase 2.1 collects actual robot pick-and-place trajectories for these bricks. The old 10K episodes/large storage estimate is a possible later scale, not a prerequisite or an authorized run.
- Qwen3.5-4B replaces the website's Qwen2.5 reference. Current 17-D actions, 52-D state, four views and 16-step chunks are canonical [contracts](../contracts.md).
- Phase 4.2 must train the actual VLA on real episodes. Toy Transformer/DeepSpeed jobs and their synthetic tests were removed; their results do not transfer automatically.
- Original targets (>50% minimum, >70% target, >85% stretch task success) remain historical aspirations. Define a justified brick-specific benchmark, denominators and stopping criteria before claiming any threshold. LEGO connection-quality targets are inapplicable to gravity-based bricks.
- The [future real-LEGO extension](phase-8-transfer-handoff.md#future-extension-real-lego) is intentionally beyond the current baseline. Bimanual coordination remains a later goal, not a requirement for the first one-arm collection pilot.

## Dependency order

Foundation and simulation → one real demonstration → pilot dataset → grounded annotations/interleaving → integrated losses and single-GPU training/resume → qualified distributed training → learned task evaluation → adaptation and physical-transfer studies.

Model component development can proceed alongside data work. Passing a component forward or producing attractive diagnostic videos does not satisfy the downstream dataset/training/task gates. Refer to the separate [repository cleanup specification](../../repository-cleanup.md) for this audit's completed implementation scope.
