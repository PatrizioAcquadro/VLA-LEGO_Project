# Status

**October 31, 2026.** The simulation corpus and bounded learning campaign are delivered. Autonomous cooperative assembly, Orin execution and physical qualification remain open.

| Capability | Current evidence | Limit |
|---|---|---|
| Native Isaac/Alex003 foundation | CUDA physics/rendering, shared WSG32/UMI and ZED assets, physical contact-based teacher demonstrations | Print/contact calibration and physical WSG32 integration are unqualified |
| Corpus | 620 admitted episodes, 500 TRAIN / 60 calibration / 60 confirmation; 30 actual-policy-state corrections | Admission concerns simulation data, not learned behavior or physical transfer |
| Decision learning | 20k classifier updates and complete causal feature preparation | Calibration fails; gated sixty-sequence offline confirmation was not run |
| Motor learning | 20k updates from robot-pretrained SmolVLA | Unassisted control 0/10; full-task extension did not cross the failed gate |
| Final physical simulation comparison | Current 0/20; historical reference 0/20; all forty attempts executed once | No qualified best; confirmation predictions are exposed and models/thresholds remain frozen |
| Software | Isaac collection/readers, replay, learning, inference, evaluation and portable checks maintained | Software tests do not qualify native physics or learned task behavior |

[Data](data.md) owns the 2,528,001-frame corpus, all 122 collection failures and admission boundaries. [Models](models.md) owns training, calibration and physical comparison results. Teacher prefixes and conditional corrections receive no autonomous-from-start credit. Recovery handoffs and route planning are integrated into maintained source and covered by software regression checks. Accepted frozen collection runtimes do not establish whole-task physical success for the current source.

## Phase status

| Phase | Position |
|---|---|
| 0 — Foundations | Native environments and ownership established; legacy software retired from main |
| 1 — Simulation | Cooperative teacher evidence exists; complete current-source and physical calibration questions remain |
| 2 — VLA feasibility | SmolVLA interface exercised; Orin feasibility and final model handoff open |
| 3 — Data/teachers | Alex teacher/pilot and grouped admission implemented; no foreign-robot actions admitted as Alex labels |
| 4 — Collection | Expanded corpus delivered; this does not waive model or calibration gates |
| 5 — Learning | Bounded negative campaign complete; advancement criteria unmet |
| 6–8 — Qualification, transfer, consolidation | Not achieved |

## Open work

- Address demonstrated selection, motor and termination failures through a separately defined development intervention; retain failed control and comparison outcomes.
- Complete source integration where whole-task proof is missing. Verify shared assets, physical WSG32 commands/feedback, print/contact properties and camera calibration.
- Establish the actual Orin software, memory, latency and sustained execution path before claiming onboard readiness.
- Use fresh independent evaluation cases if exposed confirmation cases inform subsequent development.

The campaign stopped at 20k motor updates with 80k of its ceiling unused. There is no automatic collection/training successor; new work starts from an explicit bounded assignment and the [roadmap gates](implementation_phases/index.md).
