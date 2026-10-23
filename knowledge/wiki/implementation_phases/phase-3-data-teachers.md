# Phase 3 — Data sources and teachers

## Objective

Produce a small, usable Alex corpus and a reliable way to generate and correct demonstrations. Begin with the Phase 1 minimum scene and Phase 2's provisional contract; complete the handoff jointly with Phase 2.

## Design choices

Reuse pretrained robotic knowledge and selectively admit public data. The [data strategy](../data.md) owns source terms, overlap, action mapping, case distribution and supervision. Every reviewed source receives a decision: additional training data, checkpoint prior only, deferred or excluded, with a reason.

Use geometric motion planning, feedback and bimanual coordination for the initial teacher. Human demonstrations/corrections cover gaps. Larger models may assist offline, but physical execution establishes validity.

## Conceptual activities

- Select and inspect relevant public subsets, distinguishing actual accessible episodes from papers, assets or checkpoint priors.
- Turn Phase 1 reference skills into recorded full-tower demonstrations, with correct language, synchronization and separate privileged channels.
- Add corrective continuations for the initial error curriculum and appropriate behavior in non-completable cases.
- Establish how seed demonstrations can be expanded while preserving dependencies between hands; expansion need not already be large-scale.
- Build the pilot across nominal, recovery and stop categories, keeping all attempts and measuring useful-data yield.
- Separate generation families before windowing and provide the pilot to Phase 2 for bounded adaptation.

## Expected result

Source-admission decisions, executable nominal/recovery teachers and a pilot usable by the selected model. Include successful full-tower seeds, verified corrective continuations for admitted error families, stop examples, achieved case counts and explicitly uncovered families.

## Advancement conditions

The selected model can consume the pilot under the finalized contract, including positive/corrective targets and appropriate masks for other segments. Demonstrations use actual contacts; corrections have observed outcomes. Initial gaps have named Phase 4 coverage requirements. Contract changes require a justified remapping or recollection, never fabricated missing signals. Neither a source bibliography nor successful teacher videos alone completes this phase.
