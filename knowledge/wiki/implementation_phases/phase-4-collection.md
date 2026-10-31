# Phase 4 — Collection and coverage

## Objective

Turn the pilot into a corpus that supports five-block policy learning, then close coverage gaps revealed by that policy. Entry requires the complete Phase 1 task and the joint Phase 2–3 handoff.

The expanded corpus is delivered as simulation evidence. This does not retroactively satisfy unresolved model/onboard or physical calibration gates; [status](../status.md) records those limits.

## Design choices

Use the [60/30/10 proposal](../data.md#expanded-alex003-campaign--october-26-2026) for attempted nominal/recovery/stop cases, not as an expected success rate or automatic loss mixture. Color, shape, scale, instruction and difficulty vary across these categories.

This phase has an initial handoff and subsequent expansions. The initial handoff enables Phase 5; learning curves and policy-generated errors guide later collection. Initial collection therefore does not depend on a policy that cannot yet be trained.

## Conceptual activities

- Cover all stages from selection through final release, both arm roles, target/distractor contrasts and the admitted physical catalog.
- Fill the pilot's named gaps across missed grasp, slip, wrong piece, off-center placement, occlusion, misalignment and interrupted coordination.
- Preserve full context and outcomes for nominal, corrected, failed and non-completable episodes, with distinct supervision roles.
- Expand physically valid seeds and scene variation while preventing color/shape shortcuts.
- Establish training, development and final qualification groups before scaling; reserve the stated generalization axes.
- Measure achieved coverage, usable-data yield and collection cost; redirect later batches toward actual Phase 5 failures.

## Expected result

An initial training-ready corpus, its source/case composition and held-out groups; later versions add targeted coverage without erasing earlier outcomes. [Data readiness](../data.md#further-data-decisions) defines what each handoff must contain.

## Advancement conditions

Phase 5 can start when the initial corpus covers the declared curriculum with usable targets and no unresolved observation/action mapping. Every major error family is either covered or explicitly outside that curriculum pending expansion. Final qualification requires the full declared task/error coverage. There is no universal episode-count completion rule: further collection is justified by measured gaps or learning benefit, not volume alone.
