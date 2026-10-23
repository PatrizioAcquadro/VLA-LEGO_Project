# Phase 2 — Data pipeline

Scope: original 2.1, 2.2 and 2.3. Build real lightweight-brick trajectories and grounded vision/text/action samples. **State: planned.** No maintained collector, annotator, interleaved dataset or full training-ready corpus exists. The former random dataset is retired.

Prerequisites: Phase 1 physical simulation/contracts; Phase 3 processor for annotation and token budgeting. The canonical [data specification](../data.md) defines modality, truth separation, alignment, splits and failure retention.

## 2.1 Robot trajectory generation

### Implementation sequence

1. **2.1.1 Task-space teacher:** implement the smallest controller that commands the existing 17-D action interface through `SimRunner`. Start with one arm and a graspable cuboid. Use named tool frames and bounded IK/joint increments, with joint/workspace/collision checks. Ground truth may guide this external teacher but must never masquerade as a learned observation.
2. **2.1.2 Grasp and waypoints:** plan approach, grasp, lift, transport, lower, release and retract. Check real gripper/object contact and object motion. Do not use object teleports, `xfrc_applied`, welds or force-assisted placement. Stop explicitly on unreachable approach, collision, slip or time limit. Add bimanual handoff/coordination only after one-arm feasibility.
3. **2.1.3 Recording:** write synchronized four-view observations, actual applied 17-D actions, 52-D state and separate task truth. Preserve `obs_t/action_t/obs_t+1`, seed/catalog/goal, timestamps, camera metadata and every termination reason. Choose storage from measured episode size and bounded IO, rather than the old 550-GB estimate.
4. **2.1.4 Single-episode pipeline:** generate one complete native episode, reload it in a separate process, verify alignment and replay through actual commands. Inspect video and support/release outcome. A reset-to-goal clip or a nonmoving gripper is a failed collection gate.
5. **2.1.5 Bounded pilot, then scale:** preregister a small set of seeds, dimensions, shapes, attempts and timeouts. Measure success, failure distribution, storage and throughput first. Add parallel workers only after independent seeds/output ownership and replay are correct. The original 10K objective is a possible scale target, not immediate authorization or a quality metric.
6. **2.1.6 Failure/recovery:** retain failed grasps, slips, collisions, wrong placements and timeouts with the full attempt denominator. Recovery episodes need an explicit starting failure, controller actions and measured recovery outcome. Filtering must label exclusions and counts instead of deleting inconvenient trials.

### Completion gates

- One real pick/lift/transport/place/release trajectory reproduces its outcome from recorded commands without privileged corrective execution.
- A fixed pilot covers nominal and perturbed cases, preserves failures and passes observation/control synchronization checks.
- Episode-group splits are defined before windowing. No seed/layout/geometry family leaks between splits; normalization uses training data only.
- Scale is approved only after a measured throughput/storage estimate and explicit success/failure acceptance criteria. No trajectory count alone establishes policy usefulness.

Deliver replayable episode files, catalog/config, pilot report, denominator and failure taxonomy to 2.2/2.3. None of this is implemented by the current diagnostic demo.

## 2.2 VLM-based annotation

### Implementation sequence

1. **2.2.0 Schema:** define episode/task instruction, timestamped action-stage narration, spatial/temporal QA, observed outcome and uncertainty. Distinguish human labels, VLM proposals and simulator/evaluator truth.
2. **2.2.1 Infrastructure:** use a local GPU VLM and reusable native processor first. Select representative frames/windows and use bounded batches/checkpoints so failures can be resumed without relabeling finished episodes. External API annotation requires a deliberate data-sharing decision; no automatic upload path is part of this plan.
3. **2.2.2 Descriptions/narration:** describe variable-size bricks, colors, intended structure and actual controller stages. Ground phrases in frames and timestamps. Replace all old insertion/baseplate vocabulary with pick, place, stack, support and release.
4. **2.2.3 Reasoning QA:** generate questions that can be answered from the allowed observation context: what moved, what supports a piece, what failed, or what step comes next. Separate retrospective future-aware labels from action-time conditioning. Never synthesize a successful assembly that the evaluator rejected.
5. **2.2.4 Quality:** manually review a preregistered sample across tasks, sizes and success/failure types. Measure schema validity, grounding, temporal alignment, contradiction and uncertainty rates. Fix systematic hallucinations before scaling; preserve rejected annotations with reasons.

### Completion gate and handoff

One annotated pilot must validate against its source episodes and pass defined manual/automated quality criteria. Keep annotation provenance and outcome disagreement visible. Deliver the schema, approved annotations, rejected counts and observation boundary to 2.3. Model-generated fluent text alone is not grounded training data.

## 2.3 Interleaved data construction

### Implementation sequence

1. **2.3.0 Sequence contract:** define explicit text/vision/state/action segments, token types, attention visibility, padding and action alignment. Preserve native image metadata and count expanded vision tokens. Reject sequences that exceed the complete context budget; do not silently drop a target or its observation.
2. **2.3.1 VLA sequences:** construct actual observation-conditioned action chunks, mask incomplete tails, shift text targets on the original sequence and set non-text labels to `-100`. Test that future images/actions/outcomes cannot condition earlier action predictions.
3. **2.3.2 Temporal reasoning subset:** create planning, order and verification examples tied to real episode stages, including failure and recovery. Define which frames are available to each answer.
4. **2.3.3 Spatial reasoning subset:** create visible grounding and relative-position/support questions across varied brick dimensions/layouts. Privileged coordinates used to validate labels must remain outside undeclared policy inputs.
5. **2.3.4 Chat subset:** retain only a justified language/chat mixture with an explicit source/license and loss mask. Chat-only samples contribute no fabricated action targets. Avoid a separate generic data pipeline without a measured need.
6. **2.3.5 Assembly/validation:** implement grouped splits, bounded loading, batch collation and deterministic sampling. Report sample counts, mixture ratios, truncation/rejection counts and normalization statistics. Verify one real minibatch through the native model and loss path before sharding at scale.

### Completion gate and handoff

A held-out episode must produce correct expanded modalities, masks and aligned actions without future-information leakage. One minibatch must give finite correctly routed losses/gradients; masked/padded values must not change the result. Temporal layouts must match a model implementation that actually supports them—current grouped embedding assembly is not evidence of arbitrary interleaving. Deliver the loader/collator contract and immutable split definition to Phases 3/4, with full source counts and exclusions.

## Source trace

All three subphases and their detailed substeps come from the website and baseline `docs/roadmap/phase2.1.md`, `phase2.2.md`, `phase2.3.md`. The current plan replaces LEGO insertion data with real lightweight-brick manipulation and stages scale after a validated pilot.
