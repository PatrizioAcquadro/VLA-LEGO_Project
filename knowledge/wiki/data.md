# Data specification — planned

There is currently no maintained data collector, annotation engine, dataset reader or trained policy dataset in this repository. The former random-token dataset was removed. This page defines the essential handoff for [Phases 2.1–2.3](implementation_phases/index.md); implement one real episode before scaling storage or parallelism.

## Episode boundary

A demonstration must move bricks through `SimRunner` and actual gripper contact. A task-space teacher may use simulator truth to choose waypoints, but its privileged inputs must be recorded separately from deployable policy observations. Teleportation, external object forces, retention welds and reset-to-goal layouts cannot count as robot demonstrations.

| Group | Required content |
|---|---|
| Identity | Episode ID, seed, task/goal, catalog dimensions and mass, source revision/config |
| Observations | Ordered four-view RGB, timestamps/control indices and declared raw or normalized 52-D state |
| Applied control | Actual normalized 17-D command at each tick; chunk tails explicitly masked |
| Diagnostics | Separate object truth, support/contact summaries, controller stage and failure/termination reason |
| Outcome | Per-placement errors, support and post-release dwell result; episode success and all attempts |
| Language | Instruction and timestamped verified annotations with source/uncertainty; never invented success |

Store the synchronization `observation_t → action_t → observation_t+1` explicitly. Camera metadata must include moving wrist extrinsics. Persist essential evidence using bounded writes; choose HDF5/chunked image storage after measuring one episode's size, IO and replay costs. Do not reserve the old 550-GB estimate or 10K run as an implementation requirement.

## Splits and leakage

Split by episode groups and task variation before window/chunk construction. Group related reset seeds, geometry, layouts and demonstration families so near-identical windows cannot cross train/validation/test. Fit normalization on training data only. Keep held-out dimensions, arrangement shapes, instruction paraphrases and visual perturbations distinct when making generalization claims.

Policy inputs may contain declared sensor data, robot state and instructions available at decision time. Teacher coordinates, future frames, outcome labels and evaluator contact truth must not leak into action conditioning. Annotation can use later frames for retrospective explanation only if those targets and their conditioning boundary are explicit.

## Collection and annotation quality

First obtain one complete pick/lift/transport/place/release episode. Then preregister a small pilot's seeds, task count, timeouts, perturbations and acceptance bounds. Preserve every attempted episode, including unreachable grasps, slips, collisions, timeouts, unsupported placements and failed recovery. Filtering creates a labeled subset; it never erases the denominator or turns a failed run into a success.

Annotations should describe visible pieces, action stages, spatial relations and observed outcomes. Use a local GPU VLM only after a schema and a small manually checked sample exist. Do not automatically send project images to external APIs. Separate perception uncertainty from simulation labels, and stop on systematic hallucination before a large annotation batch.

## Interleaving handoff

Build explicit text/vision/state/action segments with token types and causal boundaries. Text labels ignore non-text positions with `-100`; action masks exclude padded tails. Temporal/spatial reasoning and chat-only samples may be mixed only with explicit task tags and loss masks. The loader must report truncation, invalid samples, shard overlap and sample counts. Validate native processor expansion, future-information exclusion and one minibatch's gradients before implementing distributed sharding.
