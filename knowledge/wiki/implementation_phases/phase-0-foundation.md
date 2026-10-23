# Phase 0 — Cleanup and foundations

## Objective

Establish an essential, maintainable base for the Alex003 assembly roadmap without discarding useful behavior or consequential evidence.

## Design choices

Use Isaac Sim/Lab as the future simulation stack and the shared Alex package as the robot asset owner. Keep task logic, teachers, data and policies in this project. Separate simulation, training and Orin environments according to their actual compatibility needs; use the [resource roles](../cluster.md#planned-resource-roles).

Cleanup follows consumers and replacement readiness. Remove unused duplication and abandoned plans first. Retire MuJoCo, copied robot assets and the custom Qwen model only when their maintained roles have operational replacements. Avoid maintaining two simulators or two robot asset authorities indefinitely.

## Conceptual activities

- Identify what is used, intentionally retained, replaced or obsolete; retain provenance and meaningful failed/skipped outcomes.
- Establish ownership and compatibility boundaries across this project, Alex and upstream learning tools.
- Define the minimal path from scene and observations to teacher data, pretrained policy and onboard execution.
- Keep migration stages independently usable; recover ordinary implementation history through Git.

## Expected result

A focused development base and a staged retirement decision for each legacy subsystem, with one canonical source per topic.

## Advancement conditions

The minimum Isaac/Alex environment has a viable dependency path, the retained baseline remains usable, and obsolete material has no active consumer or protected evidence. Replacement-dependent removals remain open until their successors work. This phase authorizes no bulk deletion merely because code is old.
