# Phase 0 — Cleanup and foundations

## Objective

Turn the current repository into a usable starting point for the new roadmap. Entry is the retained MuJoCo/Qwen baseline; completion does not require a trained model, an assembled tower or a qualified Orin runtime.

## Design choices

Use Isaac Sim/Lab and the shared Alex package. Reusable robot assets belong to Alex; task scenes, calibration, teachers, data and policies belong here. Separate simulation, training and Orin environments; their readiness need not occur simultaneously. The [compute plan](../cluster.md#planned-resource-roles) assigns resources without requiring every cluster to be available.

Remove material only when it has no maintained consumer or protected evidence. Replace MuJoCo, copied robot assets and the custom Qwen path in stages, preserving the usable baseline until the corresponding successor works.

## Conceptual activities

- Classify retained, immediately removable and replacement-dependent capabilities, with a reason for each retained legacy path.
- Establish a usable GPU simulation environment and shared Alex dependency.
- Identify training and onboard environment requirements and unresolved compatibility questions for Phase 2.
- Establish task/asset ownership and a minimal route from observations to robot commands and recorded demonstrations.
- Remove consumer-free duplication; preserve consequential failures and attribution, using Git for ordinary history.

## Expected result

A usable simulation foundation, a clear ownership decision, and a staged retirement list identifying each successor and the capability that permits removal. Training and onboard environment choices may still be provisional.

## Advancement conditions

The simulation foundation is usable on at least one suitable GPU, the shared robot package can be consumed without duplicating its assets, and the retained baseline remains usable. Phase 1 can begin task construction. Unresolved Orin/model choices explicitly pass to Phase 2; they do not keep this phase open. A clean directory tree without a usable simulation foundation is insufficient.
