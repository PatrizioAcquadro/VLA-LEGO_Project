# Phase 0 — Cleanup and foundations

## Objective

Maintain a usable Isaac/Alex foundation and an essential repository for the roadmap. Native simulation and separate learning environments are established; the October 31 cleanup retires the superseded software. Completion of this phase does not qualify a trained model, an assembled tower or Orin execution.

## Design choices

Use Isaac Sim/Lab and the shared Alex package. Reusable robot assets belong to Alex; task scenes, calibration, teachers, data and policies belong here. Separate simulation, training and Orin environments; their readiness need not occur simultaneously. The [compute plan](../cluster.md#planned-resource-roles) assigns resources without requiring every cluster to be available.

Remove material only when it has no maintained consumer or protected evidence. MuJoCo, copied Alex V1 assets and Qwen are retired from main; the fixed v1.0.0 baseline preserves their source and attribution. Historical native readers/inference remain where concrete consumers require them.

## Conceptual activities

- Audit code, scripts, data/evidence readers, packaging, tests, CI and documentation; retain only demonstrated consumers and intentional development capabilities.
- Establish a usable GPU simulation environment and shared Alex dependency.
- Identify training and onboard environment requirements and unresolved compatibility questions for Phase 2.
- Establish task/asset ownership and a minimal route from observations to robot commands and recorded demonstrations.
- Remove consumer-free duplication; preserve consequential failures and attribution, using Git for ordinary history.

## Expected result

A usable simulation foundation, clear ownership, a focused installable package and canonical operating guides with explicit evidence/compatibility boundaries. Training and onboard environment choices may still be provisional.

## Advancement conditions

The simulation foundation is usable on at least one suitable GPU, the shared robot package can be consumed without duplicating its assets, and maintained task/data interfaces remain usable. Phase 1 can begin task construction. Unresolved Orin/model choices explicitly pass to Phase 2; they do not keep this phase open. A clean directory tree without a usable simulation foundation is insufficient.
