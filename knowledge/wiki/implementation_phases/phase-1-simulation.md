# Phase 1 — Alex003 and blocks in Isaac

## Objective

Obtain a physically credible assembly environment matching the Purdue robot and printed pieces, progressing from one placement to cooperative tower construction.

## Design choices

Reuse the shared fixed-base Alex003, measured pedestal, WSG32/UMI grippers and head ZED X Mini. Task-specific calibration belongs here; reusable robot assets remain in Alex. Wrist cameras are conditional on a demonstrated occlusion limitation.

Use the [printable block catalog](../bricks.md) and measured table, grasp clearance and contact behavior. Objects move through robot contact under gravity, without locking, teleports or external object assistance during demonstrations.

## Conceptual activities

- Align robot, tool, sensor and workspace geometry with the physical setup and early hardware-interface findings.
- Calibrate printed dimensions, mass, grip/contact behavior and released stability before broad randomization.
- Establish reachable grasp, lift, transport, placement and release; progress to a simple tower.
- Develop genuine stabilize–place cooperation with interchangeable arm roles and realistic distractors.
- Separate deployable observations from privileged teacher/evaluator information using the [future contract](../contracts.md#planned-alex003-contract).

## Expected result

A usable Isaac task with physical manipulation, a defined five-block goal, initial calibrated variation and a cooperative teacher feasibility demonstration.

## Advancement conditions

Pieces can be grasped and released without shortcuts, both arms perform task-relevant roles, and the scene supports meaningful failure and recovery observations. A minimum scene can unblock Phases 2 and 3 early; large-scale collection waits for the full task and interfaces. A prearranged tower is not completion.
