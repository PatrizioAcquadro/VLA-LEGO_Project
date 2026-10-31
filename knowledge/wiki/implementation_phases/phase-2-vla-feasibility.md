# Phase 2 — VLA and onboard feasibility

## Objective

Select one robot-pretrained VLA, its Alex003 interface and a demonstrated Orin execution path before mass collection. Begin with the Phase 1 minimum scene; use the Phase 3 pilot to finish selection.

## Design choices

GR00T N1.7 is the leading candidate; SmolVLA is the compact alternative. Compare π0.5 and LingBot-VLA 2.0 technically, without requiring all four integrations. Follow the [selection rule](../models.md#planned-pretrained-vla-selection): qualify the leading candidate first and exercise alternatives when a limitation justifies it.

All online learned processing used by the deployed policy runs on Orin alongside sensing and communication. Existing low-level control remains on the robot controller. The documented host-stack mismatch and WSG32 hardware interface are early feasibility questions, not deferred transfer details.

## Conceptual activities

- Give Phase 3 a provisional observation/action/timing contract before pilot collection; define actual frames, units, available signals and command meanings.
- Establish compatibility with the Orin carrier, ZED and robot services, and confirm WSG32 commands, feedback and stopping through controlled hardware readiness work.
- Run the selected model on actual Orin with representative inputs and concurrent services; establish a measured responsiveness budget.
- Use the Phase 3 pilot for bounded adaptation and simulated closed-loop behavior, including piece selection and coordinated motion.
- Select model, adaptation scope, precision, observations, action representation, execution horizon and deployable stop behavior.
- Reconcile any contract change with the pilot before accepting the joint handoff.

## Expected result

One selected model/configuration, a finalized embodiment contract and measured onboard resource/timing evidence. The pilot is usable under that contract. This is feasibility, not final tower qualification.

## Advancement conditions

Actual Orin execution meets the task-derived timing and memory budget, required robot services coexist, WSG32 integration is workable, and pilot adaptation produces meaningful task-conditioned behavior. No external VLA server substitutes for this evidence. If no candidate is feasible, report the blocker and continue bounded feasibility work; do not collect a massive corpus around an unqualified choice.
