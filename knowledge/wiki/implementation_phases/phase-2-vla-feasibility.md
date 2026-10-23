# Phase 2 — VLA and onboard feasibility

## Objective

Select a pretrained robotic VLA and a feasible onboard execution path before committing to large-scale collection or training.

## Design choices

Use GR00T N1.7 as the leading candidate and SmolVLA as the compact alternative. Compare π0.5 and LingBot-VLA 2.0 technically; integration of every model is not required. Selection follows task behavior, full observation-to-action latency, memory and real Orin compatibility, not public benchmark ranking.

All final VLA inference runs on the robot's Orin alongside sensing and communication. The [model page](../models.md) owns software constraints, including the GR00T deployment stack mismatch with the Alex guide. A container does not resolve incompatible host drivers or board support.

## Conceptual activities

- Establish the Orin, carrier, ZED and robot-service compatibility path before adopting an upstream runtime.
- Confirm actual WSG32 command, feedback and stopping behavior independently of simulated asset readiness.
- Resolve deployable observations, coordinated end-effector/gripper actions and timing in the [new contract](../contracts.md#planned-alex003-contract).
- Compare the leading candidate and compact fallback on a bounded pilot, including closed-loop behavior after initial adaptation and resource use with concurrent robot services.
- Choose adaptation scope and chunk execution/replanning from measured responsiveness.

## Expected result

One selected model, an explicit embodiment interface and a demonstrated onboard execution path with remaining limitations stated.

## Advancement conditions

The selected path fits Orin memory and task responsiveness without external VLA inference, coexists with required services, and has a workable WSG32 hardware interface. If GR00T cannot meet this boundary, use the qualified compact alternative. Mass collection waits for this decision and the Phase 3 pilot.
