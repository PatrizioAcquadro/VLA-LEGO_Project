# Phase 7 — Physical transfer

## Objective

Measure the simulation-qualified policy on Purdue Alex003, then obtain the required real-task performance through targeted adaptation with inference entirely on Orin.

## Design choices

Entry requires Phase 6 and controlled hardware readiness from Phase 2. Match the admitted printed catalog, sensor/control interface and onboard configuration. Preserve the robot's low-level control and stopping mechanisms.

Zero-adaptation means no learning from real task demonstrations or outcome-driven retuning before the baseline. Required geometric/sensor/controller calibration is allowed and recorded. The [evaluation protocol](../validation.md#physical-baseline-and-evaluation) defines this boundary and physical scoring.

## Conceptual activities

- Complete controlled manipulation readiness and document any calibration before freezing the baseline.
- Measure the fixed zero-adaptation policy on a predefined bounded set; preserve failures and safety interventions.
- Diagnose visual, geometric, contact and timing gaps, then collect targeted real demonstrations/corrections on separate episodes.
- Adapt the policy and, when justified, simulation parameters; record policy learning and calibration changes separately.
- Select the adapted candidate on development episodes; evaluate it unchanged on 50 held-out episodes across at least two sessions.
- Score physical identity/order, cooperation, release and stability using recorded observations independently of policy control.

## Expected result

Separate baseline and adapted results, the real adaptation data and recipe, and a usable onboard policy with measured physical limits. Record regression or lack of improvement as clearly as gains.

## Advancement conditions

The fixed adapted system meets the canonical real-task target across the required sessions with no external VLA server or hidden intervention. The baseline remains reported even if poor; it is not pooled with adapted results. If it cannot be run safely, record that limitation rather than inventing a transfer score. Insufficient final performance returns to targeted collection/calibration; Phase 7 is not complete until its target is demonstrated.
