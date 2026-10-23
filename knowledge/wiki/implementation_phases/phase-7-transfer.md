# Phase 7 — Physical transfer

## Objective

Demonstrate the task on Purdue Alex003 with inference entirely on Orin, and quantify how targeted real experience improves transfer.

## Design choices

Match the qualified sensor/control contract and actual printed pieces. Retain the robot's established low-level controller and stopping mechanisms. Distinguish hard safety constraints from empirical contact and stability tolerances.

Measure transfer before real task adaptation, after necessary hardware calibration and controlled motion readiness. Then use real demonstrations and corrective interventions; do not assume simulation alone will close the gap.

## Conceptual activities

- Confirm physical frames, gripper feedback, timing and sensor availability under the selected onboard runtime.
- Progress through controlled manipulation readiness to a bounded zero-adaptation campaign; preserve all outcomes and declared calibration.
- Diagnose visual, geometric, contact and timing failures against simulation.
- Collect targeted real demonstrations and corrections, keeping adaptation episodes separate from final evaluation.
- Adapt the policy and, when justified, simulation parameters; distinguish these interventions when attributing improvement.
- Evaluate the adapted system across at least two sessions with the [fixed real-robot target](../validation.md#assembly-qualification-targets).

## Expected result

Measured zero-adaptation and post-adaptation results, a usable onboard policy, and a clear description of remaining physical limitations.

## Advancement conditions

Meet the real-robot criterion with no workstation/cloud VLA inference or undisclosed teacher assistance. Report interventions, stops, regressions and failures. If performance remains insufficient, return to targeted collection or calibration rather than weakening the acceptance conditions.
