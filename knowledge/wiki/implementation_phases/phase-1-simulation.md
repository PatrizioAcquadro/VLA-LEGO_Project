# Phase 1 — Alex003 and blocks in Isaac

## Objective

Create the hardware-faithful task and demonstrate that Alex003 can physically assemble the requested tower under external reference control. This is task feasibility, not learned-policy success.

## Design choices

Use the shared fixed-base Alex003, measured pedestal, WSG32/UMI grippers and head ZED X Mini. Task calibration belongs here. Wrist cameras remain conditional on a demonstrated occlusion limitation.

Use the [block/task definition](../bricks.md), including the initial print sample and genuine stabilize–place cooperation. All manipulation uses robot contact under gravity. A reference controller may use simulation truth; the eventual policy may not.

## Conceptual activities

- Align robot, table, tool and camera frames with the physical setup.
- Establish a minimum scene: both arms/grippers, table, one graspable block, head images, real-equivalent proprioception and bounded robot commands.
- Demonstrate grasp, lift, transport and release in that minimum scene; hand it to Phases 2 and 3 while full task development continues.
- Measure initial prints and fit dimensions, mass, grip/contact behavior and released stability; clearly label provisional assumptions until measurements exist.
- Expand to five requested pieces plus distractors, physically feasible instructions and cooperative placements with either arm in either role.
- Establish an evaluator for identity/order, cooperation, support, release and dwell, with calibration-derived acceptance ranges.

## Expected result

A complete task environment, measured initial print catalog and reference execution of a five-block cooperative tower. The reference behavior supplies seed skills for the data-ready teachers in Phase 3.

## Advancement conditions

A reference execution completes the full task from loose pieces, with the cooperation defined in the task contract and no object assistance. Print/contact assumptions and evaluator ranges are resolved for the admitted initial curriculum. The minimum scene unlocks parallel work early; it is not completion of Phase 1. Wider variation and systematic recovery generation belong to Phases 3 and 4.
