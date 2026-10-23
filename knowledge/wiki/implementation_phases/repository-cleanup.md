# Maintained Repository and Lightweight Bricks

## Objective

Audit all consumers and immediately simplify the commissioned software into a coherent base for lightweight-brick manipulation. Work on `PatrizioAcquadro/audiovisual-policy`, based on `main` at `a5205b2`, with local atomic commits only. Do not merge, push, rewrite history, or delete branches/worktrees. The initial working tree was clean.

## Approved implementation

1. Retire the synthetic Transformer, random datasets, trainer/evaluator, their CLIs, configurations, tests and toy-training cluster jobs. Keep VLM/VLA components and planned loss/tracking capabilities with concrete development uses. Full VLA training remains future work.
2. Replace LEGO studs, tubes, baseplates, press-fit, retention/welds and force-assisted insertion with lightweight cuboids of configurable metric dimensions, mass and color. Preserve Alex control, 17-D actions, 52-D states, synchronized cameras and rendering. Implement seeded resets, tower goals and planar triangle layouts, with pose, support and post-release stability evaluation.
3. Replace phase-specific artifact generators with one `vla-demo --scenario {robot,bricks,all} --output-dir <directory> [--seed 42]` command. Prearranged physical scenes are diagnostics, not robot assembly or learned success. Real robot-action demonstrations belong to Phase 2.1.
4. Remove artificial tests and unused wrappers; protect maintained behavior, physics, synchronization, masks and gradients. Align package assets/configuration, optional dependencies, containers and CI. Use the real GPU for supported model/rendering workloads.
5. Create a US-English canonical wiki with index, status, concise log, current technical topics and every original roadmap plan. Recover removed plans from Git. Record historical LEGO work as retired rather than relabeling it as completed lightweight-brick work. Add a future LEGO extension plan.
6. Rewrite Phase 2.1 for physical pick-and-place demonstrations with the actual robot action pipeline; revise annotation, interleaving, training, evaluation and transfer plans. Start with one valid episode and a small pilot; preserve failures and grouped splits. Do not implement large-scale data collection or full training during this cleanup.
7. Remove tracked `CLAUDE.md`; create compact ignored local `AGENTS.md`. Preserve existing evidence and robot asset provenance. Ordinary source/draft history remains in Git.

## Acceptance

Maintained tests, formatting/lint, configuration composition, clean package installation and documentation links pass. Bricks load, settle, reset and support tower/planar-goal evaluation without LEGO connection mechanisms. The demo exports inspectable images/video without overwriting prior output. Model/rendering GPU checks and any unavailable gates are reported separately. Every roadmap phase distinguishes implementation, historical validation, new validation and remaining work.

## State

Implementation in progress. The active checklist is the repository `TODO.md`.

## Operational validation ownership

The full maintained suite passed 448 tests with EGL on the RTX 4090 before consolidating the tools below. The unified demo exported the robot, tower and triangle clips successfully. No physics/controller thresholds were relaxed.

| Retired duplicate | Maintained coverage / operator workflow |
|---|---|
| `generate_phase11_artifacts.py` | `sim.demo` exports short robot motion and gripper commands; Alex/EZGripper tests retain structural, coupling, stability and grasp-scene checks. |
| `validate_mujoco.py` | `test_mujoco.py` retains native load, deterministic stepping, finite state, energy and environment metadata; `validate_sim_smoke.py` remains the operational headless smoke. |
| `validate_assets.py` | `vla-lint-assets` and `test_asset_loader.py` retain path, missing mesh/include, scale and load contracts; clean-install validation loads every packaged scene. |
| `validate_offscreen.py` | `sim.demo`, `test_offscreen.py` and `test_sim_smoke.py` retain RGB/depth/segmentation, frame alignment, deterministic rendering, video and PNG export. |
| `validate_alex_model.py` | `test_alex_model.py`, `test_ezgripper.py`, `validate_kinematics.py` and the robot demo retain model/actuator limits, native stability, PD motion, gripper coupling and reference-FK comparison. |

Action-space, robot-state, camera and kinematics operator validators remain maintained. Native model validators are retained until the replacement real-GPU model smoke is verified; a new command existing on disk is not sufficient evidence to retire them.
