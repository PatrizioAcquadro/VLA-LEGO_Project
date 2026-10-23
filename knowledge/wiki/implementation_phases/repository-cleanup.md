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

Action-space, robot-state, camera and kinematics operator validators remain maintained.

`models.validate` passed on the actual RTX 4090 and Qwen3.5-4B with four MuJoCo views, a finite `(1, 16, 17)` action chunk and a synthetic gradient-routing probe. Peak allocated memory was 9.24 GiB. All 32 existing real-model GPU tests also passed. The first multimodal attempt exposed a missing `mm_token_type_ids` transport contract with current Transformers; its failed report remains in `outputs/model-native-smoke/`, while the corrected run is `outputs/model-native-smoke-mm/`. The wrapper now supports existing image-only callers and preserves explicit processor metadata.

The old `validate_action_head.py` and `validate_vlm_backbone.py` operator scripts are therefore replaced by `vla-validate-model`. Their native model/gradient/freeze/processor checks remain in the GPU tests, and their useful operational role is covered by the new scene-to-model smoke. The duplicate production mock/automatic CPU fallback is retired. The only executable caller, the Gilbreth validation job, now invokes `models.validate`. Historical reports remain historical; no previous measurement is relabeled as a new result.

Packaging/configuration milestone: all eight model/cluster compositions, Ruff, source formatting and mypy (34 files) pass. The full native suite passes 485 tests, including 32 real-model GPU tests. The sdist-to-wheel probe imports packaged code outside the checkout and validates bundled assets/configuration and all five CLIs. Runtime dependencies are reused for this probe; Docker separately resolves and checks dependencies. The locally built Docker image passes `pip check` and the robot/tower/triangle demo on the RTX 4090 (MuJoCo 3.17.0, Python 3.12). The host uses Python 3.10 and MuJoCo 3.5.0. Apptainer, Gilbreth/A100, multi-GPU execution and hosted GitHub Actions have not been executed.
