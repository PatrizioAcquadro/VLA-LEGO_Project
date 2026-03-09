# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLA-LEGO is a Vision-Language-Action system for bimanual robotic LEGO assembly. It replicates and extends the EO-1 model architecture (Qwen 2.5 VL backbone with autoregressive decoding + flow matching) for coordinated two-arm manipulation on the IHMC Alex humanoid robot.

## Commands

### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
pre-commit install
```

### Training
```bash
python -m train.trainer trainer=debug cluster=local                            # debug (100 steps, small batch)
python -m train.trainer cluster=local                                          # full, base model
python -m train.trainer model=large cluster=gilbreth                           # full, large model, HPC
python -m train.trainer trainer.optimizer.lr=1e-5 trainer.training.batch_size_per_device=16  # override any value
```

### Simulation
```bash
python scripts/validate_mujoco.py                                              # import, load, determinism, metadata
vla-viewer sim/assets/scenes/test_scene.xml                                    # interactive viewer
vla-viewer sim/assets/scenes/test_scene.xml --show-contacts --show-joints      # with debug overlays
python scripts/validate_offscreen.py                                           # headless rendering: video + frames
python scripts/validate_sim_smoke.py                                           # physics + rendering smoke tests
python scripts/validate_assets.py                                              # asset layout + linter + load test
python scripts/validate_alex_model.py                                          # Alex model validation + video artifacts
python scripts/validate_kinematics.py                                          # FK symmetry, workspace, joint axis validation
ALEX_SDK_PATH=../ihmc-alex-sdk python scripts/validate_kinematics.py           # with reference FK comparison
python scripts/validate_action_space.py                                        # action space contract + stability tests
python scripts/validate_robot_state.py                                         # robot state contract + consistency tests
python scripts/validate_cameras.py                                             # multi-view camera contract + sync tests
vla-lint-assets                                                                # lint all MJCF files under sim/assets/
vla-gen-bricks                                                                 # generate LEGO brick MJCF assets
python scripts/validate_lego_bricks.py                                         # LEGO brick validation (geometry, mass, contacts)
python scripts/validate_lego_contacts.py                                       # LEGO contact physics validation (insertion, retention)
python scripts/validate_lego_contacts.py --mode both                           # physics + spec-proxy validation
python scripts/validate_lego_contacts.py --mode spec_proxy                     # spec-proxy only
python scripts/sweep_lego_physics.py                                           # physics parameter sweep (tube radius x friction)
python scripts/validate_lego_baseplate.py                                      # LEGO baseplate validation (geometry, contacts, workspace)
vla-viewer sim/assets/scenes/alex_lego_workspace.xml --show-contacts           # Alex + table + baseplate workspace
python scripts/validate_episode_manager.py                                     # Episode manager validation (spawning, settle, curriculum)
python scripts/validate_episode_manager.py --n-stress 100                      # stress test with 100 resets
python scripts/validate_lego_task.py                                           # MVP-3 task validation (goal gen, scripted assembly, metrics)
python scripts/validate_lego_task.py --n-stress 20                             # stress test with 20 assemblies
```

### Testing
```bash
pytest                              # all tests
pytest tests/test_models.py -v      # single file
pytest --cov=. --cov-report=html    # with coverage
pytest -m "not slow and not gpu"    # skip slow/GPU tests
pytest tests/test_mujoco.py -v      # MuJoCo sim tests
pytest -m smoke -v                  # sim smoke tests
pytest tests/test_asset_loader.py -v  # asset loader + linter tests
pytest tests/test_alex_model.py -v   # Alex robot model tests
pytest tests/test_ezgripper.py -v   # EZGripper integration tests
pytest tests/test_action_space.py -v # action space + sim runner tests
pytest tests/test_robot_state.py -v  # robot state contract tests
pytest tests/test_cameras.py -v      # multi-view camera contract tests
pytest tests/test_lego_bricks.py -v  # LEGO brick generation + metadata tests
pytest tests/test_lego_contacts.py -v  # LEGO contact physics tests
pytest tests/test_lego_contacts.py -k "Hybrid or ConnectionManager" -v  # hybrid retention tests only
pytest tests/test_lego_baseplate.py -v  # LEGO baseplate + workspace tests
pytest tests/test_episode_manager.py -v  # Episode manager: spawn, settle, curriculum
pytest tests/test_episode_manager.py -v -m "not slow"  # episode manager (fast subset)
pytest tests/test_lego_task.py -v        # MVP-3 task: goal gen, scripted assembly, metrics
pytest tests/test_lego_task.py -v -m "not slow"  # task tests (fast subset)
```

### Code Quality
```bash
black .                                                    # format
isort .                                                    # sort imports
ruff check .                                               # lint
mypy sim models train eval tracking --ignore-missing-imports  # type check
pre-commit run --all-files                                 # all checks
python scripts/validate_configs.py                         # validate Hydra configs
```

### HPC (Gilbreth)
```bash
sbatch infra/gilbreth/job_templates/01_smoke_1gpu.sh       # single GPU test
sbatch infra/gilbreth/job_templates/04_smoke_8gpu_deepspeed.sh  # multi-node
sbatch infra/gilbreth/job_templates/06_smoke_sim_headless.sh   # headless sim smoke test
```

## Architecture

### Module Structure
- **configs/** - Hydra configuration hierarchy (model, trainer, data, cluster, logging)
- **models/** - TransformerModel with MSE loss for state prediction
- **train/** - Trainer class handling distributed training (DDP/DeepSpeed), checkpointing, validation
- **data/** - DummyDataset (testing) and SimulationDataset (real data, stub)
- **sim/** - MuJoCo simulation: `mujoco_env.py` (load/step/determinism), `env_meta.py` (metadata), `viewer.py` (interactive debug viewer), `offscreen.py` (headless rendering + video export), `asset_loader.py` (single entrypoint: `load_scene()`), `asset_linter.py` (MJCF validation), `control.py` (safety clamps, rate limiting, velocity/effort limits — `AlexController`), `end_effector.py` (`EndEffectorInterface` ABC + `EZGripperInterface` — gripper command abstraction), `action_space.py` (frozen 17-D action contract — `AlexActionSpace`, normalization, delta-q mapping), `robot_state.py` (frozen 52-D state contract — `AlexRobotState`, `RobotState` dataclass, normalization), `sim_runner.py` (fixed-rate control loop — `SimRunner`, 20 Hz with 25 substeps, returns `RobotState`), `camera.py` (frozen 2-view contract — `MultiViewRenderer`, synchronized dual-camera capture, `CameraMetadata`), `assets/` (MJCF scenes + robot models)
- **sim/lego/** - LEGO brick generation + contact physics + episode management + task (Phase 1.2): `constants.py` (geometry — `BrickType`, `BRICK_TYPES`, `BaseplateType`, `BASEPLATE_TYPES`, `STUD_PITCH`, contact isolation contype/conaffinity), `mass.py` (ABS density mass computation — bricks + baseplates), `connector.py` (`ConnectorPoint`, `BrickConnectors`, `BaseplateConnectors`, stud/tube metadata with stable IDs), `brick_generator.py` (procedural MJCF generation — `generate_brick_mjcf()`, `generate_brick_body_xml()`, `add_lego_defaults()`, `write_brick_assets()`), `baseplate_generator.py` (procedural baseplate MJCF — `generate_baseplate_mjcf()`, `generate_baseplate_body_xml()`, `write_baseplate_assets()`), `contact_scene.py` (scene builder — `generate_insertion_scene()`, `generate_baseplate_insertion_scene()`, `generate_workspace_scene()`, `generate_episode_scene()`, `load_insertion_scene()`, `load_baseplate_insertion_scene()`, `check_stud_engagement()`, `setup_connection_manager()`), `contact_utils.py` (measurement — `run_insertion()`, `apply_force_ramp()`, `InsertionResult`; `base_surface_height` param for baseplate scenes), `connection_manager.py` (hybrid retention — `ConnectionManager`, `BrickPairState`, runtime weld activation, `reset()`), `episode_manager.py` (Phase 1.2.5 — `EpisodeManager`, `EpisodeInfo`, `SpawnPose`, `ResetMetrics`, `LEVEL_SINGLE_BRICK/CONNECTION/MULTI_STEP`), `task.py` (Phase 1.2.6 — `PlacementTarget`, `AssemblyGoal`, `AssemblyResult`, `PlacementResult`, `compute_target_position()`, `generate_assembly_goal()`, `check_placement()`, `check_stability()`, `evaluate_assembly()`), `scripted_assembly.py` (Phase 1.2.6 — `ScriptedAssembler`, `AssemblyStepLog`, force-based scripted assembly executor), `cli.py` (`vla-gen-bricks` entry point — generates bricks + baseplates)
- **eval/** - Evaluator class (entry point stub)
- **tracking/** - W&B experiment tracking with distributed-safe logging, GPU monitoring, throughput metrics, run naming
- **infra/gilbreth/** - SLURM job templates and HPC setup scripts

### Configuration
All hyperparameters flow through Hydra configs in `configs/`. Key config groups:
- `model`: base (512 hidden, 6 layers, 8 heads, GELU, ~25M params) or large (2048 hidden, 24 layers, 32 heads, SwiGLU, flow matching, ~1.2B params)
- `trainer`: default or debug (100 steps, fp32)
- `cluster`: local or gilbreth (DeepSpeed, multi-GPU)

**Configuration-first principle**: Never hardcode values in code. Use `cfg.trainer.optimizer.lr` style access.

### Dependency Groups (`pyproject.toml`)
- `ci` - linters + pytest (CI only)
- `train` - wandb, accelerate, deepspeed
- `sim` - `mujoco>=3.1.0,<4.0.0`, `imageio[ffmpeg]>=2.31.0`
- `dev` - ci + train + sim + pre-commit (use this for local dev)

### Console Scripts
- `vla-train` - training entry point (`train.trainer:main`)
- `vla-eval` - evaluation entry point (`eval.evaluate:main`)
- `vla-viewer` - interactive MuJoCo viewer (`sim.viewer:main`)
- `vla-lint-assets` - MJCF asset linter (`sim.asset_linter_cli:main`)
- `vla-gen-bricks` - LEGO brick MJCF generator (`sim.lego.cli:main`)

### Pytest Markers
- `slow`, `gpu`, `mujoco`, `viewer`, `smoke`, `assets`, `lego` - auto-skipped when hardware/packages unavailable

### Training Pipeline
1. `train/trainer.py:main()` is the Hydra entry point
2. `Trainer.__init__` sets up distributed (DDP/NCCL), device, seeds
3. `Trainer.setup()` creates model, optimizer, scheduler, dataloaders
4. `Trainer.train()` runs the training loop with logging/checkpointing

### Key Paths (symlinked to scratch on cluster)
- `checkpoints/` - Model checkpoints
- `logs/` - Training logs and Hydra outputs
- `wandb/` - W&B offline logs
- `cache/` - HuggingFace/data cache

## Simulation

### Viewer
When any task requires visual verification in MuJoCo (new asset, changed collision, camera placement, etc.), append a concrete walkthrough to `docs/viewer-debug-checklist.md` with: the exact `vla-viewer` launch command, which UI panels to open and toggles to enable, step-by-step what to look at and verify (checkboxes), and what "correct" vs "wrong" looks like. Do not assume the user knows the MuJoCo viewer UI.

**Import rule**: `mujoco.viewer` is only imported inside `sim/viewer.py:launch_viewer()`. Never import it at module top level or in training/runtime code.

### Offscreen Rendering
`sim/offscreen.py` provides headless rendering (no display needed). Key API:
- `render_trajectory(model, data, n_steps, config, render_every)` -> list of `RenderedFrame`
- `save_video(frames, path, fps)` -> MP4 file
- `save_sample_frames(frames, dir)` -> PNG files
- `RenderConfig(camera_name=..., render_depth=True, render_segmentation=True)`

**Critical**: Always call `mj_forward(model, data)` before rendering (done automatically inside `render_frame`). Without it, RGB renders black.

**Camera rule**: Use named MJCF cameras for offscreen rendering. The free camera (id=-1) has no useful default viewpoint in headless mode. `test_scene.xml` has an `overhead` camera.

### Smoke Tests
- `tests/test_sim_smoke.py` - pytest suite (`@pytest.mark.smoke` + `@pytest.mark.mujoco`)
- `scripts/validate_sim_smoke.py` - standalone script, artifacts to `logs/sim_smoke/`
- Thresholds: max penetration 5 cm (`data.contact[i].dist`), energy < 1000 J, no NaN
- Render determinism uses `np.allclose(atol=1)` — allows ±1 pixel-value jitter from GPU (EGL) rasteriser
- Set `WANDB_MODE=online` before running `validate_sim_smoke.py` to attach artifacts to W&B

### Cluster Simulation Smoke
- `infra/gilbreth/job_templates/06_smoke_sim_headless.sh` - SLURM job for headless sim on Gilbreth
- Uses `MUJOCO_GL=egl` (NVIDIA EGL on GPU nodes); Apptainer container uses `osmesa` as fallback
- First run on a fresh conda env will `pip install mujoco imageio[ffmpeg]` automatically
- Artifacts land in `logs/sim_smoke/` (symlinked to scratch)
- **ThinLinc policy**: no GUI on cluster by default (headless artifacts only). Use ThinLinc only if a visual bug cannot be diagnosed from saved videos/frames

### Asset Layout
```
sim/assets/
    scenes/              # MJCF scene files (e.g., test_scene.xml)
    robots/<name>/       # Robot models: <name>.xml + meshes/ + textures/
    lego/                # LEGO assets (Phase 1.2)
        defaults.xml     # Contact material default classes (stud, tube, brick_surface, baseplate)
        bricks/          # Generated brick MJCF files (brick_2x2.xml, brick_2x4.xml, brick_2x6.xml)
        baseplates/      # Generated baseplate MJCF files (baseplate_8x8.xml)
```

**Loading scenes**: `sim.asset_loader.load_scene("test_scene")` - single entrypoint. Resolves paths under `sim/assets/scenes/`, delegates to `mujoco_env.load_model()`.

**Loading robots**: `sim.asset_loader.resolve_robot_path("alex")` - expects `sim/assets/robots/alex/alex.xml`.

**Loading LEGO bricks**: `sim.asset_loader.resolve_lego_brick_path("2x4")` - resolves to `sim/assets/lego/bricks/brick_2x4.xml`.

**Loading LEGO baseplates**: `sim.asset_loader.resolve_lego_baseplate_path("8x8")` - resolves to `sim/assets/lego/baseplates/baseplate_8x8.xml`.

**Asset linting**: `vla-lint-assets` checks absolute paths (ERROR), missing referenced files (ERROR), and suspicious mesh scales (WARNING). Run before committing new/modified MJCF files.

**Rule**: All file references in MJCF must be relative. The linter respects `<compiler meshdir="..." texturedir="...">` for path resolution.

### Alex Robot Model (Phase 1.1)
- **Source**: `ihmc-alex-sdk` commit `be25a395e35238bc6385a58bcc50aa047d936a25`. PROVENANCE.md in /sim/assets/robots/alex contains all details
- **Files**: `sim/assets/robots/alex/alex.xml` + `meshes/` (16 OBJ + 3 STL)
- **Scenes**: `alex_upper_body.xml` (robot + floor + cameras), `alex_grasp_test.xml` (robot + table + cube)
- **23 joints** (15 arm + 8 EZGripper): `spine_z` + 7 per arm + 4 EZGripper per arm
- **17 actuators** (15 arm + 2 EZGripper): one `{side}_ezgripper` actuator per hand
- **Fixed base** (no freejoint), **fixed neck** (NECK_Z/Y removed)
- **EE sites**: `left_ee_site`, `right_ee_site` (on gripper body)
- **Tool frame sites**: `left_tool_frame`, `right_tool_frame` (on EZGripper palm, frozen reference)
- **Cameras**: `robot_cam` (head-mounted), `overhead`, `third_person` (scene-level)
- **Collision geoms**: Simplified capsules/boxes (group 3) alongside visual meshes (group 1)
- **Include note**: When loading via `<include>`, the scene file sets `<compiler meshdir>` to resolve mesh paths relative to the robot directory.
- **Dynamics (Phase 1.1.2)**:
  - Integrator: `implicitfast`, solver: 50 iterations, timestep 0.002s
  - Per-joint damping: proximal=2.0, mid=1.5, distal=0.5, gripper=0.3 Ns/rad
  - Armature: 0.01 on arm joints, 0.005 on EZGripper joints
  - Contact: `solref="0.005 1.0"` (critically damped); EZGripper finger pads: `friction="1.5 0.02 0.01"`
  - Actuator `ctrlrange` clamped to joint range (`inheritrange="1"`)
  - Keyframes: `home` (all zeros), `rest` (shoulders abducted, elbows bent), `open_grippers`
- **EZGripper end-effectors (Phase 1.1.3)**:
  - SAKE EZGripper Gen2 (Dynamixel MX-64AR), STL meshes from SAKE repo
  - Underactuated: 4 joints per hand coupled via `<equality>` constraints (1:1 ratio)
  - `gripper_cmd ∈ [0, 1]`: 0 = closed (joint=0), 1 = open (joint=1.94 rad)
  - Command interface: `sim/end_effector.py` — `EZGripperInterface.set_grasp(cmd)`
  - `EndEffectorInterface` ABC supports future Ability Hand (6-DoF) swap
  - Adapter transform from IHMC URDF (`euler="3.14159 1.5708 0"`) — verified via FK symmetry (0.0 cm error, GO verdict)
- **Kinematics validation (Phase 1.1.4)**:
  - **Verdict: GO** — all Tier 1 checks pass, no axis flips or systematic offsets
  - FK symmetry: 0.0000 cm max position error (perfect left/right mirror)
  - Mirror mapping: Y-axis joints keep sign, X-axis and Z-axis joints negate sign
  - Workspace covers LEGO table region (X∈[0.3,0.6], Y∈[-0.3,0.3], Z∈[0.8,1.2] m)
  - All 14 arm joints verified kinematically active (position or orientation effect)
  - Report: `docs/kinematics-validation-report.md`, script: `scripts/validate_kinematics.py`
  - Tests: `tests/test_alex_model.py::TestAlexKinematics` (6 tests)
  - Tier 2 (reference FK comparison vs SDK) available via `ALEX_SDK_PATH` env var
- **Control pipeline** (`sim/control.py`):
  - `AlexController`: safety clamps + rate limiting (80% of hardware velocity limit per timestep)
  - `JOINT_VELOCITY_LIMITS`: per-joint velocity limits from URDF (rad/s); EZGripper: 6.6 rad/s
  - `JOINT_EFFORT_LIMITS`: per-joint effort limits matching MJCF forcerange (N·m); EZGripper: 8.0 N·m
- **Action space & control contract (Phase 1.1.5)**:
  - **Frozen 17-D action vector**: `[Δq_spine(1), Δq_left_arm(7), Δq_right_arm(7), gripper_left(1), gripper_right(1)]`
  - Per-arm joint order: SHOULDER_Y, SHOULDER_X, SHOULDER_Z, ELBOW_Y, WRIST_Z, WRIST_X, GRIPPER_Z
  - Arm actions normalized to `[-1, 1]`, mapped to Δq via per-joint `delta_q_max = vel_limit * control_dt * 0.8`
  - Gripper actions absolute `[0, 1]` (0=closed, 1=open)
  - **Control rate**: 20 Hz (50 ms per action), 25 physics substeps per action (timestep 0.002s)
  - **Pipeline**: `AlexActionSpace.apply_action()` denormalizes → computes target → `AlexController` clamps positions → `data.ctrl`
  - `AlexActionSpace` disables `AlexController` rate limiting (`rate_limit_factor=0.0`) because normalization already bounds deltas
  - `SimRunner.step(action)` applies action + runs 25 substeps; `step_sequence(actions)` for action chunks
  - Config: `configs/sim/default.yaml` (control_hz, rate_limit_factor)
  - Constants frozen in `sim/action_space.py`: `ACTION_DIM=17`, `ARM_DIM=15`, `GRIPPER_DIM=2`, `ARM_ACTUATOR_NAMES`
  - `ARM_JOINT_NAMES` derived in `sim/robot_state.py` from `ARM_ACTUATOR_NAMES` (maps actuator names to joint names)
- **Robot state contract (Phase 1.1.6)**:
  - **Frozen 52-D state vector**: `[q(15), q_dot(15), gripper(2), left_ee_pos(3), left_ee_quat(4), right_ee_pos(3), right_ee_quat(4), left_ee_vel(3), right_ee_vel(3)]`
  - Reference frame: World (Z-up, X-forward, Y-left); robot base fixed at [0, 0, 1.0]
  - Quaternion convention: MuJoCo `[w, x, y, z]`
  - EE sites: `left_tool_frame`, `right_tool_frame` (frozen reference on gripper palm)
  - EE velocity via `mj_objectVelocity()` (MuJoCo-native, exact Jacobian method, world frame)
  - Joint velocities via `data.qvel[model.jnt_dofadr[jnt_id]]` (DOF address, not joint ID)
  - `AlexRobotState(model)`: extracts state from `MjData`, provides `get_state()` → `RobotState` dataclass, `get_flat_state()` → 52-D array
  - `RobotState` dataclass: named fields + `to_flat_array()` / `from_flat_array()` / `validate()`
  - Normalization: q min-max, q_dot by vel limits, gripper [0,1]→[-1,1], EE pos by workspace, EE quat pass-through, EE vel by max
  - `SimRunner.step()` now returns `RobotState`; `SimRunner.get_state()` for on-demand extraction
  - Config: `configs/sim/default.yaml` `state:` section (dims, EE sites, frame, normalization, workspace bounds)
  - Constants frozen in `sim/robot_state.py`: `STATE_DIM=52`, `Q_DIM=15`, `Q_DOT_DIM=15`, `GRIPPER_STATE_DIM=2`, `EE_POSE_DIM=14`, `EE_VEL_DIM=6`, named slices
- **Multi-view cameras (Phase 1.2.4)**:
  - **Frozen 4-view contract**: `overhead` (fixed workspace), `left_wrist_cam` (body-attached to `left_gripper`), `right_wrist_cam` (body-attached to `right_gripper`), `third_person` (fixed external)
  - Default resolution: 320×320 (square, VLA-standard), capture rate: 20 Hz (aligned with policy rate)
  - **Per camera**: RGB + Depth + Segmentation (all modalities enabled in config; `MultiViewRenderer` defaults to RGB-only for performance)
  - `MultiViewRenderer(model)`: renders all 4 frozen views from same sim timestep via single shared `mujoco.Renderer`; each `capture()` call returns live per-camera metadata
  - `MultiViewFrame`: dataclass with `views: dict[str, RenderedFrame]`, `metadata: dict[str, CameraMetadata]`, `step_index`, `timestamp`
  - `CameraMetadata`: fovy, world-frame pos/mat, `intrinsics: CameraIntrinsics` (fx, fy, cx, cy, fovx, fovy), `is_body_attached: bool`
  - `CameraIntrinsics`: pinhole model derived from MuJoCo fovy. `compute_intrinsics(fovy, width, height)` computes K matrix. `to_matrix()` returns 3×3 K.
  - `robot_cam` (head-mounted) stays in MJCF for interactive debugging but is NOT in the frozen 4-view contract
  - Wrist cameras attached to `left_gripper`/`right_gripper` bodies (rotate with gripper_z joint); fovy=75°; xyaxes looks along gripper +Z (toward fingertips)
  - Camera body-attachment detection: `model.cam_bodyid[cam_id] != 0` → body-attached; 0 → fixed world frame
  - Images decoupled from 52-D `RobotState` — separate rendering pipeline, no changes to state contract or `SimRunner`
  - Config: `configs/sim/default.yaml` `camera:` section (4 views, 320×320, depth/segmentation enabled)
  - Constants frozen in `sim/camera.py`: `CAMERA_NAMES`, `NUM_VIEWS=4`, `DEFAULT_WIDTH=320`, `DEFAULT_HEIGHT=320`

### LEGO Press-Fit (Phase 1.2.0)
- **Approach**: Soft Real Press-Fit — geometrically accurate studs/tubes with compliant contacts (no hard interference, no magic snap constraints). Full spec: `docs/press-fit-spec.md`
- **Stud collision**: cylinder r=2.35 mm, h=1.7 mm (50 µm undersize for clearance)
- **Tube collision**: ring of 8 capsules per tube (r=0.55 mm capsules on 3.0 mm radius circle); provides geometric interlock + friction retention
- **Four contact material classes (tuned Phase 1.2.2)**:

  | Class | contype | conaffinity | solref | solimp | friction | condim |
  |-------|---------|-------------|--------|--------|----------|--------|
  | `lego/stud` | 6 | 7 | 0.003 1.0 | 0.97 0.995 0.001 0.5 4 | 0.65 | 4 |
  | `lego/tube` | 4 | 4 | 0.003 1.0 | 0.97 0.995 0.001 0.5 4 | 0.65 | 4 |
  | `lego/brick_surface` | 2 | 3 | 0.005 1.0 | 0.9 0.95 0.001 0.5 2 | 0.4 | 3 |
  | `lego/baseplate` | 2 | 3 | 0.005 1.0 | 0.9 0.95 0.001 0.5 2 | 0.6 | 3 |

- **Contact isolation**: Studs contype=6/conaffinity=7 (contacts everything); tubes contype=4/conaffinity=4 (contacts studs only); surfaces contype=2/conaffinity=3 (robot + studs)
- **Capture envelope**: lateral ±0.5 mm, angular ±3°, approach zone 2.0 mm
- **Key thresholds (tuned)**: insertion success ≥95%, retention ≥0.15 N/brick, max penetration <2 mm, drift <1 mm. See `docs/contact-tuning-notes.md` for gap analysis vs spec targets
- **Solver**: iterations increased to 80 (from 50) for multi-stud contact scenes
- **Brick set**: {2×2, 2×4, 2×6} at standard height 9.6 mm; stud pitch 8.0 mm; ABS density 1050 kg/m³
- **Config**: `configs/sim/lego.yaml` holds all tunable geometry, contact, and threshold values

### LEGO Procedural Bricks (Phase 1.2.1)
- **Generator**: `sim/lego/brick_generator.py` — procedural MJCF generation using `xml.etree.ElementTree`, no external mesh dependencies
- **Brick types**: `BRICK_TYPES` dict in `sim/lego/constants.py` — `BrickType` frozen dataclass with `nx`, `ny`, computed `n_studs`, `n_tubes`, `shell_half_*`
- **Collision geoms**: hollow shell (4 walls + top plate, `lego/brick_surface`), cylinder studs (`lego/stud`), 8-capsule ring tubes (`lego/tube`); collision group 3
- **Visual geoms**: MuJoCo primitives (group 0, contype=0/conaffinity=0); studs at real 2.4 mm radius, LEGO red
- **Mass**: computed from spec formula in `sim/lego/mass.py` (ABS density × hollow shell + studs + tubes volume)
- **Connector metadata**: `sim/lego/connector.py` — `ConnectorPoint` (id, kind, position, axis, radius), `BrickConnectors` (studs tuple, tubes tuple), stable IDs (`stud_0_1`, `tube_2`). Tubes at positive Z (inside cavity)
- **Defaults**: `sim/assets/lego/defaults.xml` — four contact material classes (includable MJCF fragment); also inlined in standalone brick MJCF
- **Generated assets**: `sim/assets/lego/bricks/brick_2x{2,4,6}.xml` — standalone MJCF with floor, solver settings, freejoint; regenerate via `vla-gen-bricks`
- **Brick origin**: center of bottom face (Z=0), stud grid centered on XY
- **Tests**: `tests/test_lego_bricks.py` (59 tests: constants, mass, connectors, MJCF loading, contact classes, physics stability, lint)
- **Validation**: `scripts/validate_lego_bricks.py` (asset existence, lint, loading, dimensions, mass, connectors, physics, regeneration consistency)

### LEGO Contact Physics (Phase 1.2.2a)
- **Scene builder**: `sim/lego/contact_scene.py` — programmatic MJCF scenes for insertion tests. `generate_insertion_scene()` (fixed base + free top), `generate_stack_scene()` (multi-brick), `check_stud_engagement()` (Z-threshold engagement check)
- **Measurement utils**: `sim/lego/contact_utils.py` — `run_insertion()` (force-driven, 5x gravity via `data.xfrc_applied`), `apply_force_ramp()` (linearly increasing force for retention measurement), `measure_position_jitter()`, `measure_position_drift()`, `InsertionResult` dataclass
- **Hollow shell**: Shell collision changed from single solid box to 5 thin boxes (4 walls + top plate), leaving bottom open for stud engagement
- **Contact isolation split**: `lego/stud_tube` → `lego/stud` (contype=6) + `lego/tube` (contype=4). Tubes only contact studs, not shell walls
- **Tube geometry**: ring radius 3.0 mm, capsule radius 0.55 mm, height 1.7 mm, 8 capsules at 45° intervals (aligns with diagonal stud positions)
- **Retention gap**: Achievable ~0.06 N/stud vs spec 0.3 N/stud. Root cause: MuJoCo rigid-body contacts cannot model ABS elastic deformation. See `docs/contact-tuning-notes.md`
- **Tests**: `tests/test_lego_contacts.py` (20 physics tests: insertion aligned/near-miss/miss/angular, retention pull-off/shear/hold/gravity, stability cycles/penetration/energy/jitter, performance)
- **Validation**: `scripts/validate_lego_contacts.py` (standalone pass/fail validation with artifacts to `logs/lego_contacts/`)

### LEGO Hybrid Retention (Phase 1.2.2b)
- **Two modes**: `retention_mode: physics` (default, unchanged) or `spec_proxy` (hybrid weld constraints). Config in `configs/sim/lego.yaml`
- **ConnectionManager**: `sim/lego/connection_manager.py` — runtime weld activation/deactivation. Monitors brick pair Z + XY alignment, activates pre-declared `<weld>` equality constraints after sustained engagement (50 steps). Release via displacement hysteresis (2 mm threshold, 25 steps dwell)
- **Scene generation**: `generate_insertion_scene(..., retention_mode="spec_proxy")` adds `<equality><weld active="false" .../>` to MJCF. `setup_connection_manager()` helper creates `ConnectionManager` and finds weld eq IDs by body matching
- **Stepping integration**: Optional `connection_manager` param on `run_insertion()`, `apply_force_ramp()`, `measure_position_drift()`, `measure_position_jitter()`, `perform_insertion_then_measure()`. All default to `None` — zero impact on physics-mode callers
- **Design constraint**: `perform_insertion_then_measure()` runs insertion physics-only (no weld during approach), then activates ConnectionManager during settle phase only
- **Proxy labeling**: All hybrid results explicitly labeled `[PROXY]` in tests, validation output, and docs
- **Achieved proxy performance**: pull-off ~0.52 N/stud, shear ~0.63 N/stud, drift ~0.02 mm (all pass spec targets)
- **Tests**: `tests/test_lego_contacts.py` — 6 `TestHybridRetention` tests (weld activation, pull-off, shear, hold, misalignment rejection, release) + 5 `TestConnectionManager` unit tests
- **Validation**: `python scripts/validate_lego_contacts.py --mode both` runs physics + spec-proxy suites with separate pass/fail

### LEGO Baseplate & Workspace (Phase 1.2.3)
- **Baseplate type**: 8×8 only (`BASEPLATE_TYPES` in `sim/lego/constants.py`). `BaseplateType` frozen dataclass: `nx_studs`, `ny_studs`, `thickness` (3.2 mm = standard LEGO plate height)
- **Generator**: `sim/lego/baseplate_generator.py` — procedural MJCF following `brick_generator.py` patterns. Solid plate surface (`lego/baseplate` class) + stud grid (`lego/stud` class), no tubes, no freejoint (fixed to world)
- **Connectors**: `BaseplateConnectors` in `sim/lego/connector.py` — studs only (no tubes). Stud Z at `thickness + STUD_HALF_HEIGHT`
- **Mass**: `compute_baseplate_mass()` in `sim/lego/mass.py` — solid plate + stud volume × ABS density
- **Generated assets**: `sim/assets/lego/baseplates/baseplate_8x8.xml` (131 geoms); regenerate via `vla-gen-bricks`
- **Asset loading**: `sim.asset_loader.resolve_lego_baseplate_path("8x8")`
- **Baseplate insertion scenes**: `generate_baseplate_insertion_scene()` in `contact_scene.py` — fixed baseplate + free brick. `contact_utils.py` functions accept `base_surface_height` param (defaults to `BRICK_HEIGHT` for backward compat; use `baseplate.thickness` for baseplates)
- **Workspace scene**: `generate_workspace_scene()` in `contact_scene.py` → `sim/assets/scenes/alex_lego_workspace.xml`. Alex robot + table (pos 0.45,0,0.75) + baseplate on table surface + cameras (overhead, third_person, workspace_closeup). Spawn region: X∈[0.25,0.65], Y∈[-0.20,0.20]
- **Config**: `configs/sim/lego.yaml` `baseplate:` (default_size, thickness_m, color) and `workspace:` (table_pos, table_size, baseplate_offset, spawn_region) sections
- **Tests**: `tests/test_lego_baseplate.py` (29 tests: constants, mass, connectors, MJCF generation, contact physics, workspace scene, asset resolution)
- **Validation**: `scripts/validate_lego_baseplate.py` (21 checks, artifacts to `logs/lego_baseplate/`)

### LEGO Episode Manager (Phase 1.2.5)
- **Approach**: Template Model — compile MJCF once at `EpisodeManager.__init__()` with all brick slots declared; use `mj_resetData()` + qpos writes for fast per-episode resets (no XML recompilation per episode)
- **Parked bricks**: Unused slots placed at Z=-10 with contacts disabled (`model.geom_contype[g]=0`); free-fall harmlessly without floor interaction during settle
- **`EpisodeManager`** (`sim/lego/episode_manager.py`): main class
  - `reset(seed, level, n_active) → EpisodeInfo` — deterministic episode reset
  - Spawn sampling: `np.random.PCG64(seed)`, min-distance constraints, optional random yaw, retry with `max_spawn_attempts=50`
  - Settle phase: step physics until `max(cvel[3:]) < 0.001 m/s` and penetration < 5 mm, or timeout at 500 steps
  - Curriculum: `LEVEL_SINGLE_BRICK=1` (1 brick), `LEVEL_SINGLE_CONNECTION=2` (1 brick), `LEVEL_MULTI_STEP=3` (2-4 bricks, random)
- **`EpisodeInfo`** (frozen dataclass): seed, level, brick_types, spawn_poses, settle_steps, settle_success
- **`SpawnPose`** (frozen dataclass): position (XYZ), quaternion [w,x,y,z]
- **`ResetMetrics`** (dataclass): tracks success_rate, avg_settle_steps, failure_reasons across all resets
- **`generate_episode_scene()`** in `contact_scene.py`: extends workspace scene with N pre-declared free brick slots at Z=-10
- **`ConnectionManager.reset()`**: clears pair engagement counters, deactivates all weld constraints between episodes
- **Template file**: written to `sim/assets/scenes/_episode_template.xml` for `<include>` path resolution
- **Config**: `configs/sim/lego.yaml` `episode:` section (max_bricks, brick_set, settle params, spawn params, curriculum)
- **Tests**: `tests/test_episode_manager.py` (37 tests: dataclasses, spawn sampling, reset determinism, settle, curriculum, reliability stress)
- **Validation**: `scripts/validate_episode_manager.py` (25 checks, artifacts to `logs/episode_manager/`); 100% success rate achieved

### LEGO MVP-3 Task (Phase 1.2.6)
- **Approach**: Force-based scripted assembly — kinematic positioning above target (qpos write) + physics-based press-fit insertion via `xfrc_applied` downward force. Proves assembly pipeline end-to-end with real contact physics
- **Task specification** (`sim/lego/task.py`):
  - `PlacementTarget` (frozen dataclass): slot_index, brick_type, target_position (world XYZ), target_quaternion [w,x,y,z], base_body_name
  - `AssemblyGoal` (frozen dataclass): ordered tuple of `PlacementTarget`, seed, level
  - `PlacementResult` (dataclass): success, position_error_m, z_engaged, stable, insertion_steps, final_position
  - `AssemblyResult` (dataclass): placements list, n_successful, all_placed, structure_stable, total_physics_steps, max_penetration_m
  - `compute_target_position()`: baseplate stud grid → world position for brick body origin
  - `compute_brick_on_brick_target()`: live base brick position → stacking target position
  - `generate_assembly_goal()`: deterministic from seed, non-overlapping footprints, optional `stacking=True` for brick-on-brick
  - `check_placement()`: XY + Z tolerance check → `(success, xy_error_m)`
  - `check_stability()`: step physics for hold_duration, check body velocities
  - `evaluate_assembly()`: aggregate metrics + final stability hold
- **Scripted assembly** (`sim/lego/scripted_assembly.py`):
  - `ScriptedAssembler(model, data)`: force-based executor
  - `execute_placement(target) → PlacementResult`: approach → insert → settle → check
  - `execute_assembly(goal, hold_duration_s) → AssemblyResult`: sequential placement + final hold
  - For brick-on-brick targets, recomputes position from base brick's live position at execution time
  - `AssemblyStepLog` dataclass for per-step logging
- **Assembly scope**: baseplate placements (side-by-side) + brick-on-brick stacking
- **Success criteria**: XY tolerance 1 mm (`placement_xy_tol_m`), Z margin 0.5 mm, stability hold 2.0 s, velocity threshold 0.001 m/s
- **Config**: `configs/sim/lego.yaml` `task:` section (tolerances, approach height, insertion force, settle steps, stud margin)
- **Tests**: `tests/test_lego_task.py` (33 tests: dataclasses, target computation, goal generation, placement check, scripted placement, multi-brick assembly, stacking, failure detection, metrics, deterministic replay)
- **Validation**: `scripts/validate_lego_task.py` (15 checks, artifacts to `logs/lego_task/`); 100% engagement on single-brick stress test

## Code Style

- **Line length**: 100 (Black)
- **Type hints**: Required for public APIs
- **Docstrings**: Google style
- **Imports**: isort with Black-compatible profile

## Git Workflow

- **Branches**: `feature/`, `fix/`, `docs/`, `refactor/`, `test/`, `exp/`
- **Commits**: Conventional Commits format (`feat:`, `fix:`, `docs:`, etc.)
- **Merge**: Squash merge for feature branches

## Environment Variables

Key variables (see `.env.example`):
- `VLA_SCRATCH_ROOT` - Scratch storage for checkpoints/logs
- `WANDB_MODE` - online/offline/disabled
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` - Consistent GPU numbering

## Containers

Docker/Apptainer images contain **dependencies only** - code is bind-mounted at `/workspace` from your git checkout. Use containers for cluster training and reproducibility; use native Python for local dev and CI.

```bash
./scripts/docker-run.sh python -m train.trainer trainer=debug cluster=local   # Docker (lab PC)
./scripts/apptainer-run.sh python -m train.trainer cluster=gilbreth            # Apptainer (Gilbreth HPC)
```

**Adding/changing dependencies**: update `pyproject.toml`, push to `main` - CI rebuilds the image automatically.

**CI rebuild triggers**: `Dockerfile`, `pyproject.toml`, `.dockerignore`, `scripts/container-entrypoint.sh`. Note: `apptainer.def` does NOT trigger rebuilds (CI builds Apptainer from the Docker image digest). Code-only changes do NOT trigger rebuilds.
