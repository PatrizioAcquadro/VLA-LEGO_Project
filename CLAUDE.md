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

### Pytest Markers
- `slow`, `gpu`, `mujoco`, `viewer`, `smoke`, `assets` - auto-skipped when hardware/packages unavailable

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
```

**Loading scenes**: `sim.asset_loader.load_scene("test_scene")` - single entrypoint. Resolves paths under `sim/assets/scenes/`, delegates to `mujoco_env.load_model()`.

**Loading robots**: `sim.asset_loader.resolve_robot_path("alex")` - expects `sim/assets/robots/alex/alex.xml`.

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
- **Multi-view cameras (Phase 1.1.7)**:
  - **Frozen 2-view contract**: `robot_cam` (head-mounted ego, moves with spine_z) + `third_person` (fixed external)
  - Default resolution: 320x240, capture rate: 20 Hz (aligned with policy rate)
  - `MultiViewRenderer(model)`: renders all frozen views from same sim timestep via single shared `mujoco.Renderer`
  - `MultiViewFrame`: dataclass with `views: dict[str, RenderedFrame]`, `step_index`, `timestamp`
  - `CameraMetadata`: fovy, world-frame pos/mat via `data.cam_xpos`/`data.cam_xmat` (live for body-attached cameras)
  - Images decoupled from 52-D `RobotState` — separate rendering pipeline, no changes to state contract or `SimRunner`
  - Reuses `sim/offscreen.py` functions (`render_frame`, `resolve_camera_id`, etc.) — no modifications to offscreen module
  - Config: `configs/sim/default.yaml` `camera:` section (views, resolution, capture_hz, depth/segmentation toggles)
  - Constants frozen in `sim/camera.py`: `CAMERA_NAMES`, `NUM_VIEWS=2`, `DEFAULT_WIDTH=320`, `DEFAULT_HEIGHT=240`

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
