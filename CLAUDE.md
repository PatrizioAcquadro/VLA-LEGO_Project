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
- **sim/** - MuJoCo simulation: `mujoco_env.py` (load/step/determinism), `env_meta.py` (metadata), `viewer.py` (interactive debug viewer), `offscreen.py` (headless rendering + video export), `asset_loader.py` (single entrypoint: `load_scene()`), `asset_linter.py` (MJCF validation), `assets/` (MJCF scenes + robot models)
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
