# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLA-LEGO is a Vision-Language-Action system for bimanual robotic LEGO assembly. It replicates and extends the EO-1 model architecture (Qwen 2.5 VL backbone with autoregressive decoding + flow matching) for coordinated two-arm manipulation on the Unitree H1 humanoid robot.

## Common Commands

### Development Setup
```bash
pip install -e ".[dev]"
pre-commit install
```

### Training
```bash
# Debug training (100 steps, small batch)
python -m train.trainer trainer=debug cluster=local

# Full training with base model
python -m train.trainer cluster=local

# Full training with large model on Gilbreth HPC
python -m train.trainer model=large cluster=gilbreth

# Override any config value
python -m train.trainer trainer.optimizer.lr=1e-5 trainer.training.batch_size_per_device=16
```

### Testing
```bash
pytest                              # All tests
pytest tests/test_models.py -v      # Single file
pytest --cov=. --cov-report=html    # With coverage
pytest -m "not slow and not gpu"    # Skip slow/GPU tests
```

### Code Quality
```bash
black .                                          # Format
isort .                                          # Sort imports
ruff check .                                     # Lint
mypy sim models train eval --ignore-missing-imports  # Type check
pre-commit run --all-files                       # All checks
python scripts/validate_configs.py               # Validate Hydra configs
```

### HPC (Gilbreth)
```bash
sbatch infra/gilbreth/job_templates/01_smoke_1gpu.sh    # Single GPU test
sbatch infra/gilbreth/job_templates/04_smoke_8gpu_deepspeed.sh  # Multi-node
```

## Architecture

### Module Structure
- **configs/** - Hydra configuration hierarchy (model, trainer, data, cluster, logging)
- **models/** - TransformerModel implementation with MSE loss for state prediction
- **train/** - Trainer class handling distributed training, checkpointing, validation
- **data/** - Dataset classes (DummyDataset for testing, SimulationDataset for real data)
- **eval/** - Evaluation scripts
- **tracking/** - W&B experiment tracking with distributed-safe logging
- **infra/gilbreth/** - SLURM job templates and HPC setup scripts

### Configuration System
All hyperparameters flow through Hydra configs in `configs/`. Key config groups:
- `model`: base (256 hidden, 4 layers) or large (512 hidden, 8 layers)
- `trainer`: default or debug (100 steps, fp32)
- `cluster`: local or gilbreth (DeepSpeed, multi-GPU)

**Configuration-first principle**: Never hardcode values in code. Use `cfg.trainer.optimizer.lr` style access.

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

## Container Model (Deps-Only)

Docker/Apptainer images contain **dependencies only** — code is bind-mounted at `/workspace` from your git checkout.

### Why deps-only?
- Code changes don't require container rebuilds
- `git pull` updates code instantly
- Reproducibility: run = (git commit) + (image digest)

### When to use containers
- Training on GPU clusters (Gilbreth) via Apptainer
- Ensuring consistent dependencies across machines
- Running on machines without local Python/CUDA setup

### When NOT to use containers
- Quick local development/debugging (use native Python)
- CI checks (runs native Python, not containers)

### Running with containers
```bash
# Docker (lab PC)
./scripts/docker-run.sh python -m train.trainer trainer=debug cluster=local

# Apptainer (Gilbreth HPC)
./scripts/apptainer-run.sh python -m train.trainer cluster=gilbreth
```

### Adding/changing dependencies
1. Update `pyproject.toml` (add to appropriate section)
2. Push to `main` (or create release tag `v*`)
3. CI automatically rebuilds the container
4. Pull the new image before running

### CI/CD container rebuild triggers
Heavy builds (Docker + Apptainer) run only when these files change on `main`:
- `Dockerfile`
- `pyproject.toml`
- `.dockerignore`
- `scripts/container-entrypoint.sh`

**Note**: `apptainer.def` does NOT trigger rebuilds. CI builds Apptainer images from the Docker image digest, not from apptainer.def directly.

Code-only changes do NOT trigger container rebuilds.
