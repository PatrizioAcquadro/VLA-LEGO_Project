# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WorldSim is a world simulation model training framework built with PyTorch and Hydra for configuration management. It uses Transformer architectures for sequence modeling of simulation states, with support for distributed training on HPC clusters (specifically Purdue's Gilbreth).

## Common Commands

```bash
# Install dependencies (development mode)
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Training
python -m train.trainer trainer=debug cluster=local    # Local debug run
python -m train.trainer model=large cluster=gilbreth   # Full training on cluster

# Override any config setting
python -m train.trainer trainer.optimizer.lr=1e-5 trainer.training.batch_size_per_device=16

# Testing
pytest                           # All tests
pytest tests/test_models.py      # Single test file
pytest -k test_forward           # Run tests matching pattern
pytest --cov=. --cov-report=html # With coverage

# Code quality
black .                          # Format code
isort .                          # Sort imports
ruff check .                     # Lint
mypy sim models train eval       # Type check
pre-commit run --all-files       # Run all hooks
```

## Architecture

### Configuration System (Hydra)
All configuration lives in `configs/` with hierarchical YAML files:
- `config.yaml` - Main entry point, composes defaults
- `model/` - Model architectures (base.yaml, large.yaml)
- `trainer/` - Training settings (default.yaml, debug.yaml)
- `data/` - Data loading settings
- `cluster/` - Cluster-specific settings (local.yaml, gilbreth.yaml)
- `logging/` - W&B integration

Override any setting via CLI: `python -m train.trainer model.architecture.hidden_size=512`

### Core Modules
- **`train/trainer.py`** - Main `Trainer` class with training loop, distributed setup, checkpointing. Entry point: `@hydra.main` decorator wires config.
- **`models/transformer.py`** - `TransformerModel` using PyTorch's TransformerEncoder. Expects input shape `[batch, seq_len, state_dim=256]`.
- **`data/dataset.py`** - `SimulationDataset` for real data, `DummyDataset` for smoke testing.
- **`data/loader.py`** - `create_dataloader()` handles distributed sampling.
- **`eval/evaluate.py`** - `Evaluator` class for model evaluation.

### Data Flow
Input tensors are continuous state vectors (not tokens): `[batch, seq_len, 256]` → TransformerModel → `[batch, seq_len, 256]` with MSE loss for next-state prediction.

### Distributed Training
Uses PyTorch DDP with NCCL backend. Environment variables `WORLD_SIZE`, `RANK`, `LOCAL_RANK` control distribution. DeepSpeed integration is configured but optional.

## Key Conventions

- **No hardcoded paths/hyperparameters** - Everything goes through Hydra config
- **Commit messages** - Follow Conventional Commits: `feat(scope): description`, `fix(scope): description`
- **Branch naming** - `feature/*`, `fix/*`, `exp/*`
- **Line length** - 100 characters (Black/isort configured)
- **Python version** - 3.10+, uses `|` union types and modern type hints
