<div align="center">

# WorldSim

**A PyTorch framework for training world simulation models with Transformer architectures**

[![CI](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## Overview

WorldSim is a modular training framework designed for world simulation models. It leverages Transformer architectures for sequence modeling of continuous state vectors, enabling next-state prediction in simulated environments.

Built with **configuration-first principles** using [Hydra](https://hydra.cc/), WorldSim supports seamless transitions between local development and HPC cluster training (Purdue Gilbreth).

## Features

- **Transformer Architecture** — State-of-the-art sequence modeling for continuous state prediction
- **Hydra Configuration** — Hierarchical, composable configs with CLI overrides
- **Distributed Training** — PyTorch DDP with NCCL backend, DeepSpeed integration ready
- **HPC Ready** — Pre-configured for Slurm clusters with container support (Docker + Apptainer)
- **Experiment Tracking** — Weights & Biases integration for logging and visualization
- **Reproducible** — Deterministic training with seed control and checkpoint management

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         WorldSim Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│   │  Config  │───▶│   Trainer    │───▶│   TransformerModel   │  │
│   │  (Hydra) │    │              │    │                      │  │
│   └──────────┘    │  • DDP Setup │    │  • Input Projection  │  │
│                   │  • Optimizer │    │  • Positional Enc    │  │
│   ┌──────────┐    │  • Scheduler │    │  • Encoder Layers    │  │
│   │ Dataset  │───▶│  • Ckpt Mgmt │    │  • Output Projection │  │
│   │          │    └──────────────┘    └──────────────────────┘  │
│   └──────────┘                                                   │
│                                                                  │
│   Input: [batch, seq_len, 256] ──▶ Output: [batch, seq_len, 256]│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/PatrizioAcquadro/VLA-LEGO_Project.git
cd VLA-LEGO_Project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Quick Start

### Local Training (Debug)

```bash
# Quick smoke test (100 steps, small batch)
python -m train.trainer trainer=debug cluster=local
```

### Full Training

```bash
# Train with base model configuration
python -m train.trainer cluster=local

# Train with large model on GPU cluster
python -m train.trainer model=large cluster=gilbreth
```

### Configuration Overrides

All settings can be overridden from the command line:

```bash
python -m train.trainer \
    model=large \
    trainer.optimizer.lr=1e-5 \
    trainer.training.batch_size_per_device=16 \
    trainer.training.max_steps=50000
```

## Configuration

WorldSim uses [Hydra](https://hydra.cc/) for configuration management. All configs are in `configs/`:

| Config Group | Options | Description |
|-------------|---------|-------------|
| `model` | `base`, `large` | Model architecture settings |
| `trainer` | `default`, `debug` | Training hyperparameters |
| `data` | `default` | Dataset and dataloader settings |
| `cluster` | `local`, `gilbreth` | Cluster-specific settings |
| `logging` | `wandb` | Experiment tracking |

### Example: Custom Configuration

```bash
# Combine multiple config overrides
python -m train.trainer \
    model=large \
    trainer=default \
    cluster=local \
    experiment.seed=123 \
    trainer.optimizer.lr=5e-5
```

## Project Structure

```
VLA-LEGO_Project/
├── configs/                 # Hydra configuration files
│   ├── config.yaml          # Main config (composes defaults)
│   ├── model/               # Model architectures (base, large)
│   ├── trainer/             # Training settings (default, debug)
│   ├── data/                # Dataset configuration
│   ├── cluster/             # Cluster settings (local, gilbreth)
│   └── logging/             # W&B integration
├── data/                    # Data loading and processing
│   ├── dataset.py           # Dataset classes
│   └── loader.py            # DataLoader utilities
├── models/                  # Model implementations
│   ├── transformer.py       # TransformerModel
│   └── utils.py             # Model utilities
├── train/                   # Training code
│   └── trainer.py           # Main Trainer class
├── eval/                    # Evaluation code
├── scripts/                 # Utility scripts
├── tests/                   # Test suite
├── docs/                    # Documentation
├── Dockerfile               # Container definition
└── apptainer.def            # Singularity/Apptainer definition
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint
ruff check .

# Type checking
mypy sim models train eval --ignore-missing-imports

# Run all checks (pre-commit)
pre-commit run --all-files
```

### Validating Configurations

```bash
# Validate all config combinations
python scripts/validate_configs.py
```

## HPC Cluster Usage

### Gilbreth (Purdue)

```bash
# Submit training job
sbatch scripts/train.sh

# Interactive session
sinteractive -A <account> -n 1 -g 1 -t 4:00:00

# Load container and run
apptainer exec --nv worldsim.sif python -m train.trainer cluster=gilbreth
```

See [docs/git-workflow.md](docs/git-workflow.md) for detailed cluster instructions.

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Development guidelines for AI assistants |
| [docs/git-workflow.md](docs/git-workflow.md) | Git branching and workflow guide |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |

## Roadmap

- [ ] Data pipeline implementation
- [ ] Advanced model architectures (Flow Matching)
- [ ] Multi-GPU scaling optimization
- [ ] Evaluation benchmarks

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a PR.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Configuration powered by [Hydra](https://hydra.cc/)
- Experiment tracking with [Weights & Biases](https://wandb.ai/)

---

<div align="center">
  <sub>Built with care at Purdue University</sub>
</div>
