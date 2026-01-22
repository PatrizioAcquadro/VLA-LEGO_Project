# WorldSim

[![CI](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/actions/workflows/ci.yml)

World Simulation Model Training

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/worldsim.git
cd worldsim

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Training

```bash
# Local debug run
python -m train.trainer trainer=debug cluster=local

# Full training on Gilbreth
python -m train.trainer model=large cluster=gilbreth
```

### Configuration

All configuration is managed via Hydra. Override any setting:

```bash
python -m train.trainer \
    model=large \
    trainer.optimizer.lr=1e-5 \
    trainer.training.batch_size_per_device=16
```

See `configs/` for all available options.

## Project Structure

```
worldsim/
├── configs/           # Hydra configuration files
│   ├── config.yaml    # Main config
│   ├── model/         # Model architectures
│   ├── trainer/       # Training settings
│   ├── data/          # Data loading settings
│   ├── cluster/       # Cluster-specific settings
│   └── logging/       # W&B and logging settings
├── sim/               # Simulation environment
├── data/              # Data loading and processing
├── models/            # Model architectures
├── train/             # Training code
├── eval/              # Evaluation code
├── scripts/           # Utility scripts
├── tests/             # Test suite
└── docs/              # Documentation
```

## Development

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_models.py

# With coverage
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint
ruff check .

# Type check
mypy sim models train eval
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

## Cluster Usage (Gilbreth)

See [docs/gilbreth.md](docs/gilbreth.md) for detailed cluster instructions.

Quick example:

```bash
sbatch scripts/train.sh
```

## License

MIT
