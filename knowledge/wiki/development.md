# Development

## Environment

Use Python 3.10 or newer and a dedicated environment. The validated host uses Python 3.10; the locally tested Docker image uses Python 3.12. GPU models require a compatible NVIDIA driver/PyTorch CUDA runtime. Current dependency ranges and optional groups are defined only in [pyproject.toml](../../pyproject.toml).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/validate_configs.py
```

Install `.[sim]` for simulation only, `.[vlm]` for model components, `.[tracking]` for optional W&B, or `.[ci,sim]` for the CPU software/physics suite. `dev` includes these and pre-commit. A host missing Python's venv/ensurepip support needs the matching OS venv package. In this audit the existing Python 3.10 user stack was reused through `--without-pip --system-site-packages`; that is an environment workaround, not an isolated dependency installation.

HF cache convention: `HF_HOME` is the Hugging Face root; `HF_HUB_CACHE` is its `hub` directory. Both are optional. Put large outputs and weights outside Git; `outputs/`, datasets, checkpoints, caches and local assistant state are ignored.

## Checks

```bash
make check
MUJOCO_GL=egl python -m pytest tests -q
python scripts/check_install.py
```

`make check` runs Ruff, Black, mypy, documentation links and all maintained config combinations. The full test command includes GPU tests when CUDA and VLM dependencies are present; skipped GPU tests are not GPU evidence. CPU-only CI explicitly excludes GPU/slow tests. OSMesa is its separate GPU-less renderer. Use real GPU access for meaningful model workloads; diagnose a permission/runtime issue instead of silently switching to CPU.

The package check builds an sdist, builds its wheel, installs it in a temporary environment and executes outside the checkout. It reuses caller runtime dependencies but asserts that project modules come from the installed wheel. It compares the complete wheel inventory against `src/` and validates packaged scenes/configs, brick physics, asset linting/export and all five CLI registrations. Docker separately exercises fresh dependency resolution.

## Change discipline

For substantial work, use a local ignored `TODO.md` checklist linked to one phase specification; do not commit completed agent-task checklists. Keep canonical operating knowledge here and completed ordinary source history in Git. Preserve pre-existing changes; work on the explicitly requested branch. The current cleanup works directly on main after the user’s merge and publishes the verified result. The user explicitly approved removing the four obsolete branches; their history was saved in a local bundle. History rewriting and worktree deletion remain outside scope. Create small local commits after coherent validated changes.

Retain tests that protect maintained behavior: native simulation, contracts, support/stability, camera alignment, masks, gradients, processor metadata and package contents. Small model doubles are useful for isolated sequence/mask regressions; they do not replace native GPU integration. Remove tests that merely mirror constants or support retired placeholder behavior.

Do not add a new dependency, abstraction or broad rewrite without a concrete consumer. Update the canonical topic and affected phase when a contract changes. Keep inline comments brief. Optional local `AGENTS.md` is ignored; it is not part of the distributable project.

## CI and publication

The [workflow](../../.github/workflows/ci-cd.yml) defines required lint/format/docs, typing, software/OSMesa simulation, config/import smoke and installed-wheel jobs. Its final success gate checks all of them. CI runs on branch pushes and PRs targeting main. Publishing is gated separately to main, tags or manual dispatch.

Existing publication behavior remains: qualifying main changes/manual dispatch/tags can build and publish the dependency container; tags can produce an Apptainer artifact and GitHub release. This task has performed none of those remote actions. Local validation does not establish publication or a green hosted run.

## Source layout

Install the package in editable mode before running tests or scripts. When updating an old checkout, remove obsolete generated root-level `vla_lego.egg-info` before reinstalling; current metadata belongs under `src/`. All runtime imports begin with `vla_lego`; generic top-level `sim`, `models`, `tracking` and `configs` packages have been removed. Hydra uses `initialize_config_module(config_module="vla_lego.configs")`, and assets resolve through `vla_lego.sim.asset_loader`. CLI names remain unchanged.

Tests are grouped under `tests/sim`, `tests/models` and `tests/tracking`, with shared setup in `tests/conftest.py` and package/config checks at the test root. Pytest uses importlib mode, so test collection does not make application code importable from the repository root. Development utilities live in `scripts/`; SLURM operations remain in `infra/gilbreth`. Use `make test` and `make package` for the full suite and installed-wheel probe.

## Root files

The tracked root contains only `README.md`, `LICENSE`, `pyproject.toml`, `Makefile`, `.gitignore` and `.pre-commit-config.yaml`. These are conventional entry points for readers, Python packaging, developer commands and Git/pre-commit tooling. Container definitions live under `infra/containers`; canonical contribution/setup instructions live here. Python compatibility is declared in `pyproject.toml`, so an unused patch-specific `.python-version` is unnecessary. Environment variables are documented beside their actual workflows; there is no automatically loaded `.env` template. Local `AGENTS.md` and `TODO.md` remain ignored.
