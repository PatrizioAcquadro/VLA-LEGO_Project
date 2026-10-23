# Development

## Environment

Use Python 3.10 or newer and a dedicated environment. The validated host uses Python 3.10; the locally tested Docker image uses Python 3.12. GPU models require a compatible NVIDIA driver/PyTorch CUDA runtime. Current dependency ranges and optional groups are defined only in [pyproject.toml](../../pyproject.toml).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m scripts.validate_configs
```

Install `.[sim]` for simulation only, `.[vlm]` for model components, `.[tracking]` for optional W&B, or `.[ci,sim]` for the CPU software/physics suite. `dev` includes these and pre-commit. A host missing Python's venv/ensurepip support needs the matching OS venv package. In this audit the existing Python 3.10 user stack was reused through `--without-pip --system-site-packages`; that is an environment workaround, not an isolated dependency installation.

HF cache convention: `HF_HOME` is the Hugging Face root; `HF_HUB_CACHE` is its `hub` directory. Both are optional. Put large outputs and weights outside Git; `outputs/`, datasets, checkpoints, caches and local assistant state are ignored.

## Checks

```bash
make check
MUJOCO_GL=egl python -m pytest tests -q
python tools/check_install.py
```

`make check` runs Ruff, Black, mypy, documentation links and all maintained config combinations. The full test command includes GPU tests when CUDA and VLM dependencies are present; skipped GPU tests are not GPU evidence. CPU-only CI explicitly excludes GPU/slow tests. OSMesa is its separate GPU-less renderer. Use real GPU access for meaningful model workloads; diagnose a permission/runtime issue instead of silently switching to CPU.

The package check builds an sdist, builds its wheel, installs it in a temporary environment and executes outside the checkout. It reuses caller runtime dependencies but asserts that project modules come from the installed wheel. It validates packaged scenes/configs, brick physics, asset linting/export and all five CLI registrations. Docker separately exercises fresh dependency resolution.

## Change discipline

Use [TODO.md](../../TODO.md) for the current implementation checklist and one linked phase specification. Keep canonical operating knowledge here and completed ordinary source history in Git. Preserve pre-existing changes; work on the explicitly requested branch. For this audit, no push, main merge, history rewrite, or branch/worktree deletion is authorized. Create small local commits after coherent validated changes.

Retain tests that protect maintained behavior: native simulation, contracts, support/stability, camera alignment, masks, gradients, processor metadata and package contents. Small model doubles are useful for isolated sequence/mask regressions; they do not replace native GPU integration. Remove tests that merely mirror constants or support retired placeholder behavior.

Do not add a new dependency, abstraction or broad rewrite without a concrete consumer. Update the canonical topic and affected phase when a contract changes. Keep inline comments brief. Optional local `AGENTS.md` is ignored; it is not part of the distributable project.

## CI and publication

The [workflow](../../.github/workflows/ci-cd.yml) defines required lint/format/docs, typing, software/OSMesa simulation, config/import smoke and installed-wheel jobs. Its final success gate checks all of them. PRs target main; hosted execution has not been triggered during this local task.

Existing publication behavior remains: qualifying main changes/manual dispatch/tags can build and publish the dependency container; tags can produce an Apptainer artifact and GitHub release. This task has performed none of those remote actions. Local validation does not establish publication or a green hosted run.
