# Development

## Environments and installation

The portable package supports Python 3.10+. Create a dedicated environment and install from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Core dependencies are NumPy, Pillow and PyTorch. `ci` provides checks/build tools, `reporting` provides the Matplotlib dependency already used by offline reports, and `dev` includes both plus pre-commit. Dependency declarations live in [pyproject.toml](../../pyproject.toml). The validated native stacks use PyTorch 2.10 for Isaac and 2.11 for LeRobot; installing the core package does not provision either application stack. [Compute](cluster.md) owns their revisions and separation.

## Checks

```bash
make check
make test
make package
```

`make check` runs Ruff, Black, mypy and local Markdown-link checks. Set `PYTHON=.venv/bin/python` when the environment is not activated. In a sandbox that blocks formatter workers, run the same check with host access; `BLACK_NUM_WORKERS=1` limits formatting parallelism.

The maintained suite protects recording masks, train/development separation, control/planning constraints, task decisions, evaluation accounting, checkpoint retention and offline reporting. Small doubles isolate these contracts; they do not establish physical execution or model behavior. Lossless RGB roundtrip tests exercise FFmpeg and PyAV independently when available; missing codec backends are reported as skips. Meaningful model/renderer checks use the actual GPU and compatible native environment.

The package check builds an sdist, builds its wheel, installs outside the checkout and compares the complete runtime/asset inventory against `src/`. It checks installed imports, block geometry, licenses, task construction and maintained module help. It reuses caller runtime dependencies, preserves the interpreter build and forces installation even when the parent has the same package version. It does not claim fresh dependency resolution or native simulation qualification.

## Layout and changes

`src/vla_lego/isaac` owns the task runtime; `pretrained` owns policy/data integration. Native modules are loaded after Isaac AppLauncher. Scripts contain launch/report/check utilities; tests remain under `tests/isaac`. All application imports use the `vla_lego` namespace.

Use an ignored local `TODO.md` checklist for substantial work. Preserve pre-existing changes and stage only owned changes. Commit coherent validated milestones; push or history rewriting requires authorization. Keep raw data, weights, environments, caches, credentials and assistant state out of Git.

Retain tests for maintained behavior and important regressions. Remove exclusive tests with a retired capability; avoid test-only production switches, speculative abstractions and duplicate documentation. Keep one canonical page per topic and update the affected phase. Comments should explain units, frames, safety meaning or non-obvious behavior briefly.

## CI

The [workflow](../../.github/workflows/ci-cd.yml) runs lint/docs, typing, software contracts and installed-wheel checks, with a final `CI Success` gate. Hosted CPU jobs do not qualify Isaac rendering or pretrained GPU execution. Container publication, Apptainer builds and release jobs belonged to the retired v1 stack and are removed. Hosted status is verified against the actual pushed commit in [GitHub Actions](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/actions/workflows/ci-cd.yml).
