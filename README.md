# Lightweight Brick Assembly VLA

Vision-language-action components for Alex to manipulate lightweight, variable-size bricks into structures. The current repository provides MuJoCo simulation, the robot/control/camera interfaces and Qwen3.5-4B with a flow-matching action head. A trained assembly policy and real demonstration dataset remain future work.

The package name `vla-lego` is retained for continuity. Real LEGO studs, press-fit and retention are a future extension.

- [Project knowledge and operating guides](knowledge/wiki/index.md)
- [Current capabilities, limits and priorities](knowledge/wiki/status.md)
- [Nine implementation plans, covering all 22 original subphases](knowledge/wiki/implementation_phases/index.md)
- [Install and develop](knowledge/wiki/development.md)
- [Run the simulation and diagnostic demo](knowledge/wiki/simulation.md)
- [Validation evidence](knowledge/wiki/validation.md)

## Quick start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[sim]'
vla-lint-assets
MUJOCO_GL=egl vla-demo --scenario all --output-dir outputs/demo-001
```

The demo exports robot motion and prearranged tower/triangle layouts. It is a simulation diagnostic, not robot assembly or learned-policy success. See the guides for GPU/model setup, contracts and remaining development.

## Source layout

```text
src/vla_lego/       Runtime package
  sim/             MuJoCo, control, cameras, bricks and assets
  models/          Qwen/VLA components and losses
  tracking/        Optional experiment integration
  configs/         Packaged Hydra configuration
tests/             Simulation, model and tracking regression suites
scripts/           Diagnostics, quality checks and container utilities
infra/gilbreth/    Scheduler jobs and environment setup
knowledge/         Canonical guides, status and nine phase plans
.github/workflows/ Automated checks and gated publication
```

Install before running tests or scripts. Python imports use `vla_lego.*`; the `vla-*` CLI names are unchanged. `make check`, `make test` and `make package` provide the main development checks. Git preserves ordinary source history.

[MIT license](LICENSE). Robot asset sources and modifications are recorded in [provenance](src/vla_lego/sim/assets/robots/alex/PROVENANCE.md).
