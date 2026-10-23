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

`sim/` owns simulation and bricks; `models/` owns VLM/VLA components and losses; `tracking/` provides optional experiment integration; `configs/` contains packaged configuration; `scripts/` and `infra/` provide operator workflows; `tests/` and `tools/` validate maintained behavior and packaging. [Knowledge](knowledge/wiki/index.md) is canonical; Git preserves ordinary source history.

[MIT license](LICENSE). Robot asset sources and modifications are recorded in [provenance](sim/assets/robots/alex/PROVENANCE.md).
