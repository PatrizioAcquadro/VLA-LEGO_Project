# Simulation workflows

Install the simulation extra as described in [development](development.md). Run commands below from the repository root with that environment activated. Native MuJoCo physics runs on the CPU; EGL rendering uses the GPU. Set `MUJOCO_GL` before importing MuJoCo.

## Inspect and export

```bash
vla-lint-assets
vla-viewer sim/assets/scenes/alex_upper_body.xml
MUJOCO_GL=egl vla-demo --scenario all --output-dir outputs/demo-001
vla-gen-bricks --output-dir outputs/brick-assets-001
```

The viewer needs a desktop/display. Use `--help` for its actual options. `vla-demo` accepts `robot`, `bricks` or `all`, plus `--seed`. It requires a new output directory and streams short MP4 clips, first/last PNG frames and `report.json`. Robot motion is a small scripted joint/gripper diagnostic. Tower and triangle pieces are initialized in their target layouts, then checked under native physics. These are presentation/debug artifacts, not a learned assembly result.

The brick exporter accepts `--config path/to/catalog.yaml` and writes portable individual MJCF files. It exports simulation assets, not STL print models.

## Programmatic use

```python
import numpy as np
from sim.bricks.episode import EpisodeManager
from sim.bricks.task import evaluate_goal, make_goal
from sim.sim_runner import SimRunner

episode = EpisodeManager()
episode.reset(seed=42)
runner = SimRunner(episode.model, episode.data)
action = np.zeros(17)
action[15:] = 1.0  # Open both grippers.
state = runner.step(action)
assert state.to_flat_array().shape == (52,)

goal = make_goal(episode.bricks, "tower")
episode.reset(seed=42, layout=goal)  # Diagnostic initialization only.
result = evaluate_goal(episode, goal)
assert result.success
```

`load_scene("alex_upper_body")` returns an `MjModel`; construct `MjData(model)` separately. Asset paths resolve from the installed package, including external working directories. Preserve the [robot provenance](../../sim/assets/robots/alex/PROVENANCE.md).

## Focused diagnostics

| Command | Purpose |
|---|---|
| `python -m scripts.validate_sim_smoke` | Native headless physics/rendering/export checks |
| `python -m scripts.validate_action_space` | Named 17-D action mapping and constraints |
| `python -m scripts.validate_robot_state` | State extraction/normalization |
| `python -m scripts.validate_cameras` | Four-view output and camera metadata |
| `python -m scripts.validate_kinematics` | Self-consistency and optional SDK reference comparison |

Set `ALEX_SDK_PATH` to a pinned SDK checkout for the independent FK comparison. A skipped comparison is not a pass. These focused tools remain useful operator checks; generic load/asset/render validators and phase-specific artifact generators were consolidated into tests, the linter and `vla-demo`.

If EGL cannot see the GPU, fix device access/runtime permissions before model/rendering checks. On GPU-less CI, OSMesa is an explicit CPU-renderer validation path, not evidence of GPU operation. See [cluster](cluster.md) for containers and remote jobs.
