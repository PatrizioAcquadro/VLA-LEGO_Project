# Isaac foundations and migration

The Phase 0–6 implementation goal starts from `main` at `7bd389f`, with no pre-existing tracked changes. Only this repository and its shared Alex dependencies are in scope. The fixed `v1.0.0` tag remains the legacy release.

## Ownership and environments

Consume `ihmc_alex_isaaclab` from the shared Alex repository (`9ac4bd4` at initial inspection). Do not copy robot meshes into the task package. The shared package owns the measured pedestal, WSG32/UMI geometry and official Stereolabs head-ZED dependency; this project owns furniture, blocks, observations, task control, data and policy integration.

Use three environments:

- **Simulation:** compatible Isaac Sim 6.0.1 / Isaac Lab PhysX environment, with the shared Alex source and its pinned ZED dependency. `scripts/isaac-run.sh` delegates native bootstrap to Isaac's `python.sh`; it does not install the legacy package's Torch/Transformers requirements into Isaac.
- **Training/inference development:** upstream GR00T N1.7 in its own environment. Its Python 3.12, Torch 2.9 and Transformers 4.57 requirements differ from the retained Qwen environment. Installation/inference/adaptation are qualified separately.
- **Onboard:** unresolved pending actual Orin host/BSP, ZED and robot-service compatibility measurements. Do not upgrade robot software or equate desktop timing with Orin. Simulated timing scenarios may advance under the user's sequence exception, with explicit assumptions.

Blackwell was directly inspected for this goal: RTX PRO 6000 Blackwell Max-Q, 97887 MiB, idle before the first probe; Isaac Sim `6.0.1-rc.7`, Isaac Lab `ffff603e`. Local RTX 4090 was occupied by unrelated workloads, which remain untouched. Blackwell runs live in `/home/pacquadr/projects/vla-lego`; the Alex dependency there is read-only from its existing shared checkout. Conversion caches belong to this project's output directory, avoiding another user's `/tmp/IsaacLab`.

## Native foundation probe

From the checkout, set `ISAAC_SIM_DIR`, `ISAAC_LAB_DIR` and `ALEX_ASSETS_ROOT` to the compatible installations. Optional `VLA_ISAAC_PYTHON` selects an already compatible simulation interpreter.

```bash
bash scripts/isaac-run.sh -m vla_lego.isaac.smoke \
  --output-dir outputs/foundation-001 --cache-dir outputs/converted-assets
```

Use a new output directory for each attempt. The probe writes a separate deployable observation, privileged truth report, head image and scene layer. It checks native GPU tensors, fixed root, a loose block settling under gravity, both WSG32 leader commands and RGB. These are component checks, not a grasp, assembly demonstration or learned result. Table height (0.80 m), 40 mm/25 g cube and friction (0.6/0.5) are provisional diagnostic assumptions, not the admitted printable catalog or physical acceptance calibration.

The low-level joint API currently validates finite positions against the shared URDF limits. Actuator effort/velocity limits are enforced by that shared configuration. Collision-aware Cartesian control, stale-command handling and the finalized VLA interface remain implementation work; do not expose this diagnostic joint API to a physical robot.

## Staged retirement

| Retained path | Active consumer | Replacement required before removal |
|---|---|---|
| MuJoCo runtime and copied Alex V1/EZGripper assets | Existing diagnostics, tests, viewer and packaged demo | Native Isaac task/control/render diagnostics, installed consumption and retained asset attribution |
| Legacy cuboid task/evaluator | MuJoCo diagnostics/tests | Ordered task, distractors, functional cooperation and calibrated released-stability evaluator in Isaac |
| Qwen and custom action head | Existing model diagnostics/configs/tests | Selected robot-pretrained VLA, real adaptation, reload and autonomous evaluation |
| Legacy Docker/cluster templates | Existing documented package/model workflows | Validated separate Isaac and selected-VLA environments on the resources actually used |
| Tracking and shared configuration utilities | Existing diagnostics | Retain while used; remove only genuinely unused consumers |

No legacy runtime is removed merely because a new scene loads. Generated outputs, datasets, checkpoints, environments and local assistant checklists remain ignored. Failed probes and future failed qualification campaigns remain evidence.
