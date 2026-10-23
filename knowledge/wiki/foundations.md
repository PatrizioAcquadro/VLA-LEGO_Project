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

## TCP control and single-block reference

`vla_lego.isaac.control` composes the shared WSG32 TCP with live link poses, shifts the native link-origin Jacobian to the TCP, and realizes named tool requests with damped least squares and bounded pilot joint speed. `kinematics` derives a seven-joint chain and limits directly from that same URDF for GPU multi-start IK. Native FK parity was about 0.5 micrometers and 1 microradian; seven physically executed small joint probes showed at most about 0.10 mm linearization residual. These validate coordinate/control components, not a collision-aware planner.

The explicit reference probe uses only robot actuation and contacts under gravity. Both arms move to staging poses; the left arm then grasps a loose 40 mm cube, lifts, transports, places, opens and retreats. The reference alone reads block truth. Head RGB and 18 named arm/neck/WSG32 leader signals remain separate; follower simulation joints are excluded from deployable observations.

```bash
bash scripts/isaac-run.sh -m vla_lego.isaac.grasp_probe \
  --output-dir outputs/grasp-001 --cache-dir outputs/converted-assets
```

Five development attempts were retained before this first complete result:

| Attempt | Outcome and observed mechanism |
|---|---|
| `grasp-001` | Aborted at an unsolved near-body fixed-orientation raise, before grasp. |
| `grasp-002` | Both staging poses passed, but joint-linear approach collided with the table edge; pregrasp failed. |
| `grasp-003` | Added high lateral waypoints; all pregrasp poses passed, then the lower target failed bounded IK. |
| `grasp-004` | Moved the loose cube into the lower grasp workspace. Descent stalled against the table; contact/geometry audit identified fingertip extent beyond TCP. |
| `grasp-005` | Complete reference grasp/lift/transport/release. Lift about 98 mm; final position about `(0.4794,0.2318,0.8200)` m, approximately 2 mm from requested XY. All ten samples over 2 s released dwell passed, with zero robot contact and environmental contact about 0.245 N. Wall time to report before app close was 115.6 s. Lift and retreat images were inspected. |

This is **one complete single-block reference out of five development attempts**, not learned success or full Phase 1 completion. Earlier attempts used distinct diagnosed configurations; do not pool them into a qualification rate or discard them. A sixth causal run changed only the leader closure target from 0 to 0.018 m. It also completed the same lift/transport/released dwell, with lift contact about 2.02 N instead of 25.01 N and instantaneous angular speed about 0.21 instead of 3.23 rad/s. This reduces observed simulated contact chatter without changing success gates; broader mass/friction calibration remains necessary. The default reference now uses this reduced preload. Neither force is a physical WSG32 measurement.

Native collider readback measured UMI contact geometry along TCP X from about −48.9 to +71.5 mm, with a 70 mm inner opening. For the top-down 40 mm cube, a TCP 60 mm above object center leaves about 8.5 mm fingertip/table clearance while overlapping about 31.5 mm of the block side. This is task calibration against existing geometry, not a changed shared frame or physical measurement. The evaluator uses provisional component thresholds: 5 mm tool-position admission, 10 mm destination XY, 3 mm supported Z, linear/angular speed below 0.01 m/s and 0.1 rad/s, and numerical-zero robot force over a 2 s dwell. They are not calibrated final campaign tolerances.

Contact readback explicitly activates every nested articulated link. The generic activation helper stops at the first rigid ancestor and missed nested links in this asset. Static-table GPU pair filtering is unsupported; the report retains supported robot contact pairs and the remaining environmental force rather than inventing a measured table-only force. Probe shutdown preserves nonzero exit status on failure. Full traces, reports and failures live in `outputs/blackwell/grasp-*`, `ik-*`, `jacobian-*` and `finger-geometry-*`, mirrored from Blackwell.

## Staged retirement

| Retained path | Active consumer | Replacement required before removal |
|---|---|---|
| MuJoCo runtime and copied Alex V1/EZGripper assets | Existing diagnostics, tests, viewer and packaged demo | Native Isaac task/control/render diagnostics, installed consumption and retained asset attribution |
| Legacy cuboid task/evaluator | MuJoCo diagnostics/tests | Ordered task, distractors, functional cooperation and calibrated released-stability evaluator in Isaac |
| Qwen and custom action head | Existing model diagnostics/configs/tests | Selected robot-pretrained VLA, real adaptation, reload and autonomous evaluation |
| Legacy Docker/cluster templates | Existing documented package/model workflows | Validated separate Isaac and selected-VLA environments on the resources actually used |
| Tracking and shared configuration utilities | Existing diagnostics | Retain while used; remove only genuinely unused consumers |

No legacy runtime is removed merely because a new scene loads. Generated outputs, datasets, checkpoints, environments and local assistant checklists remain ignored. Failed probes and future failed qualification campaigns remain evidence.
