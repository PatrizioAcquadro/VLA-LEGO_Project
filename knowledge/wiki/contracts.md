# Runtime contracts

The source constants and packaged configuration are authoritative. Changes to these interfaces require updating their actual consumers and meaningful regression tests together.

## Robot and control

Alex has a fixed base at world `(0,0,1)` m, 15 spine/arm joints and eight coupled finger joints, with 17 actuators. World axes are X forward, Y left, Z up. Quaternions use MuJoCo `wxyz`; tool poses are in the world frame. Source asset provenance remains [beside the robot](../../sim/assets/robots/alex/PROVENANCE.md).

The normalized action is 17-D: spine; seven left-arm joints; seven right-arm joints; left gripper; right gripper. Arm values lie in `[-1,1]` and represent scaled joint increments; grippers use `[0,1]`, closed to open. Canonical arm ordering follows `ARM_ACTUATOR_NAMES` in `sim/action_space.py`: spine, shoulder Y/X/Z, elbow Y, wrist Z/X, gripper Z for each arm.

`AlexActionSpace` maps commands to position-actuator targets with joint and velocity bounds. `SimRunner.step()` advances one 20-Hz control tick: 25 physics substeps at 0.002 s in the default scene. The MJCF enforces actuator effort limits. These simulation constraints are not a certified hardware safety controller.

## State

| Slice | Quantity | Units / frame |
|---|---|---|
| `0:15` | Spine and arm position | rad |
| `15:30` | Joint velocity | rad/s |
| `30:32` | Left/right gripper opening | `[0,1]` |
| `32:35`, `35:39` | Left tool position / quaternion | m world / `wxyz` |
| `39:42`, `42:46` | Right tool position / quaternion | m world / `wxyz` |
| `46:49`, `49:52` | Left/right tool linear velocity | m/s world |

`AlexRobotState` provides structured, flat and normalized forms. Normalize using its actual joint limits and declared workspace/velocity ranges; do not mix raw and normalized tensors. Workspace bounds and velocity normalization ceilings are scaling conventions, not physical safety limits.

## Cameras and timing

Canonical ordered views are `overhead`, `left_wrist_cam`, `right_wrist_cam`, `third_person`. Each defaults to 320×320 at 20 Hz. `robot_cam` remains a debug camera. RGB is `uint8 (H,W,3)`; depth/segmentation are available for diagnostics and planned auxiliary uses, and must not become undeclared policy inputs.

`MultiViewRenderer.capture(data, step_index=...)` captures all views from one simulation state, with live camera extrinsics for wrist views. A recorder must preserve timestamps and the distinction between `observation_t`, applied `action_t` and `observation_t+1`.

## Model tensors

| Quantity | Shape / convention |
|---|---|
| Token IDs / attention mask | `(B,S_text)` |
| Raw images | Ordered RGB views, processed by the native Qwen processor |
| Image metadata | Preserve `image_grid_thw` and processor token-type metadata |
| Robot state | `(B,52)` for a single observation; temporal assembly is future work |
| Action chunk | `(B,16,17)`; 0.8 s at 20 Hz |
| Hidden states | `(B,S,H)`, current Qwen hidden size 2560 |
| Training result | `total_loss`, `text_loss`, `action_loss` scalars |

Flow matching uses `t=0` noise and `t=1` data, `x_t=(1-t)noise+t*action`, target velocity `action-noise`. This is the opposite time direction of the historical EO-1 example. Missing action tails require explicit valid-step masks. Text targets use `-100` for ignored positions and causal shifting on the full sequence.

## Task truth

Brick poses, support contacts and velocities are evaluator data. [Brick evaluation](bricks.md) advances physics over a dwell interval; callers must account for that time advance. Layout resets are diagnostic initial conditions, not actions, demonstrations or learned outcomes.
