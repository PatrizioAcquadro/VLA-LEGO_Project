# Observation and action contracts

The new Alex003 interface below is a design direction, not implemented dimensions or a compatibility promise. Current source constants remain authoritative for the existing MuJoCo/Qwen path. Migration must update real consumers together; do not reinterpret legacy tensors as Alex003 commands.

## Planned Alex003 contract

| Boundary | Design direction |
|---|---|
| Policy observations | Head ZED RGB, timestamped real joint/gripper proprioception, measured or kinematically derived tool poses, and the language instruction. Use only signals obtainable on the robot at decision time. |
| Additional sensing | Stereo/depth only if explicitly supported by the selected model and onboard budget. Wrist cameras only if measured occlusion limits justify the hardware and inference cost. No mandatory four-view rig. |
| Policy actions | Coordinated left/right end-effector motion and gripper opening targets. Choose absolute versus relative poses, rotation representation, reference frame and units with the selected model and hardware controller. |
| Low-level realization | IK and bounded robot control enforce joint, velocity, effort and collision constraints. Define the role of any torso motion explicitly; no inherited spine action or extra degree of freedom is assumed. |
| Gripper semantics | Map opening/closing commands and available feedback to actual WSG32 behavior; do not assume EZGripper normalization, effort sensing or hardware support carries over. |
| Timing | Separate capture, inference/replanning and command application rates. Declare observation age, predicted horizon, executed prefix, interpolation, stale-command handling and recovery from missed deadlines. |
| Normalization | Specify joint/tool ordering, frame conventions and scaling; fit learned statistics only on training data. No fixed 17-D action, 52-D state or 16-step chunk requirement. |
| Outcome/stop | Represent completion and justified inability to continue through an explicit deployable mechanism. A hardware stop enforces limits but does not supply hidden task decisions. |

The VLA owns piece selection, sequence, cooperation and recovery. A low-level controller may reject or constrain a motion for physical reasons, but may not replace the policy's task choice. Keep requested and actually applied commands distinguishable in data and reporting.

### Provisional SmolVLA pilot decision

The first pilot uses **16 named absolute position targets**: seven left-arm joints, seven right-arm joints, then the left/right WSG32 leader opening in meters. Joint order is `vla_lego.isaac.environment.ARM_JOINTS` followed by `JAW_JOINTS`; arm units are radians, opening limits are `[0,0.034]` m. There is no spine/torso action. This is an evidence-driven provisional alternative to the Cartesian direction above, not a reuse of the legacy 17-D tensor.

The contact-only reference executes these targets and has verified grasp/transport/release. Three attempted conversions to an EEF-only differential interface failed near straight-arm/limit configurations, despite native FK/Jacobian parity. Direct joint targets preserve the physically executed control interface and match SmolVLA's supported joint-action adaptation path; no extra task planner supplies autonomous choices. All attempted Cartesian results remain in `outputs/blackwell/grasp-007` through `grasp-009`; source variants are retained in ignored experiment outputs.

The initial sensor contract remains head RGB plus 18 named arm/neck/WSG leader positions and velocities, excluding simulated followers and object truth. The model smoke consumed positions only. Selection, normalization, model input feature subset, explicit deployable completion/abstention outputs and final timing/chunk execution remain open until the pilot. Fit statistics only on training data. Keep geometric EEF planning in the collection teacher and separate evaluation channels. If the joint-action candidate cannot meet development behavior, reconsider representation using that failure evidence; do not declare this pilot choice task-qualified.

Finalize this contract in Phase 2 using the minimum scene and Phase 3 pilot before mass collection. Reproduce real observation availability and latency in simulation; missing hardware signals are not filled with simulator truth. The [data strategy](data.md) owns synchronization, source mapping and leakage prevention.

Phase 2 first supplies a provisional contract for pilot collection, then records concrete values for the final model. Required decisions are the ordered available signals, units/reference frames, pose/action meaning, gripper convention, observation history, normalization, predicted/executed horizon and the capture/replanning/command rates. These values are not left to different consumers to guess. A changed contract requires reconciling earlier pilot data or recollecting it.

Task progress or memory supplied to the policy must derive from its permitted past observations/actions; teacher stage indices and evaluator success labels are not deployable state. Select an explicit observation-based mechanism for justified stop/abstention in Phase 2, distinct from the controller's hard-limit and stale-command stops. The evaluator may terminate a scored episode after the required dwell, but does not choose corrective actions or supply progress to the controller.

## Existing MuJoCo and Qwen contract

The sections below describe the retained implementation only. They remain useful while the replacement is built and do not constrain its design.

### Robot and control

Alex has a fixed base at world `(0,0,1)` m, 15 spine/arm joints and eight coupled finger joints, with 17 actuators. World axes are X forward, Y left, Z up. Quaternions use MuJoCo `wxyz`; tool poses are in the world frame. Source asset provenance remains [beside the robot](../../src/vla_lego/sim/assets/robots/alex/PROVENANCE.md).

The normalized action is 17-D: spine; seven left-arm joints; seven right-arm joints; left gripper; right gripper. Arm values lie in `[-1,1]` and represent scaled joint increments; grippers use `[0,1]`, closed to open. Canonical arm ordering follows `ARM_ACTUATOR_NAMES` in `src/vla_lego/sim/action_space.py`: spine, shoulder Y/X/Z, elbow Y, wrist Z/X, gripper Z for each arm.

`AlexActionSpace` maps commands to position-actuator targets with joint and velocity bounds. `SimRunner.step()` advances one 20-Hz control tick: 25 physics substeps at 0.002 s in the default scene. The MJCF enforces actuator effort limits. These simulation constraints are not a certified hardware safety controller.

### State

| Slice | Quantity | Units / frame |
|---|---|---|
| `0:15` | Spine and arm position | rad |
| `15:30` | Joint velocity | rad/s |
| `30:32` | Left/right gripper opening | `[0,1]` |
| `32:35`, `35:39` | Left tool position / quaternion | m world / `wxyz` |
| `39:42`, `42:46` | Right tool position / quaternion | m world / `wxyz` |
| `46:49`, `49:52` | Left/right tool linear velocity | m/s world |

`AlexRobotState` provides structured, flat and normalized forms. Normalize using its actual joint limits and declared workspace/velocity ranges; do not mix raw and normalized tensors. Workspace bounds and velocity normalization ceilings are scaling conventions, not physical safety limits.

### Cameras and timing

Canonical ordered views are `overhead`, `left_wrist_cam`, `right_wrist_cam`, `third_person`. Each defaults to 320×320 at 20 Hz. `robot_cam` remains a debug camera. RGB is `uint8 (H,W,3)`; depth/segmentation are available for diagnostics and planned auxiliary uses, and must not become undeclared policy inputs.

`MultiViewRenderer.capture(data, step_index=...)` captures all views from one simulation state, with live camera extrinsics for wrist views. A recorder must preserve timestamps and the distinction between `observation_t`, applied `action_t` and `observation_t+1`.

### Model tensors

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

### Task truth

Brick poses, support contacts and velocities are evaluator data. [Brick evaluation](bricks.md) advances physics over a dwell interval; callers must account for that time advance. Layout resets are diagnostic initial conditions, not actions, demonstrations or learned outcomes.
