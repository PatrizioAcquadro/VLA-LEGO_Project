# Observation and action contracts

## Current Alex003 contract

The implemented SmolVLA campaign uses the following interface. It is a measured simulation contract, not a finalized physical deployment contract.

| Boundary | Current meaning |
|---|---|
| Image | One head ZED RGB view, 960 × 600, captured with named proprioception before the corresponding applied action |
| State | 18 measured positions: seven left-arm joints, seven right-arm joints, neck Z/Y and two WSG leader openings; append 14 measured arm velocities, for 32 inputs |
| Language | Full English instruction; 96-token runtime window, maximum 51 in the declared campaign sequences |
| Motor output | 16 channels: `ARM_JOINTS` then `JAW_JOINTS` from `vla_lego.isaac.environment`; arm radians and WSG leader openings in meters |
| Action representation | `observation_arm_offset`: all future arm targets are offsets from the single measured arm position at that observation; jaws remain absolute |
| Chunk/timing | Predict 50 targets, execute ten at 20 Hz; native physics uses ten 200-Hz steps per command |
| Normalization | TRAIN-only mean/std, saved processors and explicit `alex_interface.json`; state/action order and representation are checked on reload |
| Decisions | Separate causal temporal classifier with continue/complete/abstain probabilities and persistent terminal decisions |

The worker adds the observation anchor to the whole arm chunk once before serving ordinary absolute joint targets. It does not add a changing live anchor at every future tick. The decoder projects requested positions onto unchanged shared joint limits and applies the same per-tick target-rate limits as the teacher; WSG leader openings stay in `[0, 0.034]` m. Raw predictions, bounded requests and applied targets remain distinguishable.

The separate classifier reuses frozen RGB/language features and the same normalized observed proprioception, sampling at 2 Hz over three seconds of actual past observations. Timestamps select history and are not model features. A new instruction resets history. Decision checkpoint normalization must match the motor runtime. Its executed calibration and final comparison are described in [models](models.md#expanded-corpus-learning-assignment--october-26-2026); implemented interfaces do not establish reliable decisions.

## Policy, teacher and evaluator boundaries

Policy inputs contain deployable images, measured proprioception and instruction. Any policy history derives from permitted past observations/actions. Simulator object poses, identities, contact truth, teacher stages, future frames and evaluator success remain separate teacher/evaluation channels. Masked negative actions may provide context but are not positive motor targets.

The policy owns requested-piece selection, order, cooperation and recovery. Low-level kinematics/control enforce physical limits and stale-command stopping without supplying hidden task decisions. Conditional teacher initialization is explicitly labeled and cannot earn autonomous-from-start credit. The [data guide](data.md#episode-content-and-leakage-boundary) owns recording alignment and supervision roles.

Synchronous workstation inference pauses simulation; measured workstation inference time is not a real-time or Orin qualification. Deployment must declare observation age, replanning rate, executed prefix, interpolation and missed-deadline handling separately.

## Historical read and inference compatibility

Existing checkpoints include 18-position and 32-feature variants, absolute joint targets, the earlier per-tick `measured_arm_offset` experiment, and models with two embedded complete/abstain channels. The worker retains their explicit interface decoding for historical comparisons. An absolute checkpoint without its interface requires the explicit legacy-interface flag; equal tensor shapes never establish equal action semantics.

The current trainer accepts only the motor-only campaign (`--campaign`), with `observation_arm_offset` normalization and a separate classifier. Old synthetic/Qwen training and the superseded embedded-decision trainer are retired; their original sources and outcomes remain in Git and the evidence archives. Historical readers do not recreate obsolete training paths.

## Physical contract still to establish

Confirm actual WSG32 commands/feedback/stopping, available state signals, frames and camera calibration on Purdue Alex. Use only physically available modalities; stereo/depth or wrist cameras require a demonstrated benefit and onboard budget. Decide joint versus Cartesian action representation from measured control and task behavior rather than inherited tensor dimensions. All learned online computation runs on Orin; the NUC retains low-level robot control. [Models](models.md#onboard-feasibility) owns the feasibility gate.
