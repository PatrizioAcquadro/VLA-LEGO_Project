# Status

Reviewed on October 23, 2026, against the maintained source and local validation. See [validation](validation.md) for evidence and [plans](implementation_phases/index.md) for completion criteria.

| Capability | Current state |
|---|---|
| Alex fixed-base upper body and two EZGrippers | Implemented in MuJoCo; native stability/control tests pass. Physical dynamics and hardware operation are unqualified. |
| Actions, state, synchronized views | Maintained [17-D / 52-D / four-view contracts](contracts.md). |
| Lightweight bricks | Configurable cuboids, seeded resets, tower and planar triangle goals, support and post-release stability evaluation. |
| Demonstration artifacts | One `vla-demo` workflow exports scripted robot motion and prearranged brick layouts. This does not demonstrate robot assembly. |
| Qwen3.5-4B backbone | Native four-view GPU forward, processor metadata, hidden states and freeze contracts verified locally. |
| VLA head | State/action projectors, flow matching and `(B,16,17)` inference implemented; finite untrained outputs and gradient routing verified. |
| Loss modules | Standalone text/action losses and masks implemented. Combined modular objective and production integration remain incomplete. |
| Data collection, annotations, temporal interleaving | Planned; no maintained collector or qualified brick dataset. |
| Training and learned policy evaluation | Planned; synthetic placeholder trainer/evaluator retired. No learned brick success claim. |
| Packaging and infrastructure | sdist-to-wheel probe and local GPU Docker demo pass; CI definition updated. Hosted CI, Apptainer and Gilbreth/A100 execution not verified in this audit. |
| Physical transfer, LIBERO, few-shot experiments | Not implemented or measured. |

## Next development sequence

1. Calibrate a small printable brick set and verify one physical grasp with the existing action pipeline ([1.2](implementation_phases/phase-1-simulation.md#12-lightweight-brick-environment), [2.1](implementation_phases/phase-2-data.md#21-robot-trajectory-generation)).
2. Record one complete robot pick-and-place episode, then a bounded pilot with failures retained. Do not scale to the historical 10K target before replay and data alignment pass.
3. Implement evidence-grounded annotation and interleaving ([2.2](implementation_phases/phase-2-data.md#22-vlm-based-annotation), [2.3](implementation_phases/phase-2-data.md#23-interleaved-data-construction)); resolve model positional/masking limits before training.
4. Complete [4.1](implementation_phases/phase-4-training-infrastructure.md#41-loss-function-implementation) and a real single-GPU training/resume path in [4.2](implementation_phases/phase-4-training-infrastructure.md#42-distributed-training-setup). Qualify distributed execution only when the single-GPU path is correct.

## Open questions and limits

- Printed mass, friction, grip aperture and contact behavior have not been measured. Default dimensions/masses are provisional, not a qualified manufacturing specification.
- A triangle currently means three separated planar target locations. Continuous triangular frames and non-cuboid pieces require a separate design and evaluator.
- Robot demonstrations need task-space control, collision-aware approach/lift/release and actual gripper contact. Resetting pieces into goals is not a teacher trajectory.
- VLA embedding assembly uses sequential positions for injected vision/state/action embeddings. Native multimodal positional equivalence, temporal layouts and attention leakage need explicit validation.
- Kinematics self-consistency is historical evidence. The independent SDK FK comparison was skipped because the source inertia encoding did not load; it remains open.
- Single-arm manipulation is the initial collection target. Bimanual coordination remains an intended later capability, with both arms retained in the interface.
- The original website still describes LEGO and an old calendar. It is a historical planning source; these repository plans define the current scope.
