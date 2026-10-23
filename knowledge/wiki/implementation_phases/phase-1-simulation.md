# Phase 1 — Simulation environment

Scope: original 1.1 and 1.2, adapted to lightweight bricks. Deliver a stable Alex environment and a measurable task definition. Prerequisites: Phase 0 runtime/assets. Canonical behavior is documented in [contracts](../contracts.md), [simulation](../simulation.md) and [bricks](../bricks.md).

## 1.1 Alex robot integration

**State:** maintained robot, grippers, action/state interfaces and camera pipeline implemented. Native tests pass. Independent SDK FK and physical dynamics remain unqualified.

### Seven retained work areas

1. **Robot import:** retain the pinned IHMC Alex V1 upper-body extraction, fixed pelvis and neck, active spine and both arm chains. Preserve visual meshes separately from collision primitives and keep asset provenance next to the model.
2. **Joint limits/dynamics:** source limits from the model, apply position/velocity bounds through the action interface, retain finite effort ranges, and check nominal/perturbed hold and motion. Damping/solver tuning is simulation calibration, not physical identification.
3. **EZGrippers:** maintain the two coupled hands, named tool frames, normalized opening convention and opening/closing/contact behavior. Future end-effectors must implement the maintained interface only when a concrete consumer requires them.
4. **Kinematics:** verify symmetry, active axes, workspace and continuity; resolve the source full-inertia incompatibility before claiming independent SDK parity. Historical self-consistency “GO” did not run that comparison.
5. **Action/control:** retain the actual 17-D normalized command and 20-Hz runner; verify bounds and duration through native actuation. Do not change dimensions to simplify a one-arm pilot.
6. **State:** preserve named 52-D extraction, world frame, quaternion order and declared normalization. Any embodiment mapping must account for the exact slices.
7. **Cameras:** the original two-view plan was superseded by four synchronized views. Preserve wrist-camera extrinsics and the RGB/depth/segmentation boundary.

### Completion and maintenance gates

Changes to the robot/control path require native loading, finite-state stability, motion/constraint and relevant named-interface regressions. Changes to frames or links require kinematics checks. Physical Alex compatibility needs independent reference/hardware evidence; do not infer it from symmetric simulation. The existing grasp-scene tests establish component behavior, not a complete lightweight-brick demonstration.

## 1.2 Lightweight-brick environment

**State:** cuboid catalog, table scene, seeded reset, tower/triangle layouts and contact/stability evaluator implemented. Local rendering and nominal/perturbed physical tests pass. Printed calibration and robot assembly are pending.

### Revised substeps

- **1.2.0 Task/material definition:** use lightweight printed pieces with variable metric dimensions and measured effective mass. Define structures by relative/absolute placement, support and released stability. Real LEGO insertion/retention is deferred.
- **1.2.1 Geometry:** retain simple box collisions and a single `BrickSpec` catalog. Start with the three provisional pieces; add shapes only when a task/controller requires them. Keep simulation asset export separate from print-model generation.
- **1.2.2 Contact:** use native frictional contact without welds, object force assistance or hidden retention. Measure slip, compression and settling with actual prints before fitting material/contact parameters. Distinguish numerical tolerances from empirical variation and hard physical limits.
- **1.2.3 Workspace:** verify reach, table/support geometry, stable footprint and clearance for approaching, lifting and releasing each catalog piece. A visually plausible placement does not establish a reachable grasp.
- **1.2.4 Observation:** maintain the four-view 20-Hz contract and live camera metadata. Record depth/segmentation only in their declared diagnostic/auxiliary role.
- **1.2.5 Reset:** preserve deterministic seeded variation, nonoverlapping initial pieces, bounded sampling, force/state clearing and physical settling. Reject impossible layouts explicitly; never silently resample until a favorable test passes.
- **1.2.6 MVP structure/evaluator:** begin with one placed piece, progress to a stable tower, then three planar triangle locations. Require correct pose, supporting contact, release and dwell stability for every placement. Retain wrong-pose, unsupported, perturbed and held-piece failures.

### Calibration and completion gate

The software baseline is complete when catalog changes compile, reset deterministically, render through the real pipeline and pass meaningful nominal/boundary/perturbed evaluation checks. It has reached that component milestone. Task feasibility additionally requires an actual robot grasp/lift/place/release in Phase 2.1 with no initialization shortcut.

For a physical prototype, measure printed dimensions/mass and confirm gripper clearance before scaling. Calibrate pose/speed/dwell acceptance ranges from meaningful measurement error and task variation; default evaluator tolerances are provisional. Record unsupported structures and grasp failures without adjusting thresholds merely to pass them.

### Handoff

Deliver the catalog, reset seeds, goals, evaluator and frame/control contracts to Phase 2.1. Specify which layout variations are trainable, held out or physically infeasible. Keep both arms available; qualify a one-arm demonstration first and bimanual coordination later.

## Historical change

The original `phase1.2.md` planned studs/tubes, baseplate, press-fit, retention and an MVP-3 LEGO assembly. That implementation is retired. Its important force deficit and proxy distinction are preserved in [history](../history.md); they are not calibration evidence for these bricks. The original 1.1 draft is recoverable from Git. [Phase 8](phase-8-transfer-handoff.md#future-extension-real-lego) defines the separate real-LEGO extension.
