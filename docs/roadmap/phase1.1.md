# Phase 1.1 — Robot Model Integration (4 days)
**Goal:** Deliver a stable, reproducible, and Alex-compatible *upper-body fixed-base* robot model in MuJoCo, with functional bimanual end-effectors (SAKE EZGripper), a finalized action/state contract for VLA training, and a kinematics validation report that proves compatibility with the target Alex configuration.

# Repo
https://github.com/ihmcrobotics/ihmc-alex-sdk

# File path inside repo
alex-models/alex_V1_description/mjcf/alex_v1_full_body_nub_forearms_mjx.xml

**Fixed decisions (frozen for 1.1+):**
- **Sim engine:** MuJoCo (MJCF-first workflow; code-first + viewer for debugging).
- **Robot asset:** Existing **IHMC Alex** model from `ihmc-alex-sdk` (no proxy).
- **Scope:** **Upper-body fixed-base** (no locomotion/floating base in 1.x).
- **Alex compatibility target:** **Level 2+** (full kinematic compatibility + realistic limits/saturations; partial dynamics realism where impactful).
- **End-effectors:** **SAKE EZGripper** (1D grasp signal) initially; abstraction layer for future PSYONIC Ability Hand swap.
- **Action space:** **Δq joint deltas** for bimanual arms + gripper commands.
- **Robot state:** Core required + recommended kinematic signals (see Section 5).
- **Views:** **2 cameras from day 1** (robot camera + third-person).

---

## 1) Import/Create Bimanual Robot URDF/MJCF ✅
### What we will do
Integrate the existing IHMC Alex V1 asset into MuJoCo as an *upper-body fixed-base* bimanual robot, starting from the existing MJCF files in `ihmc-alex-sdk`. Strip lower-body joints and fix the head, keeping the torso rotation active.

### Why this matters
All downstream phases (multi-view data generation, action learning, sim-to-real) depend on a single “source of truth” robot model that is stable, versioned, and structurally consistent with Alex.

### Execution checklist
- **Asset ingestion**
  - ✅ Identify and pin a single authoritative Alex V1 model version (commit hash / release tag) from `ihmc-alex-sdk`. → `be25a395e35238bc6385a58bcc50aa047d936a25`
  - ✅ Starting model: `alex_v1_full_body_mjx.xml` (full-body variant with wrist joints, NOT nub_forearms). EZGripper will be added in Section 3.
  - ✅ Strip lower-body joints (12 joints: LEFT/RIGHT_HIP_X/Z/Y, KNEE_Y, ANKLE_Y/X).
  - ✅ Fix NECK_Z and NECK_Y joints at a default head pose (head camera is static).
  - ✅ Keep **SPINE_Z** (torso rotation) active — useful for bimanual reach.
- **Upper-body fixed-base setup**
  - ✅ Define a fixed root/body for the torso (no floating base). Base at z=1.0m.
  - ✅ Confirm base frame orientation conventions (world Z-up, base frame at pelvis).
- **Geometry & collision strategy**
  - ✅ Keep **visual meshes** for realism/debug (group 1, contype=0).
  - ✅ Use **simplified collision geoms** (capsules/boxes/sphere, group 3) for stability.
  - ✅ Define/verify self-collision policy (adjacent pairs excluded; cross-arm enabled).
- **Frames & naming contract**
  - ✅ Freeze canonical names for Alex joints (lowercase MuJoCo convention):
    - base_link/torso frame, spine_z
    - left arm: left_shoulder_y/x/z, left_elbow_y, left_wrist_z/x, left_gripper_z
    - right arm: right_shoulder_y/x/z, right_elbow_y, right_wrist_z/x, right_gripper_z
    - ✅ left/right end-effector sites (left_ee_site, right_ee_site)
    - ✅ camera mount: robot_cam (head-mounted), overhead + third_person (scene-level)
  - Quaternion convention: MuJoCo default (w, x, y, z). EE quaternions available via `framequat` sensors.

### Milestone (minimum success criteria)
- ✅ Robot model loads in MuJoCo reliably (no missing assets), runs **10 seconds** without catastrophic instability, and has frozen joint ordering + named EE frames suitable for dataset logging.

### Implementation notes
- Source MJCF used SDK row-major `fullinertia` format (Ixx,Ixy,Ixz,Iyy,Iyz,Izz); converted to MuJoCo format (Ixx,Iyy,Izz,Ixy,Ixz,Iyz) with `balanceinertia=”true”` for triangle inequality compliance.
- GRIPPER_Z joints added (exist in URDF but absent from source MJCF).
- Mesh paths are explicit relative (`meshes/filename.obj`) for `<include>` compatibility; scene file sets `<compiler meshdir>` to resolve correctly.

---

## 2) Configure Joint Limits & Dynamics Parameters (Level 2+ realism) ✅
### What we will do
Set joint limits, velocity limits, actuator constraints, and minimal-but-impactful dynamic parameters (damping/friction/armature) to make the robot numerically stable and physically plausible for contact-rich LEGO manipulation.

### Why this matters
LEGO assembly is sensitive to small instabilities. Incorrect limits or poorly conditioned dynamics cause jitter, penetration, unrealistic contacts, and non-transferable policies.

### Execution checklist
- **Joint limits (kinematic correctness)**
  - ✅ Source limits directly from Alex V1 URDF (already contains measured values for effort, velocity, and range per joint). Cross-reference with `hardware/` specs in `ihmc-alex-sdk`.
  - ✅ Populate min/max joint ranges for SPINE_Z + both arms (verify sign conventions left vs right).
  - ✅ Validate that the “home pose” is consistent and collision-free.
  - ✅ Add soft safety clamps in the control pipeline (separate from physics constraints). → `sim/control.py`
- **Velocity/acceleration constraints (realism + safety)**
  - ✅ Define joint velocity limits from URDF `velocity` attribute (and an implied acceleration limit via smoothing/rate limiting). → `JOINT_VELOCITY_LIMITS` in `sim/control.py`
  - ✅ Enforce rate limits for commanded targets (avoid discontinuous jumps). → `AlexController.apply()` in `sim/control.py`
- **Actuation constraints (Level 2+ dynamic realism)**
  - ✅ Define actuator `ctrlrange` and plausible saturation behavior, sourcing effort limits from URDF `effort` attribute. → `inheritrange=”1”` (clamp to joint range)
  - ✅ Introduce conservative torque/effort caps (even if policy outputs Δq). → `forcerange` on all actuators
- **Numerical stabilization**
  - ✅ Use damping/friction values from URDF as starting point (Alex URDF specifies per-joint damping). Per-joint-group tiered values.
  - ✅ Use armature / inertial stabilization parameters if needed. → `armature=”0.01”` on all joints incl. spine.
  - ✅ Ensure contact solver settings are stable for high-friction EZGripper fingertip contact. → `iterations=50`, `implicitfast` integrator, `solref`/`solimp` on collision geoms.
- **Sanity tests**
  - ✅ Drop/settle test in neutral pose (no explosions).
  - ✅ Hold-pose test under PD (no persistent drift/jitter). → `TestAlexHoldPose`
  - ✅ Joint sweep test (each joint traverses part of its range stably). → `TestAlexJointSweep`

### Milestone (minimum success criteria)
- ✅ Robot remains stable under gravity and simple PD holding.
- ✅ Joint motion respects limits and saturations.
- ✅ No persistent high-frequency jitter in a static pose.
- ✅ Joint sweep passes without self-collision blow-ups.

### Implementation notes
- **Integrator**: `implicitfast` (unconditionally stable, handles damping implicitly). Replaced Euler with `eulerdamp=”disable”`.
- **Solver**: 50 iterations, 10 ls_iterations (up from 3/5) for contact-rich stability.
- **Per-joint damping**: Proximal joints (spine, shoulders) = 2.0 Ns/rad, mid-chain (shoulder_z, elbow) = 1.5, distal (wrists) = 0.5, gripper = 0.3.
- **Contact params**: `solref=”0.005 1.0”` (critically damped, 5ms), `solimp=”0.9 0.95 0.001 0.5 2”` (near-rigid).
- **URDF velocity limits**: spine/shoulder_y/x = 9.0 rad/s, shoulder_z/elbow = 11.5, wrist/gripper = 25.0.
- **URDF effort limits**: spine/shoulder_y = 150 N·m, shoulder_x = 100 (conservative), shoulder_z/elbow = 80, wrist/gripper = 20.
- **Right arm URDF**: Has placeholder effort/velocity values (1000/100); left arm values used for both sides.
- **Control pipeline**: `sim/control.py` provides `AlexController` with rate limiting at 80% of hardware velocity limit per timestep.

---

## 3) Set Up End-Effector Models (SAKE EZGripper + Abstraction Layer)
### What we will do
Integrate **SAKE EZGripper** end-effectors for both arms into the Alex MuJoCo model, starting from the existing "nub forearms" MJCF and the EZGripper URDF adapters in `ihmc-alex-sdk`. Design an abstracted command interface that can later be swapped for the **PSYONIC Ability Hand** (6-DoF dexterous hand) without changing the upstream action contract.

### Why this matters
Without reliable grasping, you cannot generate meaningful LEGO datasets or demonstrate bimanual assembly. Using the real Alex end-effectors (EZGripper) from day one ensures sim-to-real alignment and avoids training on a gripper geometry that doesn't exist on the real robot.

### Execution checklist
- **EZGripper integration into MJCF**
  - Convert EZGripper URDF adapter (`alex_v1.leftEZGripperAdapter.urdf` / `alex_v1.rightEZGripperAdapter.urdf`) to MJCF elements.
  - Attach to the "nub forearm" model, replacing the nub termination with EZGripper geometry.
  - Ensure left/right end-effectors share the same semantics and scaling.
- **Contact geometry**
  - Use collision primitives for EZGripper finger pads (30mm × 50mm pads; avoid mesh collisions).
  - Set friction/contact parameters to support non-slipping grasps, accounting for EZGripper's underactuated compliance.
- **Command interface (frozen)**
  - Define `gripper_cmd ∈ [0, 1]` per arm: 0 = closed, 1 = open (maps to EZGripper's Dynamixel servo 0–180° range).
  - Define optional `gripper_force_limit` parameter (may be static in 1.1).
  - Freeze a `tool_frame` per end-effector for EE pose logging.
- **Abstraction layer (future-proofing)**
  - Design an `EndEffectorInterface` abstraction so the Ability Hand (6-DoF, 5 individually actuated fingers) can replace EZGripper later without changing the upstream action/state contracts.
  - Document the mapping: EZGripper uses 1D grasp signal; Ability Hand would expand to 6+ DoF (future Phase).
- **Grasp sanity tests**
  - Grasp a simple cube/brick proxy reliably with EZGripper geometry.
  - Lift and hold object for several seconds without explosive contacts.
  - Repeat open/close cycles without accumulating instability.

### Milestone (minimum success criteria)
- Both EZGrippers open/close stably and can execute a repeatable grasp+lift on a simple object in sim.
- End-effector command semantics and tool frames are frozen.
- Abstraction layer documented for future Ability Hand integration.

---

## 4) Verify Kinematics Match Target (Alex-Compatible) — Validation Report
### What we will do
Produce a **Kinematics Validation Report** demonstrating that the integrated model matches the target Alex kinematics (Level 2), and documenting any residual differences and their impact.

### Why this matters
This is your proof that the sim robot is not a “proxy”. It protects the project from hidden frame/joint mismatches that would break sim-to-real or invalidate evaluation.

### Execution checklist
- **Define the kinematic reference**
  - Use the pinned Alex model definition as the reference (from `ihmc-alex-sdk`).
  - Freeze EE frame definitions used for comparison.
- **FK consistency tests**
  - Sample N (e.g., 50–200) random joint configurations within limits for each arm.
  - Compute forward kinematics for both arms:
    - EE position error (cm)
    - EE orientation error (degrees)
- **Workspace sanity**
  - Verify workspace overlap and reachability for typical LEGO workspace placements.
  - Check for mirrored-axis mistakes (left vs right) using known symmetric poses.
- **Documentation**
  - Include:
    - joint ordering table
    - frame convention description
    - error distributions (mean/max)
    - list of known deviations (if any) and mitigation notes

### Milestone (minimum success criteria)
- Report exists and is reproducible.
- FK tests show no gross mismatch (no axis flips, no large systematic offsets).
- Clear “Go/No-Go” statement: kinematics are acceptable for dataset generation and policy training.

---

## 5) Action Space & Low-Level Control Contract (Δq + PD/Impedance)
### What we will do
Freeze the action vector definition and the low-level control strategy that converts model outputs into stable joint motion under MuJoCo physics.

### Why this matters
Your entire learning system (dataset, policy output head, chunking, evaluation) depends on an unambiguous action contract. If this changes later, you risk invalidating collected data and learned models.

### Execution checklist
- **Action vector definition (frozen)**
  - `action = [Δq_spine(1), Δq_left(7), Δq_right(7), gripper_left(1), gripper_right(1)]` → **17-D**
  - Alex joint ordering per arm (7 DoF): SHOULDER_Y, SHOULDER_X, SHOULDER_Z, ELBOW_Y, WRIST_Z, WRIST_X, GRIPPER_Z
  - Normalize action to a fixed range (e.g., [-1, 1]) and map to Δq via `Δq_max` per joint.
- **Control rate and substepping**
  - Policy/control rate: **20 Hz** (one action per 50 ms).
  - Physics substeps: run higher-frequency integration (e.g., 5–10 substeps per action).
- **Low-level controller**
  - Joint-space PD / impedance tracking to convert target positions into stable actuation.
  - Safety clamps:
    - joint limit clamps
    - max velocity clamps
    - rate limiting (avoid abrupt target jumps)
- **Stability tests**
  - Apply random bounded actions and ensure no explosions.
  - Verify smooth motion under repeated action chunks (h=16) without drift.

### Milestone (minimum success criteria)
- Action contract is frozen and documented.
- Executing sequences of actions results in stable, bounded, repeatable motion.
- Rate/substep settings yield stable contacts (no consistent jitter).

---

## 6) Robot State Contract (Core + Recommended Kinematic Signals)
### What we will do
Define the robot state vector that will be logged and used as input for the VLA policy, including both core proprioception and recommended kinematic signals for manipulation and debugging.

### Why this matters
A stable and informative state representation improves learning efficiency, supports precision assembly, and enables rigorous debugging and evaluation.

### State definition (frozen for dataset + training)
**Core (required):**
- Joint positions: `q` (SPINE_Z + both arms)
- Joint velocities: `q_dot` (SPINE_Z + both arms)
- Gripper state: `gripper_width` or `gripper_cmd` (left/right)

**Recommended (included as requested):**
- End-effector pose for each arm: `EE_pos (x,y,z)` + `EE_quat`
- End-effector velocity for each arm: `EE_lin_vel` (+ optionally `EE_ang_vel`)

**Implementation/consistency rules:**
- All poses/velocities must specify the reference frame (e.g., base/torso frame).
- Quaternion ordering convention is documented and enforced consistently.
- State normalization ranges are documented (clip ranges and scaling).

### Execution checklist
- Implement extraction of q, q̇ (SPINE_Z + both arms), gripper signals each step.
- Implement FK-based EE pose computation each step.
- Implement EE velocity computation (finite differences or simulator-provided).
- Validate state consistency:
  - EE pose changes match joint motion qualitatively
  - no discontinuities in quaternion outputs
- Document the complete state schema (names, units, shapes, frames).

### Milestone (minimum success criteria)
- State vector is stable and consistent across runs.
- State fields are logged and can be replayed/validated.
- State schema is frozen and ready for dataset generation.

---

## 7) Multi-View Cameras (2 Views from Day 1)
### What we will do
Set up two synchronized camera streams in MuJoCo for data generation and policy inputs, aligned with the LEGO assembly task requirements.

### Why this matters
Bimanual LEGO assembly requires both an ego-centric robot perspective and an external observation view. Multi-view inputs strengthen generalization and make demos more compelling for industrial evaluation.

### View specification (frozen)
1. **Robot Camera** — mounted on Alex's head (NECK_Z/NECK_Y fixed at default pose), providing the robot's own perspective of the workspace
2. **Third-Person Camera** — external camera looking directly at the robot for observation, debugging, and presentation

### Execution checklist
- Define camera frames and mounts in MJCF (head-mounted with fixed neck, third-person).
- Set default NECK_Z/NECK_Y angles for optimal workspace framing.
- Choose initial resolution and capture rate (aligned with policy rate unless constrained).
- Ensure synchronization (both cameras correspond to the same sim timestep).
- Validate:
  - robot camera moves correctly with the robot
  - third-person provides stable framing of the full workspace and robot
- Log camera intrinsics/extrinsics metadata if available/needed for reproducibility.

### Milestone (minimum success criteria)
- Both camera streams render reliably (headless-capable) and are time-aligned.
- Robot camera tracks correctly with no frame drift.
- Camera configuration is frozen for dataset logging.

---

# Final Definition of Done (Phase 1.1)
Phase 1.1 is complete when:
- The **Alex upper-body fixed-base MJCF** loads and simulates stably.
- **Joint limits + Level 2+ constraints** are configured and validated.
- **Bimanual SAKE EZGripper end-effectors** work reliably for basic grasp+lift.
- The **action contract (17-D: Δq spine + bimanual arms + gripper)** is frozen and stable under repeated action chunks.
- The **robot state contract** includes core proprioception + EE pose/velocity and is consistent.
- **Two cameras** (robot camera, third-person) render reliably and synchronously.
- A **Kinematics Validation Report** exists with FK-based evidence of Alex compatibility.
- **End-effector abstraction layer** documented for future PSYONIC Ability Hand integration.

---

# Notes / Risks (explicit)
- LEGO contact richness can cause solver instability; prioritize collision simplification and stable solver settings early.
- Kinematic mismatches (axis flips, wrong EE frame) are the #1 silent failure mode; validation report is mandatory before dataset generation.
- Do not expand to full-body locomotion in 1.x; keep scope aligned with bimanual tabletop assembly.
