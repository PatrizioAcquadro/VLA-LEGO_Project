# Phase 1.1 — Robot Model Integration (4 days)
**Goal:** Deliver a stable, reproducible, and Alex-compatible *upper-body fixed-base* robot model in MuJoCo, with functional bimanual grippers, a finalized action/state contract for VLA training, and a kinematics validation report that proves compatibility with the target Alex configuration.

**Fixed decisions (frozen for 1.1+):**
- **Sim engine:** MuJoCo (MJCF-first workflow; code-first + viewer for debugging).
- **Robot asset:** Existing **IHMC Alex** model (no proxy).
- **Scope:** **Upper-body fixed-base** (no locomotion/floating base in 1.x).
- **Alex compatibility target:** **Level 2+** (full kinematic compatibility + realistic limits/saturations; partial dynamics realism where impactful).
- **Action space:** **Δq joint deltas** for bimanual arms + gripper commands.
- **Robot state:** Core required + recommended kinematic signals (see Section 5).
- **Views:** **4 cameras from day 1** (overhead + wrist-L + wrist-R + third-person).

---

## 1) Import/Create Bimanual Robot URDF/MJCF
### What we will do
Integrate the existing IHMC Alex asset into MuJoCo as an *upper-body fixed-base* bimanual robot, ensuring it loads reliably, simulates stably, and exposes consistent frames and signals for downstream VLA training.

### Why this matters
All downstream phases (multi-view data generation, action learning, sim-to-real) depend on a single “source of truth” robot model that is stable, versioned, and structurally consistent with Alex.

### Execution checklist
- **Asset ingestion**
  - Identify and pin a single authoritative Alex model version (commit hash / release tag).
  - Convert to **MJCF** if necessary (MJCF-first is the final truth).
  - Remove or disable unused full-body components for 1.x (legs/locomotion) *without breaking arm kinematics*.
- **Upper-body fixed-base setup**
  - Define a fixed root/body for the torso (no floating base).
  - Confirm base frame orientation conventions (world axes, base frame axes).
- **Geometry & collision strategy**
  - Keep **visual meshes** for realism/debug.
  - Use **simplified collision geoms** (capsules/boxes) for stability.
  - Define/verify self-collision policy (enable only where needed; exclude unstable pairs).
- **Frames & naming contract**
  - Freeze canonical names for:
    - base/torso frame
    - left/right arm joint names
    - left/right end-effector frames (tool frames)
    - camera mount frames (workspace, wrists, third-person)
  - Verify quaternion convention is consistent across logging and model usage.

### Milestone (minimum success criteria)
- Robot model loads in MuJoCo reliably (no missing assets), runs **10 seconds** without catastrophic instability, and has frozen joint ordering + named EE frames suitable for dataset logging.

---

## 2) Configure Joint Limits & Dynamics Parameters (Level 2+ realism)
### What we will do
Set joint limits, velocity limits, actuator constraints, and minimal-but-impactful dynamic parameters (damping/friction/armature) to make the robot numerically stable and physically plausible for contact-rich LEGO manipulation.

### Why this matters
LEGO assembly is sensitive to small instabilities. Incorrect limits or poorly conditioned dynamics cause jitter, penetration, unrealistic contacts, and non-transferable policies.

### Execution checklist
- **Joint limits (kinematic correctness)**
  - Populate min/max joint ranges for both arms (verify sign conventions left vs right).
  - Validate that the “home pose” is consistent and collision-free.
  - Add soft safety clamps in the control pipeline (separate from physics constraints).
- **Velocity/acceleration constraints (realism + safety)**
  - Define joint velocity limits (and an implied acceleration limit via smoothing/rate limiting).
  - Enforce rate limits for commanded targets (avoid discontinuous jumps).
- **Actuation constraints (Level 2+ dynamic realism)**
  - Define actuator `ctrlrange` and plausible saturation behavior.
  - Introduce conservative torque/effort caps (even if policy outputs Δq).
- **Numerical stabilization**
  - Add joint damping/friction terms to reduce oscillations and improve settle behavior.
  - Use armature / inertial stabilization parameters if needed.
  - Ensure contact solver settings are stable for high-friction fingertip contact.
- **Sanity tests**
  - Drop/settle test in neutral pose (no explosions).
  - Hold-pose test under PD (no persistent drift/jitter).
  - Joint sweep test (each joint traverses part of its range stably).

### Milestone (minimum success criteria)
- Robot remains stable under gravity and simple PD holding.
- Joint motion respects limits and saturations.
- No persistent high-frequency jitter in a static pose.
- Joint sweep passes without self-collision blow-ups.

---

## 3) Set Up Gripper Models (Parallel Jaw) — Functional Bimanual End-Effectors
### What we will do
Implement **parallel-jaw** grippers (1-DoF open/close) for both arms, with stable contacts and a future-proof command interface that can later map to the Alex real end-effector implementation.

### Why this matters
Without reliable grasping, you cannot generate meaningful LEGO datasets or demonstrate bimanual assembly. A simple, stable gripper beats a complex hand early on.

### Execution checklist
- **Gripper mechanical model**
  - Implement 1-DoF gripper width per arm (open/close).
  - Ensure left/right grippers share the same semantics and scaling.
- **Contact geometry**
  - Use collision primitives for fingertips (avoid mesh collisions).
  - Set friction/contact parameters to support non-slipping grasps (tunable but stable defaults).
- **Command interface (frozen)**
  - Define `gripper_cmd ∈ [0, 1]` per arm: 0 = closed, 1 = open.
  - Define optional `gripper_force_limit` parameter (may be static in 1.1).
  - Freeze a `tool_frame` per gripper for EE pose logging.
- **Grasp sanity tests**
  - Grasp a simple cube/brick proxy reliably.
  - Lift and hold object for several seconds without explosive contacts.
  - Repeat open/close cycles without accumulating instability.

### Milestone (minimum success criteria)
- Both grippers open/close stably and can execute a repeatable grasp+lift on a simple object in sim.
- Gripper command semantics and tool frames are frozen.

---

## 4) Verify Kinematics Match Target (Alex-Compatible) — Validation Report
### What we will do
Produce a **Kinematics Validation Report** demonstrating that the integrated model matches the target Alex kinematics (Level 2), and documenting any residual differences and their impact.

### Why this matters
This is your proof that the sim robot is not a “proxy”. It protects the project from hidden frame/joint mismatches that would break sim-to-real or invalidate evaluation.

### Execution checklist
- **Define the kinematic reference**
  - Use the pinned Alex model definition as the reference (same model family / authoritative source).
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
  - `action = [Δq_left(7), Δq_right(7), gripper_left(1), gripper_right(1)]` → **16-D**
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
- Joint positions: `q` (both arms)
- Joint velocities: `q_dot` (both arms)
- Gripper state: `gripper_width` or `gripper_cmd` (left/right)

**Recommended (included as requested):**
- End-effector pose for each arm: `EE_pos (x,y,z)` + `EE_quat`
- End-effector velocity for each arm: `EE_lin_vel` (+ optionally `EE_ang_vel`)

**Implementation/consistency rules:**
- All poses/velocities must specify the reference frame (e.g., base/torso frame).
- Quaternion ordering convention is documented and enforced consistently.
- State normalization ranges are documented (clip ranges and scaling).

### Execution checklist
- Implement extraction of q, q̇, gripper signals each step.
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

## 7) Multi-View Cameras (4 Views from Day 1)
### What we will do
Set up four synchronized camera streams in MuJoCo for data generation and policy inputs, aligned with the LEGO assembly task requirements.

### Why this matters
Bimanual LEGO assembly requires both global context and local precision views. Multi-view inputs also strengthen generalization and make demos more compelling for industrial evaluation.

### View specification (frozen)
1. **Workspace Overhead** — global layout of bricks and task context
2. **Left Wrist Camera** — precision grasp/alignment view
3. **Right Wrist Camera** — precision grasp/alignment view
4. **Third-Person Camera** — debugging + presentation + additional context

### Execution checklist
- Define camera frames and mounts in MJCF (workspace, wrists, third-person).
- Choose initial resolution and capture rate (aligned with policy rate unless constrained).
- Ensure synchronization (all cameras correspond to the same sim timestep).
- Validate:
  - wrist cameras move with their respective end-effectors correctly
  - overhead sees full workspace
  - third-person provides stable diagnostic framing
- Log camera intrinsics/extrinsics metadata if available/needed for reproducibility.

### Milestone (minimum success criteria)
- All 4 camera streams render reliably (headless-capable) and are time-aligned.
- Wrist cameras track end-effectors correctly with no frame drift.
- Camera configuration is frozen for dataset logging.

---

# Final Definition of Done (Phase 1.1)
Phase 1.1 is complete when:
- The **Alex upper-body fixed-base MJCF** loads and simulates stably.
- **Joint limits + Level 2+ constraints** are configured and validated.
- **Bimanual parallel-jaw grippers** work reliably for basic grasp+lift.
- The **action contract (16-D Δq + gripper)** is frozen and stable under repeated action chunks.
- The **robot state contract** includes core proprioception + EE pose/velocity and is consistent.
- **Four cameras** (overhead, wrist-L, wrist-R, third-person) render reliably and synchronously.
- A **Kinematics Validation Report** exists with FK-based evidence of Alex compatibility.

---

# Notes / Risks (explicit)
- LEGO contact richness can cause solver instability; prioritize collision simplification and stable solver settings early.
- Kinematic mismatches (axis flips, wrong EE frame) are the #1 silent failure mode; validation report is mandatory before dataset generation.
- Do not expand to full-body locomotion in 1.x; keep scope aligned with bimanual tabletop assembly.
