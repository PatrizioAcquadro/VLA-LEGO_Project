# Phase 8 — Sim-to-real preparation and handoff

Scope: original 8.1, 8.2 and 8.3, plus the deferred real-LEGO extension. **State:** current software knowledge and operating baseline delivered; physical characterization, hardware control mapping and final trained-model/data handoff remain planned. A documentation cleanup does not complete the research project.

## 8.1 Sim-to-real gap analysis

### Work to complete

1. Inventory simplifications: fixed base/neck, primitive collisions, tuned dynamics, idealized joint sensing, coupled gripper model, uniform cuboid inertia, provisional mass/friction, renderer/camera assumptions and no communication/control latency.
2. Measure an actual printed pilot set: dimensions/mass, grasp aperture, slip, compliance and support stability. Record nominal variability and measurement uncertainty. Use these observations to prioritize errors that affect grasp/release rather than randomizing every parameter indiscriminately.
3. Characterize observation differences: lighting/background/material, wrist extrinsic calibration, image timing, occlusion and sensor noise. Keep deployable inputs matched to the [contract](../contracts.md); simulator truth must not fill missing hardware sensors silently.
4. Propose bounded domain-randomization or calibration interventions tied to observed failure modes. Compare an unchanged baseline with calibrated/perturbed cases. Avoid stacked arbitrary uncertainty margins that reject otherwise feasible tasks without evidence.
5. Estimate transfer risks and what must be measured next. Do not turn simulation success rates into numeric real-world performance predictions without supporting physical trials.

### Completion gate

Deliver a ranked gap analysis with measured versus assumed parameters, justified acceptance ranges, bounded experiments and unresolved risks. Physical qualification needs actual repeatable printed-brick grasps/releases and the permitted hardware setup; none was performed during this audit.

## 8.2 Alex action-space mapping

### Work to complete

1. Compare the pinned SDK model and actual robot configuration: joint order/signs/units, base/tool frames, gripper hardware and sensing, position/velocity/effort interfaces and available state signals.
2. Resolve the skipped independent FK comparison. Correct source import/inertia representation without changing the kinematic chain, then compare named tool poses on nominal and perturbed configurations. Preserve failed/skipped outcomes and do not substitute self-symmetry for parity.
3. Specify the mapping from normalized 17-D commands to the hardware controller. Define rate, interpolation, clipping, timeouts, latency, gripper opening convention and which components enforce physical limits. Simulation velocity clamps are not a hardware emergency-stop system.
4. Reconstruct the 52-D observation from actual sensors/kinematics, including world/reference-frame alignment and quaternion order. Declare any unavailable signal instead of inventing a simulator-only equivalent.
5. Validate first through offline/replay and the authorized hardware integration process, then bounded nominal/perturbed motion. Keep robot safety requirements, empirical task calibration and numerical tolerances distinct.

### Completion gate

Deliver an explicit checked mapping table, reference/frame tests, timing/latency observations, unavailable signals and the hardware validation boundary. No physical robot deployment or controller certification is implied by this source repository.

## 8.3 Project documentation and handoff

### Current baseline

The canonical wiki covers setup, use, architecture, contracts, status, limitations and all original implementation plans. Maintained code, package/assets and local diagnostic artifacts can be reproduced through documented commands. Tracked asset provenance and consequential negative historical findings are retained; ordinary history remains in Git. Local assistant instructions are ignored.

### Remaining research handoff

1. Keep the index/status accurate when capabilities change; one page per topic, links for shared contracts, a short consequential-decision log and Git for ordinary edits.
2. Provide a technical report that separates implementation, component verification, learned task metrics, physical qualification and unavailable outcomes. Include failure taxonomy, full denominators and a clear contribution claim suitable for ONR/advisor review.
3. Package actual trained checkpoints, dataset/split definitions, normalization, evaluation protocol and reproduction instructions when those artifacts exist. State licenses/access restrictions and resource requirements. Do not package an untrained head as a task policy.
4. Reproduce representative successful and failed workflows from the intended installed/container environment. Record limits and pending cluster/hardware checks rather than implying universal reproducibility.
5. Publish or integrate into main only through separately authorized actions. Local commits, a validated image and a readable wiki do not establish a release.

### Completion gate

An independent developer must be able to identify the current capability boundary, run the maintained checks, locate each artifact and resume the next phase without contradictory instructions. Final research completion additionally requires the promised data/model/evaluation outputs or an explicit report that their objectives were not achieved.

## Future extension: real LEGO

**Deferred; no active runtime implementation.** Real LEGO means actual stud/tube connection and insertion/retention, not the current variable-size printed gravity blocks. Begin this extension only after a viable lightweight-brick policy and a concrete physical connection requirement exist.

1. Establish real part geometry, materials and measured force/displacement/retention ranges for a small part family. Respect source licenses and quantify dimensional/manufacturing variation.
2. Revisit contact/compliance with a bounded identification study. The old pure-contact result was about 0.06 N/stud against an original 0.3 N/stud target; the gap was not solved by later threshold relaxation. Preserve that negative result.
3. Distinguish material/contact simulation from any explicit weld or spec-proxy mode. A proxy may serve a labeled planning experiment but cannot qualify physical retention or contribute hidden assistance to learned-policy success.
4. Develop real robot grasp/alignment/insertion demonstrations, with measured forces and failure/recovery labels. Do not relabel existing gravity-brick trajectories as connected LEGO demonstrations.
5. Add connection-specific evaluation, calibration and transfer experiments with fixed criteria and full failure denominators. Reuse maintained observation/control components; add complexity only for a demonstrated connection requirement.

Completion requires qualified physical insertion/retention evidence and an explicitly measured learned task outcome. This future plan preserves the original ambition while keeping the current repository focused on feasible lightweight-brick work.
