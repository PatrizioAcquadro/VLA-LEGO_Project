# Status

Reviewed October 23, 2026. The active assignment is the **marker-free 50 training + 10 development episode data pilot**, as described in [data strategy](data.md#approved-5010-collection-milestone). The explicit marker-free revision requires fresh collection at continuous valid build positions and preserves the earlier marked corpus separately (22 passed, 11 failed, 2 interrupted; zero new-quota contribution). Marker-free collection and native spatial verification are in progress. It supersedes earlier automatic progression through Phases 0–6: stop at corpus delivery, with no training, expansion or qualification campaign. Native simulation, a single-block physical reference and causal command replay are verified on Blackwell. Earlier SmolVLA diagnostics do not establish autonomous assembly; the full assembly checkpoint and qualified corpus remain unverified. Prior component evidence is preserved in [validation](validation.md).

## Implementation progress

| Phase | Verified state and remaining work |
|---|---|
| 0 | Native CUDA PhysX scene consumes shared Alex003 WSG32/UMI, measured pedestal and official head ZED without copied task robot assets. Gravity/table support, both jaw commands, fixed root, finite states and visible block passed. Ownership, separate environments and staged retirement documented in [foundations](foundations.md). |
| 1 | Single-block reference, shared TCP/FK checks, licensed catalog, loose five-target scene and rotated-grasp reach checks verified. Loaded-target servo bias resolved. The eighth reference passed released stability but failed cooperative alignment. The ninth achieved three seconds of qualifying cooperation and correct released geometry. It failed historical speed gates but passes an explicit reassessment using functional released-dwell displacement: maximum corner motion 0.169 mm over two seconds. Original outcomes and raw labels are preserved. Twenty-one analytical scoring/motion regressions pass; the reassessed 4,925-frame seed has been explicitly converted for training. A fresh event-settled teacher also passes (3,316 frames, one second of cooperation); physical calibration remains open. [Evidence and limits](foundations.md#tcp-control-and-single-block-reference). |
| 2 | GR00T native inference blocked by gated-backbone HTTP 403. SmolVLA CUDA adaptation/reload/resume and sensor-only autonomous execution demonstrated; all twelve bounded learned attempts failed across distinct diagnostic conditions. Provisional named joint interface follows three retained failed Cartesian replays. Successful task-conditioned pilot, final selection, Orin and physical WSG32 readiness remain open. |
| 3–5 | Causal single-block diagnostic: six single-block collection attempts: the first excluded for clock alignment; three admitted nominal producer conditions (1,066/704/594 frames) and two verified corrective continuations (618/508 frames), all in one training family. Exact replay verified; paired correction and integrated-servo adaptations still failed autonomously; exact absolute native command replay; real adaptation and resume, with autonomous outcomes retained. One complete cooperative seed is now explicitly admitted and converted (4,925 frames, tower-seed-0 training only). Varied full-task nominal/recovery/stop pilot, corpus and autonomous training iterations remain pending. [Diagnostic boundaries](foundations.md#loose-scene-and-causal-learning-diagnostic). |
| 6 | Independent frozen nominal/generalization/recovery campaigns have not begun. |

Native evidence is retained locally in `outputs/blackwell/foundation-001` through `foundation-005` and on Blackwell in `/home/pacquadr/projects/vla-lego/outputs`. Attempts 001–002 failed during bootstrap/configuration. Attempt 003 passed physical component checks but visual inspection found the table outside the image; 004 preserved the failed visibility check. Attempt 005 passed all nine component/visibility checks (5.04 simulated seconds; 16.29 s to report before app close), and its head image was inspected. This is component readiness only. Actual printed-block measurements, physical WSG32 command/feedback/stop and concurrent Orin timing remain open; the user's exception permits independent simulation work with declared assumptions.

## Existing implementation

| Capability | Current state and limit |
|---|---|
| Robot simulation | MuJoCo fixed-base Alex V1 upper body and two EZGrippers; native control/stability evidence exists, hardware dynamics unqualified. |
| Interfaces | Legacy 17-D actions, 52-D state, four synchronized views and 16-step chunks. These remain current code, not requirements for the new embodiment. |
| Blocks and diagnostics | Configurable cuboids, seeded resets, tower/planar-triangle evaluation and scripted robot clips. Prearranged goals are not assembly demonstrations. |
| Model components | Qwen3.5-4B plus custom untrained action head; real GPU forward and gradient-routing evidence, no learned task capability. |
| Losses and tracking | Standalone components exist; no maintained production trainer, collector or learned rollout evaluator. |
| Data and policy | No qualified assembly corpus or trained assembly checkpoint. |
| Packaging | Local installed-package and GPU Docker evidence. Apptainer, cluster training and onboard inference are unqualified. |

## Planned target

| Area | Direction |
|---|---|
| Task | Ordered five-block tower, color/size selection among distractors, stabilize–place cooperation and recovery. |
| Platform | Isaac Sim/Lab using shared Alex003 assets, fixed pedestal, two WSG32/UMI grippers and head ZED X Mini. |
| Policy | Robot-pretrained VLA; GR00T N1.7 leads, SmolVLA is the compact alternative. Selection remains conditional on task and Orin feasibility. |
| Data | Selected public knowledge plus physically executed Alex simulation demonstrations, corrections and later targeted real data. |
| Deployment | Entire VLA inference on Orin, with the established low-level robot controller. |
| Progression | Qualify simulation, measure transfer before adaptation, adapt on separate real data, then qualify the physical task and consolidate its handoff. |

## Immediate priorities and unresolved facts

1. Expand the verified single-block reference into the cooperative five-block task and calibrate coverage/contact assumptions. Retire legacy paths only as replacements become operational.
2. Resolve the Orin software compatibility path and actual WSG32 command/feedback interface early. Simulated gripper readiness does not establish physical integration.
3. Calibrate the initial printed set and physically executed cooperative teacher, then select the model using a bounded pilot. Large-scale collection waits for both Phase 2 and Phase 3.
4. Iterate coverage, fine-tuning and corrections toward the [qualification targets](validation.md#assembly-qualification-targets).

The [handoff order and decision ownership](implementation_phases/index.md#dependencies-and-handoffs) distinguish partial readiness from completed phases. The current project closes with the qualified five-block system and Phase 8 consolidation. Walls/bridges, insertion blocks, LEGO and a paper are optional subsequent scope.

The [GR00T/JetPack mismatch](models.md#onboard-feasibility) is unresolved; no upgrade or onboard run is claimed. Print properties, Alex-specific latency, head-camera sufficiency, physical performance and resource allocations remain to be measured. Existing SDK FK parity was skipped; legacy self-consistency does not qualify the new robot.

LIBERO, custom interleaving, EO-1 replication and paper-oriented ablations are outside the required roadmap. The historical website/calendar no longer determines development order.
