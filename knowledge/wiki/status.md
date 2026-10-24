# Status

Reviewed October 24, 2026. The **marker-free 50 training + 10 development episode pilot is complete**, and the separately authorized **SmolVLA learning campaign is running on Gilbreth**. [Data strategy](data.md#approved-5010-collection-milestone) owns collection evidence; [models](models.md#active-marker-free-pilot-adaptation--october-24-2026) owns the substantial training base, stopping rules and bounded before/after panel. The previous marked corpus contributes zero to this run. No autonomous pilot result, larger collection, hardware/Orin run or Phase 6 qualification is claimed. Native teacher demonstrations and admitted data do not establish learned assembly. Earlier diagnostics remain in [validation](validation.md).

## Implementation progress

| Phase | Verified state and remaining work |
|---|---|
| 0 | Native CUDA PhysX scene consumes shared Alex003 WSG32/UMI, measured pedestal and official head ZED without copied task robot assets. Gravity/table support, both jaw commands, fixed root, finite states and visible block passed. Ownership, separate environments and staged retirement documented in [foundations](foundations.md). |
| 1 | Single-block reference, shared TCP/FK checks, licensed catalog, loose five-target scene and rotated-grasp reach checks verified. Loaded-target servo bias resolved. The eighth reference passed released stability but failed cooperative alignment. The ninth achieved three seconds of qualifying cooperation and correct released geometry. It failed historical speed gates but passes an explicit reassessment using functional released-dwell displacement: maximum corner motion 0.169 mm over two seconds. Original outcomes and raw labels are preserved. Twenty-one analytical scoring/motion regressions pass; the reassessed 4,925-frame seed has been explicitly converted for training. A fresh event-settled teacher also passes (3,316 frames, one second of cooperation); physical calibration remains open. [Evidence and limits](foundations.md#tcp-control-and-single-block-reference). |
| 2 | GR00T native inference blocked by gated-backbone HTTP 403. SmolVLA CUDA adaptation/reload/resume and sensor-only autonomous execution demonstrated; all twelve bounded learned attempts failed across distinct diagnostic conditions. Provisional named joint interface follows three retained failed Cartesian replays. Successful learned task-conditioned pilot, final selection, Orin and physical WSG32 readiness remain open. |
| 3–5 | Marker-free 50+10 pilot complete, with native towers, bimanual restraint, verified corrections, justified stops and preserved attempts/splits. Actual GPU adaptation/reload/resume passed; one substantial SmolVLA run now uses masked targets, train-only normalization and fixed train/development diagnostics. The first learned before/after panel is pending. Earlier seed diagnostics remain separate. The broader collection/learning loop is not complete. |
| 6 | Independent frozen nominal/generalization/recovery campaigns have not begun. |

Native evidence is retained locally in `outputs/blackwell/foundation-001` through `foundation-005` and on Blackwell in `/home/pacquadr/projects/vla-lego/outputs`. Attempts 001–002 failed during bootstrap/configuration. Attempt 003 passed physical component checks but visual inspection found the table outside the image; 004 preserved the failed visibility check. Attempt 005 passed all nine component/visibility checks (5.04 simulated seconds; 16.29 s to report before app close), and its head image was inspected. This is component readiness only. Actual printed-block measurements, physical WSG32 command/feedback/stop and concurrent Orin timing remain open; the user's exception permits independent simulation work with declared assumptions.

## Existing implementation

| Capability | Current state and limit |
|---|---|
| Robot simulation | Native Isaac/Alex003 WSG32/UMI physical teacher and admitted simulated pilot; retained MuJoCo Alex V1 diagnostics. Hardware dynamics remain unqualified. |
| Interfaces | Native pilot: 32 measured state features, 16 applied motor commands plus 2 decision outputs, head RGB at 20 Hz. Legacy 17-D/52-D/four-view/16-step consumers remain separate. |
| Blocks and diagnostics | Configurable cuboids, seeded resets, tower/planar-triangle evaluation and scripted robot clips. Prearranged goals are not assembly demonstrations. |
| Model components | Qwen3.5-4B plus custom untrained action head; real GPU forward and gradient-routing evidence, no learned task capability. |
| Losses and tracking | Resumable masked SmolVLA trainer, balanced sampling and fixed diagnostics run natively on A100. Marker-free evaluation components, worker/video and physical recovery initialization pass; the learned comparison remains pending. |
| Data and policy | Admitted marker-free 50+10 pilot and running adaptation; autonomous pilot improvement and final qualification remain unmeasured. |
| Packaging | Local installed-package/GPU Docker evidence and actual isolated Gilbreth training. Apptainer and onboard inference remain unqualified. |

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

1. Complete the authorized substantial training campaign, prepare the paired ten-case development comparison while it runs, then report measured improvement or an honest negative result. [Compute operations](cluster.md#active-pilot-training) owns the single-job/15-minute monitoring path.
2. Use the admitted family splits and masks. Decide whether targeted corrections or a larger corpus are warranted only after the before/after result; the development set does not replace new final qualification episodes.
3. Physical print properties, WSG32 command/feedback and Orin compatibility/timing remain unresolved measurements. They did not block the simulated pilot and are not newly authorized work.

The [handoff order and decision ownership](implementation_phases/index.md#dependencies-and-handoffs) distinguish partial readiness from completed phases. The current project closes with the qualified five-block system and Phase 8 consolidation. Walls/bridges, insertion blocks, LEGO and a paper are optional subsequent scope.

The [GR00T/JetPack mismatch](models.md#onboard-feasibility) is unresolved; no upgrade or onboard run is claimed. Print properties, Alex-specific latency, head-camera sufficiency, physical performance and resource allocations remain to be measured. Existing SDK FK parity was skipped; legacy self-consistency does not qualify the new robot.

LIBERO, custom interleaving, EO-1 replication and paper-oriented ablations are outside the required roadmap. The historical website/calendar no longer determines development order.
