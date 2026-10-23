# Status

Reviewed October 23, 2026. This revision implements the **documentation roadmap only**. All [new phases](implementation_phases/index.md) remain planned. Prior component evidence is preserved in [validation](validation.md), without being relabeled as Isaac, Alex003 or learned assembly success.

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

1. Begin Phase 0 ownership/environment cleanup and Phase 1 minimum Isaac scene. Retire legacy paths only as replacements become operational.
2. Resolve the Orin software compatibility path and actual WSG32 command/feedback interface early. Simulated gripper readiness does not establish physical integration.
3. Calibrate the initial printed set and physically executed cooperative teacher, then select the model using a bounded pilot. Large-scale collection waits for both Phase 2 and Phase 3.
4. Iterate coverage, fine-tuning and corrections toward the [qualification targets](validation.md#assembly-qualification-targets).

The [handoff order and decision ownership](implementation_phases/index.md#dependencies-and-handoffs) distinguish partial readiness from completed phases. The current project closes with the qualified five-block system and Phase 8 consolidation. Walls/bridges, insertion blocks, LEGO and a paper are optional subsequent scope.

The [GR00T/JetPack mismatch](models.md#onboard-feasibility) is unresolved; no upgrade or onboard run is claimed. Print properties, Alex-specific latency, head-camera sufficiency, physical performance and resource allocations remain to be measured. Existing SDK FK parity was skipped; legacy self-consistency does not qualify the new robot.

LIBERO, custom interleaving, EO-1 replication and paper-oriented ablations are outside the required roadmap. The historical website/calendar no longer determines development order.
