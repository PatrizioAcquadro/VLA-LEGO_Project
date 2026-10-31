# Data and admission

## Expanded Alex003 campaign — October 26, 2026

The completed corpus contains **620 admitted episodes / 2,528,001 aligned frames**. It is simulated teacher/correction evidence; it does not establish a qualified policy or physical robot dataset.

| Category | TRAIN | Calibration | Confirmation |
|---|---:|---:|---:|
| Single block | 160 | 15 | 15 |
| Ordered tower | 240 | 25 | 25 |
| Recovery | 75 | 10 | 10 |
| Justified stop | 25 | 10 | 10 |
| Total | 500 | 60 | 60 |

Recovery coverage includes missed grasp, slip/drop, wrong piece, incomplete placement and interrupted progress: 15 TRAIN and two cases in each development partition per mechanism. At least 30 TRAIN corrections arise from actual recorded policy states, with a physical teacher prefix/continuation and masked negative policy actions. An interruption declared in a case does not count unless the intended physical event occurred and the same task was completed. The final correction retains 123 negative policy commands before its verified continuation.

All **122 collection failures** remain excluded and preserved: 56 physical, nine absent-event, 24 unsupported, two scoring, 27 setup and four budget outcomes. Separate data exclusions, resource/startup interruptions and seven over-quota originals retain their own ledgers and receive no quota credit. Prospective layout changes did not overwrite admitted/development definitions or prior failures. Accepted source variants remain tied to their exact frozen recordings; unexercised repairs stay unpromoted.

Canonical storage is documented in [compute](cluster.md#canonical-campaign-storage). The corpus root contains `admitted-episodes.json`, declared case definitions, `alex_interface.json`, recordings, original reports and producer controls. `final-delivery-1838.json` records final accounting. Source/data/visual review is the admission authority; a declaration, count or setup pass alone cannot unlock learning.

## Episode content and leakage boundary

Each recording retains `episode.json`, clock-aligned `frames.jsonl`, supervision/action-origin `annotations.jsonl`, privileged evidence separately, and head RGB as PNG or lossless FFV1 `head.mkv`. The original decoded RGB frame identity is preserved. Named positions/velocities and the applied joint/gripper action share the control clock; capture does not advance physics.

`PilotEpisode` and the campaign readers construct valid future-action windows without crossing error boundaries. Failed/error-producing actions remain context or negative targets, never positive motor imitation. Stop prefixes provide context; only verified justified released-stop segments provide their operational motor targets. Decisions use causal current/past physical outcomes with separate labels; future outcome timing cannot leak into policy inputs.

Generation families and descendants remain in one partition. Normalization uses TRAIN only; calibration selects model/thresholds, and independent confirmation evaluates a frozen candidate. Paired/counterfactual instructions share their original family boundary. Exposed confirmation cannot be reused for future selection under the same independence claim. [Contracts](contracts.md) owns signal order/action decoding; [models](models.md) owns the learning recipe.

## Teacher and demonstration generation

The teacher physically grasps, lifts, carries, stabilizes, places, releases and retreats through robot actuators. It may use simulator truth that the policy never receives. Reset-to-goal scenes, object teleports during execution, external assisting forces, retention welds and untrained action tensors are not demonstrations or learned success.

Admit a whole original only after physical support/cooperation/release checks, complete recording/mask review and individual-image inspection. Isolated IK, a restored-state component or a later easier attempt cannot replace failed whole-task proof. Record requested and actually observed recovery mechanisms separately. Conditional teacher handoffs and true autonomous-from-loose-start outcomes remain distinguishable.

Reuse object-relative demonstration expansion and structured scene variation only within measured physical ranges, preserving bilateral timing and contact dependencies. Human interventions or larger-model annotations need executed physical verification before becoming supervision. [References](references.md) records the research rationale.

## Approved 50+10 collection milestone

The earlier marker-free pilot contains 50 TRAIN episodes (30 nominal / 15 recovery / five stop) and ten development episodes (six / three / one), in 333,373 frames: 328,886 positive and 4,487 masked error frames. It used continuous free construction sites, eight visible objects, variable stabilizer roles, 20 admitted color/shape/scale combinations and four reserved combinations. Later placements track the actual base, without a visible destination marker.

Its yield was 60/116 native attempts: 53 failed and three interrupted attempts remain preserved. Including two launcher-only failures gives 60/118. The earlier marked corpus contains 35 attempts (22 passed, 11 failed, two interrupted) and contributes zero marker-free quota credit. The 50+10 pilot and later correction/paired-instruction cohorts are distinct historical data, not additional entries silently pooled into the 620-episode campaign.

The pilot root retains delivery/review reports, case indices, interface and frozen sources. Its raw attempts are on [Blackwell](cluster.md#historical-pilot-recordings--october-27-2026). Historical readers and inference remain supported; obsolete single-seed converters and old launchers are recovered through Git rather than maintained as new-data workflows.

## Public sources and admission

Start from a robot-pretrained checkpoint and inventory public data broadly. Reuse its existing knowledge without automatically reproducing its pretraining. Add relevant public episodes when they improve manipulation, language grounding, coordination or robustness; preserve sufficient Alex task exposure.

The following is a source shortlist reviewed October 23, 2026, not a downloaded or admitted corpus. Public cards do not prove local access or release completeness. For each selected release, resolve its actual files, terms, embodiment, sensors, action semantics, overlap and intended learning role before use.

| Source and availability/terms | Embodiment, observations and actions | Intended value and boundary |
|---|---|---|
| [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) and [BridgeData V2](https://rail-berkeley.github.io/bridgedata/): public releases; OXE terms vary by constituent, BridgeData is CC BY 4.0. | Multiple robots; Bridge uses WidowX. Images, proprioception, language and embodiment-specific motion/gripper actions. | Broad manipulation and block stacking priors. Bridge overlaps OXE; do not count or sample the same episodes twice. |
| [DROID](https://droid-dataset.github.io/droid/the-droid-dataset.html): public RLDS/raw releases; the [LeRobot conversion](https://huggingface.co/datasets/cadene/droid) declares Apache-2.0. Confirm terms for the selected upstream release. | Franka, wrist/exterior images, robot state, language, Cartesian/joint control fields. | Real visual diversity and grasp/place behavior, not direct bimanual Alex supervision. The raw-data documentation notes incomplete release coverage; verify actual files. |
| [ALOHA](https://tonyzhaozh.github.io/aloha/): select an identified real or simulated release; the [released simulated insertion subset](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) declares MIT. Do not generalize one card's terms to every ALOHA dataset. | Dual-arm platforms, images and joint/gripper state/commands; views differ by release. | Coordination and action-chunk priors. Distinguish human-operated simulation from physical demonstrations; do not copy foreign joint targets into Alex. |
| [RoboMIND / 2.0](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND): HF card declares Apache-2.0 with access conditions and links the 2.0 release on ModelScope. Exact 2.0 subset access/terms remain to be confirmed. | Multiple single/dual-arm and humanoid embodiments; visual observations, proprioception, language and robot-specific commands. | Prefer relevant dual-arm manipulation. The original HF collection and the [2.0 paper](https://arxiv.org/abs/2512.24653) are not the same release. |
| [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026): released task archives, CC BY-NC-SA 4.0. | Real G2 dual-arm platform; camera video, state/actions and task/subtask annotations in a LeRobot-based format. | Long task sequences, coordination and language. Select manipulation segments and honor noncommercial/share-alike terms; no assumed count from older AgiBot releases. |
| [Galaxea Open-World](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset): released archives with access conditions, CC BY-NC-SA 4.0. | R1 Lite; head/wrist RGB, arm/gripper/torso/base state, joint/gripper targets and whole-body commands, subtask language. | Real bimanual and language priors. Fixed-base Alex does not inherit mobile-base supervision. |
| [NVIDIA PhysicalAI](https://huggingface.co/collections/nvidia/physical-ai): several released datasets; access and licenses are dataset-specific. | Real/simulated robot trajectories, human data and assets across distinct collections; modalities/actions vary. | Prioritize GR00T teleoperation and cross-embodiment manipulation subsets. Asset-only or video-only releases are not automatically action datasets; keep evaluation-only collections out of training. |
| [RoboTwin 2.0](https://robotwin-platform.github.io/): generator and [data/assets](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) released; card declares MIT, imported assets retain their own terms. | Simulated dual-arm embodiments; rendered views, state/actions, language and task programs. | Select useful episodes if mappings are valid; principally reuse structured scene variation in Isaac. Do not add a second simulator to the maintained task stack. |

Admission is source-specific: record the release and attribution, available channels, action units/frames/timing, permitted use, overlap and chosen subset. A foreign embodiment can contribute supported cross-embodiment training or representation learning; it becomes Alex action supervision only through a justified, physically validated mapping. Preserve masks for absent modalities rather than inventing observations.

For each shortlisted source, Phase 3 records whether it is used as additional training data, already represented only through the chosen checkpoint, deferred or excluded. Direct ingestion is not mandatory for every source. A large volume with unresolved action semantics or no useful task contribution is not an acceptable substitute for Alex coverage.

## Current source decisions

No public foreign-robot actions are admitted as Alex action supervision. Robot-pretrained SmolVLA contributes its existing priors through the fixed weights. The acquired GR00T/DROID sample served only a failed backbone-access runtime check and is excluded from Alex training. Additional OXE/Bridge, DROID, ALOHA, RoboMIND, AgiBot, Galaxea, PhysicalAI and RoboTwin ingestion remains deferred until an exact subset, useful gap, terms and physically justified mapping are established. Avoid duplicate Bridge/OXE exposure and do not map mobile-base or foreign joint targets directly into Alex commands.

EgoDex/EgoScale human data remain deferred; human video/hand motion is not an Alex action label. Exact release terms and the supported learning objective must be established before ingestion. The source shortlist above records the October 23 review rather than a new access/license verification.

## Further data decisions

Choose any expansion from measured coverage and actual TRAIN policy failures, with independent family partitions and all prior failures preserved. The completed negative campaign does not authorize an unchanged rerun, automatic recollection or weaker thresholds. If confirmation examples inform the next intervention, reserve fresh independent cases for the next claim. Additional volume, lower loss and teacher success are separate from learned qualification.
