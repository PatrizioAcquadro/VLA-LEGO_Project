# Data strategy — planned

Data coverage and correction quality are the central development priority. There is currently no maintained collector, qualified assembly corpus or trained policy. [Phases 3–5](implementation_phases/index.md) build this capability; large-scale collection waits for the teacher pilot and the selected deployable model/interface.

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

## Human egocentric data

[EgoDex](https://github.com/apple-aiml-research/ml-egodex) provides public egocentric video, language and 3D head/hand/body trajectories; dataset terms are CC BY-NC-ND, separate from its code license. Treat additional use as conditional on compatible training/processing/redistribution terms and measured task benefit. Human hand motion is not a robot action label.

[EgoScale](https://research.nvidia.com/labs/gear/egoscale/) motivates human–robot alignment and transfer. A published method or checkpoint prior does not establish availability of its complete human corpus. First exploit applicable priors in the selected VLA; add human-video training only when the supported objective and benefit justify it. Raw videos without robot actions may support perception or language, never fabricated Alex commands.

## Teacher and demonstration generation

Start with a geometric teacher using collision-aware motion planning, feedback and coordinated end-effector/gripper control. It must physically grasp, lift, transport, support, place and release through robot actuation. Simulator truth may guide the teacher and evaluator, but is stored separately from policy observations. Reset-to-goal scenes, teleports during execution, external object forces and retention welds are not demonstrations.

Add human demonstrations or corrective interventions where scripted behavior lacks useful coverage. A larger model may propose programs or annotations offline; execute and verify the resulting behavior before accepting it. Fluent instructions or plausible imagined trajectories are insufficient.

Use [Isaac Lab Mimic](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/overview/teleop_imitation.html) and [DexMimicGen](https://dexmimicgen.github.io/) principles to transform object-relative subsequences and expand seeds. Preserve hand synchronization, contact phases and dependencies such as stabilization before placement. Reuse RoboTwin's structured variation of clutter, lighting, backgrounds, table geometry and language within measured physical ranges.

A declared disturbance may create a failure state for recovery collection or evaluation; record its cause and timing separately. It may not secretly help grasp, transport or complete the structure. Recovery actions after that disturbance still come from the teacher during collection or the policy during autonomous evaluation, through the normal robot interface.

## Initial case distribution

| Share of attempted collection cases | Coverage |
|---|---|
| **60% varied nominal cases** | Piece identity, dimensions, colors, initial positions, instructions, valid order, distractors and arm roles. |
| **30% recoverable errors** | Missed grasp, slip, off-center placement, wrong piece, occlusion, misalignment and interrupted bimanual coordination, followed by corrective behavior. |
| **10% non-completable or stop cases** | Required piece absent, unreachable configuration, non-recoverable instability or insufficient observations; appropriate termination instead of unsafe continuation. |

These are starting collection quotas, not promised success rates or mandatory training-loss weights. Appearance, geometry and difficulty cross all categories. Distinguish intended perturbation from actual observed error: an injected perturbation does not guarantee a failure or a useful recovery. Record every attempt and the achieved distribution; adjust future collection using coverage and development outcomes. Do not relabel past failures or teach blanket refusal on feasible difficult cases.

## Learning from mistakes

Preserve the lead-up to the error, actions actually executed, intervention boundary, corrective actions and subsequent outcome. Accepted nominal and successful corrective segments supply positive imitation targets. Failed actions remain diagnostic context; they are not automatically positive targets. Any failure-prediction, preference or reinforcement objective requires explicit labels and a supported learning method.

| Segment | Learning role |
|---|---|
| Verified nominal action | Positive action supervision |
| Verified corrective continuation | Positive correction supervision conditioned on the available error context |
| Failed action or unsuccessful correction | Preserved context/diagnosis; excluded from positive action imitation unless a separately validated useful subsegment is identified |
| Justified abstention/stop | Supervision for the selected deployable stop mechanism, including allowed hold behavior; not a successful tower label |

Collection categories and training weights are different decisions. Report both, including how much usable nominal, corrective and stop supervision remains after filtering. Context frames must not accidentally turn every action in a failed episode into an imitation target.

Once a VLA exists, collect corrections to errors it actually produces. [IntervenGen](https://arxiv.org/abs/2405.01472) and [RECAP](https://www.pi.website/blog/pistar06) motivate this iterative rollout–correction–learning loop. Reuse the principle without making their full algorithms mandatory. Report assisted collection separately from autonomous performance.

## Episode content and leakage boundary

An episode retains the instruction, block/scene description, source and generation family, timestamped deployable images/state, requested and applied actions, stage/intervention labels, termination reason and outcome. Record physical-piece identity and measured print properties when relevant. Keep object poses, contacts, teacher state and evaluator labels in a separate channel.

Preserve observation → applied action → next observation alignment, real camera geometry and timing, command clipping and valid chunk tails. Prefer the selected model's supported dataset format, such as a compatible LeRobot version, over a new custom framework. Choose storage and scale from measured pilot size, throughput and useful-data yield.

Annotations describe visible pieces, substeps, corrections and observed outcomes with uncertainty. They cannot supply future information or simulator coordinates to earlier policy decisions. Mandatory synthetic reasoning chains or custom temporal interleaving are not part of the baseline.

## Growth and generalization

Split complete episodes and related seed/generation families before creating windows. Keep transformed descendants of one seed demonstration together; group related physical instances/sessions where needed to prevent leakage. Fit normalization only on training data.

Reserve color–shape combinations, sizes, layouts and instruction variants; distinguish recombination of known attributes from genuinely unseen geometry or dimensions. Report each axis separately, including whether the reserved material was already seen in public pretraining. Cross color and shape assignments so color does not become a shortcut for geometry.

Scale pilot → broader corpus → policy-driven corrections according to coverage and learning curves, not an inherited episode quota. Keep nominal ability while adding recoveries and real adaptation. Never tune routinely on final qualification episodes. If public pretraining exposure is unknown, say so: a project-held-out combination is not proof that the foundation model has never seen it.

## Corpus readiness and growth

The Phase 3 pilot supplies full-tower seed demonstrations, corrections for its admitted initial error families and stop examples. It establishes that recording, supervision and selected-model adaptation form a usable path; it is not the full training corpus.

The initial Phase 4 handoff contains:

- Usable demonstrations for selection, grasp, transport, stabilization, placement, release and continuation across all five levels, with both arm roles represented.
- Variation that separates color from size and geometry, including misleading distractors and different feasible instructions/layouts.
- Explicit nominal/recovery/stop coverage, source decisions, action supervision roles, useful-data yield and remaining gaps.
- Training/development/qualification groups defined before windowing, with no transformed seed descendants crossing groups.
- Data consumable under the Phase 2 contract, with no unresolved mapping, missing-signal fabrication or privileged policy inputs.

This initial handoff starts Phase 5. Later batches address measured errors; the full collection/learning cycle must cover all seven named recovery families and the declared generalization axes before final qualification. At the end of each bounded batch or learning iteration, decide which gap to address next or whether the candidate is ready for Phase 6. No fixed episode quota or improving training loss alone establishes sufficient coverage.
