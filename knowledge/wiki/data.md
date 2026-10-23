# Data strategy — planned

Data coverage and correction quality are the central development priority. The native collector records causal nominal/corrective sensor–action pairs and one complete cooperative tower seed; there is no qualified assembly corpus or trained assembly policy. [Phases 3–5](implementation_phases/index.md) build this capability; large-scale collection waits for the teacher pilot and the selected deployable model/interface.

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

## Admission state during the implementation pilot

No public foreign-robot actions are currently admitted as Alex action supervision. The actual robot-pretrained SmolVLA checkpoint is the compact candidate under feasibility work; its priors are retained through the published weights. Exact pretraining exposure is not inferred from a source's popularity or from GR00T embodiment names.

| Reviewed source | Current decision and reason |
|---|---|
| OXE / BridgeData V2 | Additional ingestion deferred until the Alex pilot identifies a transferable gap and an exact constituent/mapping is selected; avoid Bridge/OXE duplicate exposure. |
| DROID | The official GR00T sample was acquired for runtime validation only; that inference failed at gated backbone access. Excluded from Alex training and qualification. Broader ingestion deferred pending a justified mapping. |
| ALOHA | Additional ingestion deferred. Bimanual priors are relevant, but foreign joint targets are not Alex commands and the selected release/supervision role must earn its cost. |
| RoboMIND / 2.0 | Deferred: exact admitted subset, access and 2.0 release terms are not resolved. |
| AgiBot World 2026 | Deferred: no selected task archives or validated fixed-base action mapping; the stated source-specific terms still apply to any future use. |
| Galaxea Open-World | Deferred: no selected subset or validated mapping; do not transfer mobile-base targets into Alex. |
| NVIDIA PhysicalAI | Deferred for direct episode ingestion; identify a specific action-bearing release first. GR00T weights were acquired, but its backbone access remains unavailable. |
| RoboTwin 2.0 | Deferred for direct episodes. Use its scene-variation principles within Isaac once the contact teacher is valid; do not introduce another task simulator. |
| EgoDex / EgoScale | Deferred for direct human-data ingestion. No fabricated robot actions; data terms, supported objective and incremental benefit remain prerequisites. |

The complete ninth tower reference is now explicitly admitted through its passing functional reassessment: 4,925 aligned frames in `tower-seed-0`, training only, converted to LeRobot with head RGB, 32 measured position/arm-velocity features and 16 actually applied joint/gripper commands. `convert_dataset --reassessments` checks that the positive assessment belongs to the requested complete nominal reference and covers its recorded frames. The conversion report preserves the original negative label and identifies the new assessment; raw reports and labels are untouched. This is one teacher seed, not independent coverage, a trained full-task policy or a qualified corpus.

These are initial source decisions, not completion of Phase 3. The first required useful corpus remains physically executed Alex data with verified nominal/corrective/stop labels and grouped splits. Revisit additional public data when a measured pilot or development failure supports its value.

## Human egocentric data

[EgoDex](https://github.com/apple-aiml-research/ml-egodex) provides public egocentric video, language and 3D head/hand/body trajectories; dataset terms are CC BY-NC-ND, separate from its code license. Treat additional use as conditional on compatible training/processing/redistribution terms and measured task benefit. Human hand motion is not a robot action label.

[EgoScale](https://research.nvidia.com/labs/gear/egoscale/) motivates human–robot alignment and transfer. A published method or checkpoint prior does not establish availability of its complete human corpus. First exploit applicable priors in the selected VLA; add human-video training only when the supported objective and benefit justify it. Raw videos without robot actions may support perception or language, never fabricated Alex commands.

## Teacher and demonstration generation

Start with a geometric teacher using collision-aware motion planning, feedback and coordinated end-effector/gripper control. It must physically grasp, lift, transport, support, place and release through robot actuation. Simulator truth may guide the teacher and evaluator, but is stored separately from policy observations. Reset-to-goal scenes, teleports during execution, external object forces and retention welds are not demonstrations.

Add human demonstrations or corrective interventions where scripted behavior lacks useful coverage. A larger model may propose programs or annotations offline; execute and verify the resulting behavior before accepting it. Fluent instructions or plausible imagined trajectories are insufficient.

Use [Isaac Lab Mimic](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/overview/teleop_imitation.html) and [DexMimicGen](https://dexmimicgen.github.io/) principles to transform object-relative subsequences and expand seeds. Preserve hand synchronization, contact phases and dependencies such as stabilization before placement. Reuse RoboTwin's structured variation of clutter, lighting, backgrounds, table geometry and language within measured physical ranges.

A declared disturbance may create a failure state for recovery collection or evaluation; record its cause and timing separately. It may not secretly help grasp, transport or complete the structure. Recovery actions after that disturbance still come from the teacher during collection or the policy during autonomous evaluation, through the normal robot interface.

## Initial case distribution

### Approved 50+10 collection milestone

The explicit marker-free revision supersedes all marked-area and fixed-destination requirements in that proposal. The main corpus must contain 60 freshly executed marker-free episodes; changing an old rendering does not create a new episode. The previous marked corpus is preserved separately under `outputs/pilot-collection-20261023` (35 attempts: 22 passed, 11 failed, 2 interrupted at the transition), contributing zero to the new quota. The new collection root is `outputs/pilot-markerfree-20261023`.

The teacher selects continuous free tabletop space relative to the loose inventory, without a marker or a destination cue in the image/instruction. After the first release, subsequent placements track the actual base. The evaluator uses the shared workspace and physical outcomes rather than distance to the teacher point. Both hands must release and withdraw, with the unchanged two-second support/stability dwell. Native checks at several positions precede full collection; code/kinematic tests alone do not establish spatial coverage. Scene derivatives retain their original family and split. Position distributions and collection yield include actual outcomes and failures.

The October 23 collection assignment is governed by the approved local specification `outputs/pilot-planning-20261023/pilot-proposal.md` and exact `pilot-cases.csv`. Its fixed usable-episode targets are 30 nominal / 15 recovery / 5 stop for training and 6 / 3 / 1 for development. Families T1–T5 and all descendants stay in training; D1–D2 and descendants stay in development. The former seven-object tower seed is auxiliary evidence and does not fill this eight-object pilot.

`vla_lego.isaac.pilot` loads the approved CSV, rejects reserved combinations and constructs eight-object scenes with the declared order, stabilizer, cooperation level and variation. `tower_reference --cases ... --case ...` executes complete nominal/corrective cases; `pilot_stop` replaces a missing requested piece while keeping eight visible objects. Recoveries are explicitly teacher-induced robot commands: missed lift, incomplete load transfer or actual distractor lift/return. Their physical consequences must be verified before admission. No object teleports, retention welds or external object forces assist completion.

Raw recordings retain 20 Hz head RGB, 18 measured positions/velocities, 16 applied joint/jaw targets, requested targets and English instructions. Continue/complete/abstain are output labels. `privileged.jsonl` contains simulator evidence; `annotations.jsonl` contains stages and `positive_imitation`. Neither file is a policy input. False mask values retain error context while excluding those actions from positive imitation. `pretrained.pilot_dataset.PilotEpisode` reads one admitted episode with lazy image loading, 32 measured state features, 18 motor/decision outputs and `action_is_pad`. Chunks stop at an error boundary; corrective samples start from their own observed state. No normalization or training runs during collection. All transformations remain inside the original family/split. The historical diagnostic converter does not consume these pilot masks; use the masked reader for this corpus.

`pilot_audit --root ... --cases ... --output ...` lists all attempts, checks case identity, synchronized files/clocks, finite named signals, required terminal labels and the assigned physical cooperative event, and indexes admitted cases separately from missing cases. It also validates the actual joint-name order and exports `alex_interface.json` using the existing VLA interface convention. Execution evidence and source snapshots for the new condition live in `outputs/pilot-markerfree-20261023`; previous marked evidence stays in its original directory. Configurations and passing contract tests are not collected episodes. Collection yield and visual acceptance are reported only from finalized attempts. Printing, physical measurement and Orin checks do not block this simulated milestone. Stop after delivery; subsequent adaptation and roadmap progression require a new assignment.

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
