# Research rationale and references

Reviewed October 23, 2026. The goal is a useful bimanual assembly system and successful physical transfer, not a new architecture or a paper prerequisite. External results guide design; they do not qualify Alex003.

## Decisions informed by existing work

| Evidence | Decision for this project | What not to infer |
|---|---|---|
| [GR00T](https://github.com/NVIDIA/Isaac-GR00T), [SmolVLA](https://huggingface.co/blog/smolvla), [openpi](https://github.com/Physical-Intelligence/openpi), [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) | Adapt a robot-pretrained policy; use the [model selection and Orin gate](models.md) before scaling collection. | Public benchmark scores, model size or desktop speed do not identify the best Alex policy. |
| [Isaac Lab imitation learning](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/overview/teleop_imitation.html) and [DexMimicGen](https://dexmimicgen.github.io/) | Expand object-relative demonstration segments while retaining bimanual temporal dependencies. | Transformed paths are not automatically collision-free, physically successful or compatible with the chosen Lab release. |
| [RoboTwin 2.0](https://robotwin-platform.github.io/) | Reuse structured scene/language variation and simulation-verified teacher generation inside Isaac. | No need to maintain its separate simulator or import its performance claims. |
| [IntervenGen](https://arxiv.org/abs/2405.01472) and [RECAP](https://www.pi.website/blog/pistar06) | Collect corrections to actual policy errors and iterate learning, with failures retained. | Failed actions are not automatically useful positive imitation targets; full reproduction of either algorithm is not required. |
| [EgoDex](https://github.com/apple-aiml-research/ml-egodex) and [EgoScale](https://research.nvidia.com/labs/gear/egoscale/) | Consider human manipulation priors and supported human–robot alignment only when useful. | Human video or hand pose is not an Alex action, and a paper does not make its complete training corpus available. |
| [Real-time action chunking](https://www.pi.website/research/real_time_chunking) | Account for observation age, execution continuity and replanning latency when selecting horizons. | A predicted chunk length or inference frequency is not the actuator update rate. |
| [RGB-Objects / RGB-Stacking](https://github.com/google-deepmind/dm_robotics/tree/main/py/manipulation/props/rgb_objects) | Reuse documented printable geometry, vary appearance independently and measure grasp/support properties. | Published object identities and simulation parameters do not characterize our printed parts. |

Recent research such as [π0.7](https://www.pi.website/blog/pi07) informs language steering and generalization questions. The implementable comparison here uses released openpi checkpoints, including π0.5; a newer research report does not imply an available, qualified deployment path.

## Canonical source inventories

- [Data](data.md#public-sources-and-admission) owns dataset release/access/terms, embodiment mappings, public overlap and the teacher/correction strategy.
- [Models](models.md#planned-pretrained-vla-selection) owns candidate selection, deployment constraints and the distinction between reported and measured performance.
- [Blocks](bricks.md#printable-catalog) owns exact STL links, geometry and print calibration.
- [Architecture](architecture.md#planned-robot-and-ownership) owns the shared Alex source and hardware guide; [compute](cluster.md#planned-resource-roles) owns GPU roles and renderer requirements.

Resolve sources against the actual release used during implementation. Use the selected upstream model/data tooling before creating new infrastructure; add methods only to address an observed task limitation.

## Historical design references

[EO-1](https://arxiv.org/abs/2508.21112), [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and [Flow Matching](https://arxiv.org/abs/2210.02747) explain the retained custom model. The [current legacy contracts](contracts.md#existing-mujoco-and-qwen-contract) preserve its noise-to-data convention and dimensions. Custom interleaving, EO-1 replication and LIBERO are not dependencies of the new roadmap.

The [original website roadmap](https://websitepresentation.vercel.app/roadmap), calendar and 22 subphases are superseded. [History](history.md) provides source retrieval and consequential negative results; the website was not edited.
