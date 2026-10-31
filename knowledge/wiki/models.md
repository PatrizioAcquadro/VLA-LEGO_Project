# Models and learning

## Expanded-corpus learning assignment — October 26, 2026

The completed bounded campaign used robot-pretrained `lerobot/smolvla_base`, with a separate temporal decision classifier. Motor learning has 450,046,176 total parameters and 99,880,992 trainable action-expert/state-projection parameters; the pretrained vision/language backbone stays frozen. The decision head has 962,755 parameters. LeRobot and native environment revisions belong to [compute](cluster.md#native-environments).

| Stage | Executed result | Interpretation |
|---|---|---|
| Preparation | Complete 620 admissions, 500-TRAIN normalization, 1,195 causal original/counterfactual sequences and finite 992-wide frozen features | Readiness/features are separate from optimization and task behavior |
| Separate decision learning | 20k updates; 45/50 true completions, one premature completion, three false abstentions, 6/10 correct impossible-case abstentions; all fifty counterfactuals nonterminal | Calibration fails; the gated sixty-sequence offline confirmation was not executed |
| Independent motor control | 20k updates; effective batch 64; 0/10 integrated physical task successes (0/5 familiar, 0/5 held-out), no unavailable cases or head contacts | Fails the 8/10 advancement target; no full-task extension |
| Final frozen comparison | Current 20k: 0/20; historical i04-15k reference: 0/20; all forty attempts executed once, no unavailable outcomes/setup errors | Both remain negative task results; no qualified physical best |

Each twenty-case panel contains six single-block, eight tower, three conditional recovery and three justified-stop cases. Current/reference false completions are 2/1 and head-contact failures 4/5; current peak head force is 55.43 N versus 5.46 N, so the smaller contact count is not a safety improvement. Teacher prefixes and transient recovery progress receive no autonomous task-success credit.

One control case passes physical acquisition, transport, released placement and stability but never receives a learned completion call. Another declares complete without acquisition; another picks the wrong color. This separates motor components, instruction grounding and termination from integrated success. Bounded existing-data/native CUDA diagnosis found no demonstrated pipeline defect justifying an unchanged rerun. The motor campaign stopped at 20k with 80k of its ceiling unused. The final confirmation predictions are exposed; checkpoints and thresholds remain frozen after comparison.

Detailed results remain in `outputs/data-expansion-20261026`: `final-delivery-1838.json`, `FINAL_REPORT.md`, `decision-interpretation-1733.json`, `motor-control-closure-1764.json` and the physical panel controls/results. Original raw reports, checkpoints and prior failures remain canonical on Blackwell. [History](history.md#earlier-learning-results) preserves the consequential earlier negative outcomes.

## Maintained learning workflow

The current input/action representation is defined in [contracts](contracts.md#current-alex003-contract). Preparation checks complete admission, family separation, real correction coverage and measured signal order before writing learning inputs. Use the compatible LeRobot environment, exposing this checkout's `src` on `PYTHONPATH`.

| Entry point | Purpose |
|---|---|
| `python -m vla_lego.pretrained.prepare_campaign` | TRAIN-only normalization and measured motor interface |
| `python -m vla_lego.pretrained.prepare_decisions` | Frozen features from causal admitted observation histories |
| `python -m vla_lego.pretrained.train_decisions` | Balanced classifier learning, sequence calibration and gated confirmation |
| `python -m vla_lego.pretrained.train_pilot --campaign` | Resumable 16-motor learning with separate decisions and the physical control gate |
| `python scripts/pilot_training_report.py --run RUN --output REPORT` | Recorded offline curves/CSV for current and historical metric formats |

Use `--help` for required corpus, statistics, cases, model and fresh output paths. The required `--campaign` flag retains current callers while preventing a silent return to the superseded embedded-decision trainer. Historical inference/readers remain supported; they do not imply old training producers are maintained. The completed campaign is not automatically resumed by these examples.

The motor recipe remains effective batch 64, bfloat16, AdamW at 1e-4 with betas (0.9, 0.95), epsilon 1e-8, weight decay 1e-10 and gradient clipping 10. There is one 1k warmup and one cosine schedule to a 2.5% multiplier at the 100k motor ceiling. Microbatches divide 64; resumptions preserve optimizer, scheduler, RNG, global draw order, interface and normalization.

The first 20k updates sample single-block control. Later sampling would use 30/50/15/5 single-block/tower/recovery/stop, balanced by family, episode and operational phase, only after the unchanged unassisted 8/10 control gate. That gate failed in the executed campaign. Fixed offline diagnostics occur every 5k updates and preserve executed-prefix versus full-chunk errors and the hold baseline. Offline shortlisting does not promote a physical best. Pruning retains the loaded resume source, fixed milestones and any explicit physical-best checkpoint.

The classifier uses class-balanced TRAIN sampling with natural complete sequences for calibration, reviews every 2k updates and a 20k bound. Temperature/class offsets and persistent thresholds are selected on calibration; the first terminal prediction owns each outcome. Later correct calls cannot erase premature completion. Classifier updates do not consume the motor-update budget. No source or threshold is relaxed to turn a negative outcome into success.

## Planned pretrained VLA selection

SmolVLA is the implemented compact path. Task/onboard selection remains open: compact size, finite predictions and workstation latency do not establish qualified Alex behavior. GR00T was investigated as the larger roadmap candidate; its native inference failed at gated `nvidia/Cosmos-Reason2-2B` backbone access (HTTP 403). The acquired weights/DROID runtime sample do not establish action generation or Alex training admission. That failed path and the original component smoke remain preserved, without credential/access workarounds.

GR00T, released openpi and LingBot-VLA sources remain [research references](references.md), not maintained parallel integrations. Add another model only for a demonstrated task/runtime advantage and compare the same Alex observation boundary, controller and declared cases. Phase 2 requires actual onboard execution, usable WSG32 integration and an instruction-conditioned physical pick/place feasibility result; Phase 5/6 own full cooperative assembly.

## Onboard feasibility

The [Alex guide](https://docs.google.com/document/d/17QtexPK_RqmfRammA7CsvEUuhFZFkicrJfONCdlWkEg/edit?tab=t.sfoy44dcf7kc) describes a Jetson AGX Orin 64 GB on the robot, a Forge carrier and JetPack 6.2.1. As reviewed October 23, 2026, the [GR00T deployment guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/README.md) specifies JetPack 7.2 / Jetson Linux 39.2 / CUDA 13.2 for its current Orin path and excludes JetPack 6.x. That is an unresolved integration constraint, not an instruction to upgrade the robot.

Establish compatibility with the carrier/BSP, ZED drivers, communication and control services before selecting the runtime. A container cannot supply missing host support. If a compatible GR00T path cannot be demonstrated, qualify SmolVLA or another justified smaller path; workstation VLA inference is not an accepted final fallback.

NVIDIA's [hardware guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md) reports approximately 6.6 Hz TensorRT inference on Orin for one camera and four denoising steps. This is external evidence for a particular configuration, not an Alex measurement, actuator rate or guaranteed task responsiveness. Include image acquisition/preprocessing, state transport, policy inference, command delivery, observation age and missed deadlines in the local budget.

The same guide recommends at least 40 GB for its default fine-tuning path and 80 GB for full fine-tuning. The local 24 GB GPU is therefore for inference/profiling or explicitly supported smaller adaptations, not assumed default GR00T training. Use the [compute plan](cluster.md#planned-resource-roles) for larger runs. Select precision, adaptation scope, predicted chunk and executed prefix from measured behavior; no inherited 16-step requirement applies.
