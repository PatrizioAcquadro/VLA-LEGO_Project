# Validation and qualification

## Evidence boundaries

Report software implementation, native component behavior, admitted demonstrations, learned task outcomes and hardware qualification separately. The completed campaign's measured results live in [models](models.md#expanded-corpus-learning-assignment--october-26-2026); earlier negative outcomes and retired checks are summarized in [history](history.md). Neither a clean repository nor a lower offline loss qualifies assembly.

The current evaluator requires instructed pieces/order, real task-relevant overlapping stabilization and placement, complete release and supported stability. Functional stability measures actual displacement over the two-second released dwell, with the current provisional 3 mm corner-motion bound. Instantaneous solver microvelocity alone is not physical drift. Preserve hard joint/collision/contact limits separately from empirical task calibration and numerical tolerances; printed objects and physical measurements still require calibration.

Repeated favorable retries, setup-only checks, teacher prefixes, reset-to-goal scenes and conditional progress cannot earn autonomous task success. An exact replay reproducing failure remains failure. Explicit reassessments retain the original result and identify the changed criterion rather than rewriting raw labels.

## Assembly qualification targets

These are initial roadmap acceptance targets, **not achieved results or guarantees from external benchmarks**. The [task definition](bricks.md#target-task) governs success: correct five pieces and order, meaningful stabilize–place cooperation, full release and stability through a calibrated interval. Use evidence of each arm's task contribution rather than counting motion or the presence of two arms.

| Campaign | Initial acceptance target |
|---|---|
| Nominal simulation | At least 90 successful episodes out of 100 |
| Dedicated generalization | At least 80 out of 100 held-out episodes; report color–shape combinations, dimensions, layouts and instructions separately |
| Recoverable-error simulation | At least 70 out of 100 episodes completing the full tower after a predefined recoverable error |
| Physical robot after adaptation | At least 40 out of 50 episodes across at least two sessions, with VLA inference entirely on Orin |

All three simulation rows must pass for the same checkpoint and execution configuration: 300 distinct qualification episodes in total. The percentage targets apply to observed success counts, not to confidence-interval lower bounds; report uncertainty alongside them. Fix the distribution of held-out axes and recovery families before training exposure to the final sets. Cover missed grasp, slip, off-center placement, wrong piece, occlusion, misalignment and interrupted coordination in the recovery campaign; report subgroup results so an aggregate does not conceal missing coverage.

Measure the physical zero-adaptation baseline separately, as defined below. The adapted 50-episode campaign uses one fixed checkpoint/configuration across at least two sessions; no between-session learning or performance-driven retuning within that campaign. Declare session allocation beforehand and report each session. A workstation policy server does not satisfy the physical criterion.

Before each campaign, fix the task distribution, distractors, required cooperative roles, timeout, allowed recovery attempts, perturbation types/severity and physical acceptance ranges. Calibrate pose, speed and dwell tolerances against meaningful measurement error and print/contact variability. Distinguish hard robot safety limits, empirical acceptance ranges and numerical tolerances; avoid arbitrary stacked margins and unjustified refusals.

Simulated qualification includes measured Orin-equivalent observation age, inference/replanning and command timing. Report successes/attempts, confidence intervals, completion times, placement/recovery errors, interventions, stops and failures. A rollout begins when autonomous execution is enabled with the instruction and initial scene. A timeout, unrecovered error or intervention after that point is a non-success for that episode. Setup failures before rollout are unavailable cases reported separately; they do not earn success, reduce the required campaign size or permit choosing an easier replacement. Use a predefined replacement rule for unavailable cases.

Non-completable cases receive a separate appropriate-stop/unsafe-continuation report, with unnecessary refusal measured on feasible cases. There is no additional numerical stop-rate target in the approved roadmap; absence of one does not waive hardware limits or allow those cases to be counted as successful towers. Predetermine the assessed cases and report all outcomes.

Qualification is autonomous: no teacher coordinates, future frames, hidden corrective controller, external object assistance or human intervention counted as learned success. Declared adverse disturbances may create recovery cases as described in [data](data.md#teacher-and-demonstration-generation); they may not help the robot complete the task. Preserve results when revising a model; if qualification episodes inform training, use fresh independent episodes for the next claim. Do not adjust thresholds merely to pass a campaign. Simulation success permits controlled transfer work, not a claim of physical safety certification or performance.

## Physical baseline and evaluation

Zero-adaptation excludes learning from real task demonstrations/corrections and outcome-driven policy retuning. It permits the mechanical, camera, frame and low-level control calibration needed to run the robot, declared before freezing the baseline. Controlled hardware readiness is separate from the scored baseline. If readiness requires task-specific policy adaptation, report the no-adaptation baseline as unavailable and explain why; do not rename an adapted result as zero-adaptation.

Choose the baseline's bounded episode count and distribution before it starts, retain the unchanged candidate and report successes, failures and interventions without pooling them with adapted results. Baseline episodes may inform later adaptation only after baseline reporting; final adapted evaluation uses separate episodes. The real qualification target applies after adaptation, so a poor baseline is a result to learn from, not a reason to erase it.

Physical evaluation uses recorded observations and a predefined scoring rubric for identities/order, cooperative overlap, complete release and dwell stability. Manual review is acceptable when automated estimates are insufficient; evaluator-only observations may not guide policy actions. Establish in advance how each condition is observable. If a started episode cannot be scored, report it as unscorable/non-success rather than assuming success or replaying until favorable. Between-episode reset and normal session initialization are allowed; hidden mid-episode assistance is not.

## Software and integration checks

[Development](development.md#checks) owns the portable suite and installed-wheel checks. It protects masks/causality, family separation, physical-gate contracts, checkpoint retention, decision persistence, failure accounting and recorded-metric reporting. Mocks isolate these contracts; native physics and pretrained models require their actual GPU environments.

The October 31 cleanup compared baseline and simplified motor code on the same two existing TRAIN samples on native Blackwell CUDA. Preprocessing, predictions, loss and all 155 gradient tensors matched exactly; no optimizer updates occurred. This validates the exercised refactor path, not learning effectiveness, full trainer resume execution, simulation dynamics or physical transfer. Evidence remains in `outputs/repository-cleanup-20261031/gpu-parity.json`.

The recovery integration in `0f8f554` passed a bounded Blackwell CUDA regression restored from `TR-R-placement-02-60-0`, frame 2516. Checked empty-hand retirement and the original next-source edge preserved the physically supported two-level prefix, with no head, other-piece or nonpiece robot contacts; the retained images were reviewed. This verifies that component on maintained source, without granting whole-task, corpus or learned-policy credit. Reports and native traces remain in `outputs/isaac-recovery-integration-20261031/` on the project hosts.

Hosted CI status belongs to the [actual commit](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/actions/workflows/ci-cd.yml). Missing codecs, unavailable hardware and skipped checks are reported explicitly. No Orin timing, real printed-block behavior, physical WSG32 integration, distributed-training scaling or robot qualification is established by the cleanup.
