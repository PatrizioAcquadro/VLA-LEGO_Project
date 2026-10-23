# Validation and evidence

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

## Documentation revision — October 23, 2026

The roadmap contains nine plans with the requested five-part structure and no software recipes. The clarity review confirms all 122 local Markdown links/anchors resolve and makes task semantics, handoffs, decision ownership and project completion explicit. The prior revision passed Ruff, Black on all 65 versioned Python files, mypy on 36 source files and all eight existing configuration combinations; runtime code, assets and packaging remain unchanged and were not requalified during the clarity review.

The full `make check` invocation encounters two pre-existing, ignored history-preparation scripts under `outputs/history-messages-20261023/` that Black would reformat. Those local artifacts were preserved; the remaining checks were completed separately. This is not a failure of the versioned documentation or Python formatting. No simulator, training, cluster or robot qualification campaign was run.

## Existing local component evidence — October 23, 2026

The new native `isaac.evaluation.TowerEvaluator` scores privileged history independently of policy control. Its provisional simulation rubric checks loose initial pieces, visible-attribute matching with interchangeable equivalent instances, ordered physical support, footprints inside the assembly square, all-object robot release and two seconds of sampled stability. Cooperation is detected from actual bilateral hand contacts, an unsupported upper approach and load transfer while the lower structure is constrained; teacher stage labels do not determine the result. These component ranges await full-task and print/contact calibration before qualification freeze.

Ten analytical unit cases passed, covering interchangeable pieces, wrong color/size/order, prearranged goals, decorative grasping, held distractors and insufficient dwell evidence. These fixtures are logic tests, not physical demonstrations. Separate replay scoring rejected retained native `tower-reference-003`, `004` and `005`: their partial robot sequences did not establish the required full tower and cooperation. No passing assembly or qualification result follows from the scorer implementation alone.

The results below predate this roadmap revision and apply to the retained MuJoCo/Qwen implementation. They were not rerun as Isaac/Alex003 or pretrained-policy experiments.

| Check | Result and boundary |
|---|---|
| Maintained test suite | 481 passed with CUDA/EGL, including 32 native model GPU tests. Obsolete configuration-only tests and unused model-helper tests were removed. |
| Model smoke | Qwen3.5-4B on RTX 4090: native four-view logits `(1,426,248320)`, finite VLA actions `(1,16,17)`, synthetic text-only gradient-routing probe; 9.24 GiB peak allocated GPU memory. |
| Simulation demo | Native robot-motion, prearranged tower and triangle clips exported and images visually inspected. No robot assembly or learned-task success measured. |
| Static/config quality | Ruff, Black source formatting, mypy across 36 maintained source files, shell syntax and eight Hydra model/location combinations passed in the final maintained source. |
| Installed package | Built sdist, then wheel, imported from temporary installed location outside checkout; four scenes, configuration, tracking, brick evaluation, five CLIs and asset export/lint passed. Caller runtime dependencies reused. |
| GPU-less CI path | 449 CPU/OSMesa tests passed in the dependency container; 32 GPU tests explicitly deselected. |
| Docker | Local image resolves dependencies, passes `pip check`, detects RTX 4090 and exports all three diagnostics with Python 3.12/MuJoCo 3.17. Host checks use Python 3.10/MuJoCo 3.5. |

Ignored local evidence directories include `outputs/maintained-demo/`, `outputs/container-final-20261023/`, `outputs/model-native-smoke-mm/`, `outputs/container-build-20261023/` (including `osmesa-final.log`), `outputs/cleanup-final-maintained-tests.log` and `outputs/package-final-maintained.log`. These are local artifacts, not published or tracked results. Commands to reproduce the capabilities live in [development](development.md), [simulation](simulation.md) and [models](models.md).

## Preserved failures and unavailable checks

The first native multimodal smoke failed because current Transformers required image token-type metadata. Its report remains in `outputs/model-native-smoke/`; the corrected wrapper and regression test preserve explicit metadata and derive it for older image-only callers. The later success does not erase that failure.

The first Docker build encountered an externally managed Python installation; the image now installs in a venv. Dependency checking then exposed an unused inherited build-tool constraint; removing that tool made `pip check` pass. Build logs retain both diagnosis and corrected result. The first CPU/OSMesa test attempt also exposed a missing `USER` assumption in the Gilbreth preset; portable location fallback fixed it, and all 449 selected tests then passed.

Not verified here: interactive GUI operation, independent SDK FK parity, physical printed-brick grasps/dynamics, a real demonstration dataset, multimodal training convergence, learned assembly success, A100 memory/throughput, multi-GPU resume, Apptainer, SLURM execution or container/release publication. Full-backbone training memory cannot be inferred from frozen inference memory.

Historical robotics and model measurements are summarized in [history](history.md). Do not present the former LEGO retention proxy, historical “complete” reports or a synthetic loss decrease as evidence of lightweight-brick policy readiness.

## Future experiment reporting

Report implementation, component validation, task performance and physical qualification separately. Preregister seed/task sets, attempted episodes, controller assistance, stopping conditions and evaluator tolerances. Retain failed/unavailable outcomes in denominators. Calibrate empirical geometry/contact thresholds before a benchmark; never relax them only to turn a failed case into success.

## Source layout verification

After migration to `src/vla_lego`, the complete native CUDA/EGL suite passed all 481 tests, including the 32 native model GPU checks. The installed wheel contained exactly the 71 expected Python/configuration/asset files from `src/`, and its probe passed outside the checkout. `make check` passed with the new namespace, subsystem test directories and that revision's 100 local documentation links. Evidence: `outputs/src-native-tests.log`, `outputs/src-package-check.log` and `outputs/src-quality-check.log`.

The dependency-only Docker image was rebuilt with `/workspace/src` exposure and exported the robot/tower/triangle GPU diagnostics successfully (`outputs/src-container-demo/`). All five installed CLIs resolve to `vla_lego.*` and the strict asset linter passes. Obsolete generated root-level egg metadata was removed after it was found shadowing current entry-point metadata; no tracked evidence was changed.

The earlier feature branch was published and subsequently integrated. Hosted CI runs on branch pushes; [GitHub Actions](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/actions/workflows/ci-cd.yml) is the source for a revision's status. This documentation revision performs no remote publication or hosted validation.

Root-file cleanup preserved that runtime unchanged. `make check`, the 71-file installed-wheel probe, all five CLIs and the relocated Docker build passed; that revision had 94 valid local documentation links. Container definitions now live in `infra/containers`. Evidence: `outputs/essential-root-check.log`, `outputs/essential-root-package.log` and `outputs/essential-root-docker-build.log`.
