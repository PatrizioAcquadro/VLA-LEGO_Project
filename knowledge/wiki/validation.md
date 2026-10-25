# Validation and evidence

## SmolVLA pilot before and after — October 25, 2026

The bounded experiment is complete: the selected 70,000-update SmolVLA checkpoint and its step-zero reference each executed the same ten marker-free development cases, once per model/case, with matching seeds, instructions, eight-object scenes, physical assumptions and a 600-second whole-case simulation budget. Recovery time includes the recorded physical prefix; a two-second released dwell is unchanged. All twenty attempts are available, with no retries, runtime exceptions, nonfinite predictions or video errors. This is a development result, not Phase 6 qualification.

| Outcome | Step zero | Step 70,000 |
|---|---:|---:|
| Complete nominal tower | 0/6 | 0/6 |
| Complete conditional recovery | 0/3 | 0/3 |
| Justified stop for absent requested piece | 0/1 | 0/1 |
| False abstention on feasible nominal cases | 0/6 | 1/6 |
| False completion on recovery cases | 0/3 | 1/3 |
| Unavailable cases | 0/10 | 0/10 |

The trained policy moved both arms but acquired no grasp-and-lift in any nominal case. Its failure occurs before stacking or final stability scoring. An ordered-support maximum of one includes the base already resting on the table; it is not one learned placement. D1-N1 falsely abstained after 9.45 seconds of policy control. D2-R1 falsely declared completion after 85.55 seconds, adding no level to the three inherited from its teacher prefix. Both models failed to abstain on the missing-piece case.

D1-R2 begins after a 343.3-second teacher prefix with four levels and the fifth piece already above them. Step zero released that piece and obtained stable tower geometry but never signaled completion. Its cooperation event occurred inside the teacher prefix, so this is not learned assembly or cooperation from loose blocks. Step 70,000 reached five supported levels transiently, then lifted a yellow distractor and dismantled the tower. No original outcome or threshold was changed to count either as a successful recovery. Other recoveries made no ordered-level progress beyond handoff.

Review of recorded observations, actions and contacts found no explicit runtime, signal-order or normalization failure: all twenty workers ran on CUDA, and saved processors, normalization and interface are identical before/after. This does not isolate every possible learning defect. The motor development improvement in [models](models.md#active-marker-free-pilot-adaptation--october-24-2026) did not transfer into task success; action sampling, contact acquisition, grounding and decision generalization remain open explanations. Train-start autonomous behavior was not measured, so failure on familiar conditions is not yet distinguished from held-out generalization failure.

The ten cases belong to only two development families, with one seed each. Their counts do not establish a population success rate, and they do not establish the behavior of other checkpoints. Model inference medians were 82.7–84.2 ms on Blackwell, with simulation paused during inference; Orin and real-time timing remain unqualified. The local `outputs/pilot-training-20261024/report.md` contains every case, complete curves, videos, resource use and the proposed [next data decision](data.md#next-data-decision-after-the-first-learned-baseline). Full per-tick traces remain on Blackwell. No new training, 500-episode collection or hardware work is authorized by this result.

## Correction-cohort autonomous review

The first 5,000-update checkpoint trained on the separate 57/10 correction cohort executed four matched loose-start diagnostics: T1-N1, T2-N1, D1-N1 and D2-N1, each once with the full instruction, horizon 10 and a 90-second budget. All four native runs were available and healthy. T1/T2/D1 exhausted the budget; D2 falsely abstained. **No case contacted or acquired its first requested piece, made an ordered placement or completed a cooperative tower.**

D1 provides a narrower positive result: the left jaws autonomously grasped and raised the large red prism by **6.69 cm**, with bilateral grasp-and-lift sustained for **1.95 seconds** near the end of the control budget. That prism is the **fourth requested target**, picked before the first orange cube. Images and actual policy-observation video confirm the held piece. No teacher prefix or hidden assistance was present. This is motor-component progress with incorrect task selection/order; it is neither a successful first acquisition nor candidate promotion.

Every outcome, trace, image and video is retained in `outputs/policy-repair-20261025/iteration-02/panel-005000`; `physical-review-005000.md` owns the detailed interpretation. The 15k checkpoint then completed the same four diagnostics: all exhausted the budget, none falsely abstained, and none contacted the first requested piece or achieved a functional grasp/lift. T1 contacted its fourth yellow-prism target, T2 touched a small red cube distractor instead of the first large red cube, and D1 contacted its fourth red-prism target without repeating the 5k lift. Thus false refusals improved, while the previous motor-component result was not retained; neither checkpoint is promoted for the task. Its complete evidence and interpretation are preserved in the adjacent panel-015000 and physical-review-015000.md.

The completed 30k review adds a first correct-piece component: T2 grasps the requested large red cube with the right jaws and lifts it 3.26 cm, sustaining bilateral grasp-lift for 0.6 seconds before dropping it. All four still fail the task: no ordered placement or cooperative tower, three budget stops and one false abstention in D2. T1 and D2 contact large yellow cube distractors; D1 touches no block. The other first pieces stay untouched. Actual observation video confirms the T2 hold and drop. Full evidence and interpretation remain in panel-030000 and physical-review-030000.md. This transient correct acquisition becomes the component reference without promoting a task candidate. Later interventions must address observed errors using training families; development scenes remain excluded. These diagnostics do not qualify Phase 6, recovery, generalization or Orin timing.

A subsequent **jaw-only overlap test** preserves the exact 30k weights and matched T1/T2 starts, averaging two predictions for the same jaw-command times while leaving fresh arm targets and decisions unchanged. Both native attempts completed once without runtime errors or teacher. T2's correct-piece lift improved to **9.74 cm and 3.3 seconds** of bilateral grasp-lift, followed by another drop; T1 still acquired nothing. Neither made an ordered placement or tower. This is useful retention progress in one training case, not repeatability or task promotion. Full traces, observations and both outcomes are retained in `iteration-03/jaw-overlap`; `jaw-overlap-review.md` owns the comparison. The identical frozen candidate proceeds once to D1/D2 in a separate development panel; the training starts are not retried.

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

The default `pose_dwell` criterion measures actual block displacement during the same continuous two-second released interval. Every corner of each requested block must remain within **3 mm of its position at the start of that interval**, evaluated at the 20 Hz control samples. Tracking corners includes translation and rocking without separate angular margins. This is a provisional functional simulation tolerance, consistent with millimeter-scale harmless variation, not a hard safety bound or measured print range. Static placement offsets are accepted when order, physical support and assembly-square geometry remain valid. Task-specific precision and genuine safety constraints still apply; these tolerances do not govern insertion tasks.

This criterion follows the user's direction to avoid disproportionate work on physically harmless deviations. It replaces the instantaneous speed gate, which rejected a supported tower whose largest corner displacement was only **0.169 mm**. The original `solver_velocity` and `pose_step_peaks` modes remain available for reproducing historical outcomes and motion diagnostics. PhysX documents that constraint-solver velocity can differ from transform motion; even actual high-frequency speed peaks can be unsuitable proxies for whether a tower remains standing. [PhysX constraint-solver documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Simulation.html).

Explicit offline reassessment of the existing ninth reference and its exact native replay passes the current functional rubric, including three seconds of physical cooperation, ordered support and complete release. This reused recorded poses and contacts without another simulation or physical-parameter changes. Original reports and recording labels remain unchanged; the separate reassessment identifies both criteria. The seventh reference remains rejected after its collapse; the eighth remains rejected for missing qualifying cooperation. The tenth orientation experiment was interrupted as unnecessary microstability work and remains incomplete. These are development teacher results, not learned performance or qualification; positive training admission must explicitly consume the reassessment rather than silently relabel raw evidence.

Twenty-one analytical regressions cover the historical scoring/motion behavior and the functional criterion: harmless 2 mm displacement and placement offset, centimeter-scale drift, rocking, loss of support and incomplete release. These fixtures are logic tests, not physical demonstrations. Native construction and print/contact calibration remain separate requirements.

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
