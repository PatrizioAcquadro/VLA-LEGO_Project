# Validation and evidence

## Assembly qualification targets

These are initial roadmap acceptance targets, **not achieved results or guarantees from external benchmarks**. The [task definition](bricks.md#target-task) governs success: correct five pieces and order, meaningful stabilize–place cooperation, full release and stability through a calibrated interval. Use evidence of each arm's task contribution rather than counting motion or the presence of two arms.

| Campaign | Initial acceptance target |
|---|---|
| Nominal simulation | At least 90 successful episodes out of 100 |
| Dedicated generalization | At least 80 out of 100 held-out episodes; report color–shape combinations, dimensions, layouts and instructions separately |
| Recoverable-error simulation | At least 70 out of 100 episodes with predefined recoverable errors |
| Physical robot after adaptation | At least 40 out of 50 episodes across at least two sessions, with VLA inference entirely on Orin |

Measure the physical zero-adaptation baseline separately, after required calibration and hardware readiness but before learning from real task demonstrations. Predefine its bounded episode set; no numerical success guarantee is assumed. Adapt on separate real demonstrations/corrections, then evaluate on independent episodes. A workstation policy server does not satisfy the physical criterion.

Before each campaign, fix the task distribution, distractors, required cooperative roles, timeout, allowed recovery attempts, perturbation types/severity and physical acceptance ranges. Calibrate pose, speed and dwell tolerances against meaningful measurement error and print/contact variability. Distinguish hard robot safety limits, empirical acceptance ranges and numerical tolerances; avoid arbitrary stacked margins and unjustified refusals.

Simulated qualification includes measured Orin-equivalent observation age, inference/replanning and command timing. Report successes/attempts, confidence intervals, completion times, placement/recovery errors, interventions, stops and failures. All attempted episodes remain visible; favorable retries cannot replace failed trials. Report unavailable episodes explicitly instead of silently shrinking the denominator. Non-completable cases need appropriate-stop and unnecessary-refusal reporting, separate from the recoverable-error success rate.

Qualification is autonomous: no teacher coordinates, future frames, hidden corrective controller, object forces or human intervention counted as learned success. Preserve results when revising a model; if qualification episodes inform training, use fresh independent episodes for the next claim. Do not adjust thresholds merely to pass a campaign. Simulation success permits controlled transfer work, not a claim of physical safety certification or performance.

## Documentation revision — October 23, 2026

The new roadmap contains nine plans with the requested five-part structure and no software recipes. All 121 local Markdown links/anchors resolve. Ruff, Black on all 65 versioned Python files, mypy on 36 source files and all eight existing configuration combinations pass; runtime code, assets and packaging were not changed or requalified.

The full `make check` invocation encounters two pre-existing, ignored history-preparation scripts under `outputs/history-messages-20261023/` that Black would reformat. Those local artifacts were preserved; the remaining checks were completed separately. This is not a failure of the versioned documentation or Python formatting. No simulator, training, cluster or robot qualification campaign was run.

## Existing local component evidence — October 23, 2026

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
