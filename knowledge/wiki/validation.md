# Validation and evidence

## Current local evidence — October 23, 2026

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

Not verified here: interactive GUI operation, independent SDK FK parity, physical printed-brick grasps/dynamics, a real demonstration dataset, multimodal training convergence, learned assembly success, A100 memory/throughput, multi-GPU resume, Apptainer, SLURM execution, hosted CI or publication. Full-backbone training memory cannot be inferred from frozen inference memory.

Historical robotics and model measurements are summarized in [history](history.md). Do not present the former LEGO retention proxy, historical “complete” reports or a synthetic loss decrease as evidence of lightweight-brick policy readiness.

## Future experiment reporting

Report implementation, component validation, task performance and physical qualification separately. Preregister seed/task sets, attempted episodes, controller assistance, stopping conditions and evaluator tolerances. Retain failed/unavailable outcomes in denominators. Calibrate empirical geometry/contact thresholds before a benchmark; never relax them only to turn a failed case into success.

## Source layout verification

After migration to `src/vla_lego`, the complete native CUDA/EGL suite still passes all 481 tests, including the 32 native model GPU checks. The installed wheel contains exactly the 71 expected Python/configuration/asset files from `src/`, and its probe passes outside the checkout. `make check` passes with the new namespace, subsystem test directories and 100 current local documentation links. Evidence: `outputs/src-native-tests.log`, `outputs/src-package-check.log` and `outputs/src-quality-check.log`.

The dependency-only Docker image was rebuilt with `/workspace/src` exposure and exported the robot/tower/triangle GPU diagnostics successfully (`outputs/src-container-demo/`). All five installed CLIs resolve to `vla_lego.*` and the strict asset linter passes. Obsolete generated root-level egg metadata was removed after it was found shadowing current entry-point metadata; no tracked evidence was changed.
