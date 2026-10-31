# Isaac runtime and operations

## Ownership and environments

Use Isaac Sim/Lab with shared Alex assets for the fixed-base Purdue Alex003, pedestal, WSG32/UMI grippers and head ZED. This repository owns task scenes, teachers, control adaptation, recording and evaluation. Do not copy shared robot assets into another authority. [Architecture](architecture.md) owns hardware responsibilities; [compute](cluster.md#native-environments) identifies validated revisions.

Set `ISAAC_SIM_DIR`, `ISAAC_LAB_DIR` and `ALEX_ASSETS_ROOT` to their compatible installations. Set `VLA_ISAAC_PYTHON` only when the verified native environment uses a separate interpreter. `scripts/isaac-run.sh` exposes this checkout, the shared Alex package and Lab extensions before executing Isaac's Python launcher. LeRobot runs in its own environment; `VLA_VIDEO_PYTHON` may select that environment for the existing PyAV encoder when FFmpeg is unavailable.

## Native foundation probe

```bash
bash scripts/isaac-run.sh -m vla_lego.isaac.smoke \
  --output-dir outputs/native-smoke --cache-dir outputs/converted-assets
```

Use a fresh output directory. The probe checks actual CUDA, finite state, fixed base, jaw movement, table support and a nonblank head image. `scene_probe` checks loose task inventory; `grasp_probe` executes a single-block contact-only grasp/lift/transport/release component. Native imports occur after AppLauncher. Every probe retains failure details and exits nonzero on failure; component passes do not establish cooperative assembly.

## Teacher collection

`tower_reference` executes task-specific contact-based references and the independent online tower evaluator. Inspect its maintained interface with:

```bash
bash scripts/isaac-run.sh -m vla_lego.isaac.tower_reference --help
```

For declared expanded cases, provide `--campaign-cases`, `--case`, a fresh output directory, cache directory, `--record` and optionally `--rgb-storage ffv1`. The case defines family/partition, instruction, variation and correction mechanism. Historical CSV cases and exact-prefix continuation remain supported where required by existing evidence. `pilot_stop`/`stop_reference` represent justified inability without replacing a missing requested piece with an easier task.

Admit a recording only after physical completion, full sensor/action/mask readback and image review. Keep the exact executed generator with each attempt. A component repair, reset-state diagnostic or successful over-quota attempt does not promote a source revision or fill a different case. [Data](data.md#teacher-and-demonstration-generation) owns corpus admission and coverage.

## Replay and independent review

- `isaac.replay` compares recorded single-block commands, state and optionally images under the original conditions.
- `isaac.tower_replay` reconstructs full references; alternate solver iterations require the original baseline replay first.
- `isaac.score_trace` scores an existing full trace into a fresh output, preserving the original report and marking a changed criterion as a reassessment.
- `pilot_audit` verifies recorded alignment, membership and error masks; `PilotEpisode` reads admitted PNG/FFV1 observations and valid future-action windows.

Replay agreement and admission are separate. An exact replay of a failed reference remains failed. The ninth historical tower reference replay reproduced its original state exactly and preserved its velocity-gate failure; an explicit functional dwell reassessment later passed while retaining the original negative report. This finding supports displacement-based stability assessment, not erasing failures or qualifying a learned policy.

## Learned execution

`scripts/run_policy_probe.py` starts the LeRobot worker and native simulator through a private socket with bounded owned process groups. `scripts/evaluate_campaign_panel.py` executes the declared control/comparison panel once per model/runtime/case and preserves failure/unavailable denominators. Their `--help` documents required corpus, checkpoint, protocol, interpreter and cache arguments. Existing model/case directories are evidence and are not an invitation to retry.

The worker consumes only the [declared observations](contracts.md), checks checkpoint semantics and reports actual CUDA. Teacher prefixes, conditional handoffs, physical task outcomes and learned terminal calls remain separately labeled. Rendering/model validation needs the compatible GPU stacks; portable software tests do not replace it.

## Physical and source limits

Current object masses/friction and table dimensions are simulation assumptions. Contact readback activates nested articulated links; unsupported static-table GPU pair filtering is not reported as measured table-only force. TCP geometry, IK/FK checks and collision admission protect executed motion, while full-task source acceptance requires the original physical sequence and data review. Follow [validation](validation.md) for hard limits, empirical calibration and task-success interpretation.

The MuJoCo/Qwen runtime and its five CLI commands are retired from main. [History](history.md) identifies their fixed release and attribution; all original native failures, recordings and frozen producer sources remain preserved.
