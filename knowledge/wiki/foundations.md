# Isaac runtime and operations

## Ownership and environments

Use Isaac Sim/Lab with shared Alex assets for the fixed-base Purdue Alex003, pedestal, WSG32/UMI grippers and head ZED. This repository owns task scenes, teachers, control adaptation, recording and evaluation. Do not copy shared robot assets into another authority. [Architecture](architecture.md) owns hardware responsibilities; [compute](cluster.md#native-environments) identifies validated revisions.

Use NVIDIA Isaac Lab at `67fe12806afbdb8bf8e125866661dd201088a54d` in a
dedicated Python 3.12 environment. Use the AlexDoor lock correction for the
NumPy/Numba pair. Install the upstream `isaacsim` extra in that environment
to retain the RTX camera renderer. The following command uses a dedicated
Isaac Lab checkout with the corrected lock already applied.

```bash
runtime="$HOME/.local/share/isaac-steel-rtx"
UV_PROJECT_ENVIRONMENT="$runtime/.venv" uv --directory "$runtime/IsaacLab" \
  sync --locked --python python3.12 --extra isaacsim
```

Follow the [pinned checkout and lock setup](https://github.com/PatrizioAcquadro/AlexDoor-XAS/blob/main/scripts/install-isaac-native.sh)
for the Isaac Lab source. `ISAAC_NATIVE_RUNTIME` selects another installation.
The launchers use the sibling `Alex` checkout unless `ALEX_ASSETS_ROOT` is set.
The Alex package must include support for Isaac Lab's native actuator limits
and the official ZED asset.

LeRobot runs in its own environment. `VLA_VIDEO_PYTHON` can select the
PyAV encoder environment when FFmpeg is unavailable.

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

Policy correction windows retain both the requested mechanism and the physically observed error. An actual bilateral lift of another piece remains a wrong-piece event even when a missed grasp was requested. Released prefix loss requires settled evidence before rebuilding. Recovery uses measured source poses, admits displaced-piece approaches and empty-hand retirement before motion, and withdraws the stabilizer before rebuilding. These maintained paths and their regression tests do not grant new corpus admission or whole-task qualification.

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

## Physics profiles

The Isaac workflows use NVIDIA's `NewtonCfg` with `SteelSolverCfg` and
CUDA Graph. The default uses Newton Steel's persistent friction patches
(`friction_anchor_beta=0.2`) and records `newton_steel_native_patch_friction`.
The RGB camera uses Isaac RTX and contact readback uses Isaac Lab sensors.
Set `ISAAC_PHYSICS_PROFILE=newton_steel_native` to replay the earlier native
Steel recordings with point friction, or `ISAAC_PHYSICS_PROFILE=isaacsim_physx`
for PhysX. Replay gates continue to require the recorded physics profile.

In the second pickup of `TR-T22-04-1`, the green cube slid through the WSG32
jaws and fell despite the wrist reaching the requested lift pose. With the
same jaw target and recorded approach, persistent patches retained the cube
97.6 mm above its resting height after a one-second hold; point friction
returned it to the table. Greater jaw closure did not correct the slip.

The earlier compatibility-profile comparison used `tower_reference`, the original
`PickPlaceEvaluator`, and the same FFV1 recording configuration for both
backends. `physics_profile`, `physics_cuda_graph`, and the task-plus-recording
wall time after initial settling are recorded in the report. A planner
exception before the released dwell remains a failed attempt.

On November 12, 2026, both backends passed all five evaluator checks for
`TR-P04-00-0` with event settling. Steel completed the recorded task in 85.86 s
with 50.20 simulated seconds. Native PhysX took 152.56 s with 50.35 simulated
seconds on the same RTX 4090. The launcher then selected Steel automatically
and passed the same evaluator again in 88.94 s. Released-hand retreat planning
refused the left-hand `TR-P01-00-1` reference on both engines, including with
event settling. The matched timing and completion evidence covers the
single-block reference.

On November 15, 2026, the default persistent-friction profile passed the
existing evaluator on the complete `TR-T22-04-1` five-block reference on
the elab2 RTX 4090. The launcher used Alex003, WSG32 and the head ZED RGB
stream at 20 Hz, with CUDA Graph and no physics override. The evaluator
accepted the ordered, supported tower with both hands withdrawn and
confirmed functional cooperation. During the final released dwell, the
largest corner displacement across the five pieces was 1.60 mm against
the existing 3 mm limit. This is a simulated teacher/controller result for
this nominal case. Learned execution and hardware qualification remain open.
