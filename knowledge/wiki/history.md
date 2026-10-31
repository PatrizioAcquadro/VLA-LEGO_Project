# Historical findings and retired work

Ordinary source, plans and run instructions remain in Git. `v1.0.0` fixes commit `445536818b399f456a38aea16e788c3d94da59de`; it preserves the retired MuJoCo/Alex V1/EZGripper, Qwen, Hydra, tracking, five CLI commands and deployment infrastructure. Main is unreleased v2 development. No source history or released tag was rewritten by the v2 cleanup.

## Consequential findings

| Earlier evidence | Finding and continuing interpretation |
|---|---|
| MuJoCo kinematics | Self-consistency/symmetry checks passed; independent SDK FK parity was unavailable because the original inertia encoding failed to load. Self-consistency is not independent validation. |
| LEGO contact experiments | Pure contact retained about 0.06 N/stud against the original 0.3 N/stud target. Later weld/spec proxies did not establish material-faithful retention or current physical LEGO capability. |
| Synthetic trainer/evaluator | Toy tensors and checkpoints exercised infrastructure, not maintained task learning. Their multi-GPU results do not qualify the SmolVLA trainer. |
| Qwen/model/package checks | Historical component GPU and installed-package evidence remains tied to its original revision; it does not qualify Isaac/Alex003, Orin or the current task. |
| Ninth native tower reference | Exact replay preserved the original velocity-gate failure; explicit functional displacement reassessment passed while retaining the negative original. Small solver microvelocities are distinct from physical drift. |
| Frozen collection repairs | Whole-original physical/data/image proof admits only the exact exercised runtime. Component-only, unexercised or over-quota variants remain separate. |

Legacy robot attribution remains at `v1.0.0:src/vla_lego/sim/assets/robots/alex/PROVENANCE.md`. Retained RGB-Objects attribution remains [beside the active block assets](../../src/vla_lego/isaac/assets/blocks/PROVENANCE.md). Raw/native evidence and producer snapshots are preserved outside Git; [compute](cluster.md) identifies storage.

## Earlier learning results

| Experiment | Preserved outcome |
|---|---|
| 50+10 marker-free pilot, step zero versus selected 70k | Both fail all ten development cases: six nominal, three conditional recovery and one stop. All twenty attempts executed once; trained false abstention/completion remain failures. Lower offline error did not become task success. |
| Correction cohort, iteration-02 | Four-case reviews at 5k/15k/30k remain task failures. A 5k wrong-order red-prism lift and a 30k correct first-cube lift/drop are component evidence, not assembly. |
| Jaw overlap, iteration-03 | The frozen 30k candidate improves one TRAIN correct-piece hold to 9.74 cm/3.3 s before dropping it. No placement/tower; held-out first-piece acquisition remains absent. |
| Paired-instruction cohort, iteration-04 | At 15k one TRAIN case autonomously picks, carries and releases the first requested piece, then falsely declares the whole tower complete. No task promotion. The 30k review loses acquisition; all four remain failed. |
| Expanded campaign | Current 20k and historical i04-15k each score 0/20. [Models](models.md#expanded-corpus-learning-assignment--october-26-2026) owns this final result and comparison limits. |

The original pilot panel used two development families and one seed each; these small panels do not establish population rates or Phase 6 generalization. Conditional teacher progress and unsupported initial blocks are not learned placements. Detailed sources remain in `outputs/pilot-training-20261024/report.md` and the `outputs/policy-repair-20261025/iteration-*` reports, observations, traces, videos and checkpoint archives.

## Retrieving retired source

```bash
git show v1.0.0:src/vla_lego/sim/assets/robots/alex/PROVENANCE.md
git show v1.0.0:knowledge/repository-cleanup.md
git show c6f7fe2:src/vla_lego/pretrained/train.py
git show c6f7fe2:src/vla_lego/pretrained/pilot_training_data.py
```

The former 22-subphase roadmap is available at `v1.0.0:knowledge/wiki/implementation_phases/`; the [nine-phase roadmap](implementation_phases/index.md) supersedes its numbering and completion labels. Earlier assessments/contact reports are recoverable at `c8e0bcd:docs/`. Do not restore run-specific code, permissions or stale operating instructions as current requirements.
