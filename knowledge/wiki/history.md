# Historical findings and retired work

Ordinary code, plans and run-specific instructions are retained by Git. The earlier cleanup's source baseline is `c8e0bcd36550d740801dc0bb33b2fc19e5e8a0de`; historical paths below can be read with `git show REV:path` without restoring obsolete code into the active tree.

## Consequential findings

| Historical source at baseline | Finding to preserve | Current interpretation |
|---|---|---|
| `docs/phase1.1-assessment.md` | March 3 assessment called the seven robot integration sections complete; initially used two views. | The legacy MuJoCo camera contract later became four views. Neither that contract nor a completion label qualifies Alex003 hardware. |
| `docs/kinematics-validation-report.md` | Self-consistency, symmetry, workspace and joint-axis checks passed. Independent SDK FK was skipped because the original full-inertia encoding failed to load. | Keep the skipped comparison open; self-consistency is not independent parity. |
| `docs/contact-tuning-notes.md`, `docs/press-fit-spec.md` | Pure contact retained about 0.06 N/stud against an original 0.3 N/stud target. Thresholds were later relaxed; hybrid retention added a weld/spec proxy. | Original material-faithful retention was not demonstrated. Proxy success is distinct. Neither result qualifies real LEGO or current bricks. |
| `docs/phase3.1-gpu-validation.md`, `docs/phase3.1-3.2-report.md` | RTX 4090 backbone/head component checks and memory observations; A100 characterization remained pending. | Historical measurements only. Old dependency/config descriptions, training feasibility estimates and Phase 3.3 references are superseded by current plans. |
| Former random dataset/Transformer/trainer/evaluator and cluster jobs | Infrastructure exercised synthetic tensors and toy checkpoints. | Retired because it did not train or evaluate the maintained VLA on a real task. Its multi-GPU results do not qualify a future VLA trainer. |

Robot extraction sources and transformations remain fully tracked in [asset provenance](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/blob/v1.0.0/src/vla_lego/sim/assets/robots/alex/PROVENANCE.md). They have a continuing scientific and maintenance purpose.

## Completed repository cleanup

The earlier cleanup retired synthetic training/evaluation and LEGO retention mechanisms, consolidated diagnostics, and moved the maintained runtime into one installable namespace. Its source-layout, package and container results remain in [validation](validation.md); consequential decisions remain in the [log](log.md).

The completed implementation specification is recoverable at `445536818b399f456a38aea16e788c3d94da59de:knowledge/repository-cleanup.md`. Its former branch/publication permissions and phase numbering are historical, not instructions for future work. Ordinary cleanup chronology is kept in Git rather than duplicated in the active wiki.

## Recovering original plans

```bash
git show c8e0bcd36550d740801dc0bb33b2fc19e5e8a0de:docs/roadmap/phase1.2.md
git show c8e0bcd36550d740801dc0bb33b2fc19e5e8a0de:docs/roadmap/phase2.1.md
git show 4670ec5ca1d70fac50791cd24139c76aaf134747:docs/roadmap/phase0.2.md
git show 57e3ef3903ab16d41adc544a8651b73979ad7eff:docs/roadmap/phase1.1.md
```

The other detailed plans `phase2.2`, `phase2.3`, `phase3.1`, `phase3.2`, `phase4.1` and `phase4.2` also exist at the baseline under `docs/roadmap/`. The reconciled 22-subphase MuJoCo/Qwen roadmap is recoverable at `445536818b399f456a38aea16e788c3d94da59de:knowledge/wiki/implementation_phases/`. The [current nine-phase roadmap](implementation_phases/index.md) replaces that sequence rather than preserving its numbering or completion labels. Draft layouts, tuning procedures and obsolete recipes are not copied into a second archive.
