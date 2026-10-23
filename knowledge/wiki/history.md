# Historical findings and retired work

Ordinary code, plans and run-specific instructions are retained by Git. The baseline for this cleanup is `a5205b297ab27f68b788aea0c2084bc259c7cd33`; historical paths below can be read with `git show REV:path` without restoring obsolete code into the active tree.

## Consequential findings

| Historical source at baseline | Finding to preserve | Current interpretation |
|---|---|---|
| `docs/phase1.1-assessment.md` | March 3 assessment called the seven robot integration sections complete; initially used two views. | Robot/control interfaces remain. The current camera contract is four views. A report's completion label does not qualify hardware. |
| `docs/kinematics-validation-report.md` | Self-consistency, symmetry, workspace and joint-axis checks passed. Independent SDK FK was skipped because the original full-inertia encoding failed to load. | Keep the skipped comparison open; self-consistency is not independent parity. |
| `docs/contact-tuning-notes.md`, `docs/press-fit-spec.md` | Pure contact retained about 0.06 N/stud against an original 0.3 N/stud target. Thresholds were later relaxed; hybrid retention added a weld/spec proxy. | Original material-faithful retention was not demonstrated. Proxy success is distinct. Neither result qualifies real LEGO or current bricks. |
| `docs/phase3.1-gpu-validation.md`, `docs/phase3.1-3.2-report.md` | RTX 4090 backbone/head component checks and memory observations; A100 characterization remained pending. | Historical measurements only. Old dependency/config descriptions, training feasibility estimates and Phase 3.3 references are superseded by current plans. |
| Former random dataset/Transformer/trainer/evaluator and cluster jobs | Infrastructure exercised synthetic tensors and toy checkpoints. | Retired because it did not train or evaluate the maintained VLA on a real task. Its multi-GPU results do not qualify a future VLA trainer. |

Robot extraction sources and transformations remain fully tracked in [asset provenance](../../sim/assets/robots/alex/PROVENANCE.md). They have a continuing scientific and maintenance purpose.

## Recovering original plans

```bash
git show a5205b297ab27f68b788aea0c2084bc259c7cd33:docs/roadmap/phase1.2.md
git show a5205b297ab27f68b788aea0c2084bc259c7cd33:docs/roadmap/phase2.1.md
git show 31bffb75cf1b6456e307d6daadf0f1892fbea338:docs/roadmap/phase0.2.md
git show efb89ab5a44c7969a66d6d47463b9cd3897c47cd:docs/roadmap/phase1.1.md
```

The other detailed plans `phase2.2`, `phase2.3`, `phase3.1`, `phase3.2`, `phase4.1` and `phase4.2` also exist at the baseline under `docs/roadmap/`. The [current implementation index](implementation_phases/index.md) reconciles every original website phase with retained code and remaining work. Draft folder layouts, old setup commands, ad hoc tuning procedures and obsolete training recipes are intentionally not copied into a second archive.
