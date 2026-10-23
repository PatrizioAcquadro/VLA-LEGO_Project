# Decision log

Keep only decisions that change interpretation or future development. Use Git for ordinary edits and commit history.

## 2026-10-23 — Maintained scope and task change

- Start with lightweight variable-size bricks and gravity-based pick/place/stack tasks. Defer real LEGO studs, press-fit and retention to a separate extension.
- Retire synthetic training/evaluation and their exclusive consumers. Build the next trainer around real robot demonstrations and the maintained Qwen/VLA components.
- Consolidate presentation diagnostics into `vla-demo`; label prearranged layouts and untrained model outputs explicitly.
- Make the wiki and all 22 reconciled phase plans canonical. Preserve important failed/skipped historical outcomes and asset provenance; recover ordinary old drafts through Git.
- Keep assistant guidance local and ignored. Work remains on the requested feature branch with local commits; no remote integration or publication is part of this audit.

## 2026-10-23 — Source layout and feature-branch publication

- Consolidate runtime code, Hydra configuration and assets under `src/vla_lego`; keep CLI names and simulation/model contracts unchanged.
- Group tests by subsystem and keep repository utilities in `scripts/`. Use installed-package imports and verify the wheel inventory against `src/`.
- Run CI on feature-branch pushes. Publishing remains separately gated. The user authorized pushing the requested feature branch; main integration is still outside scope.

## 2026-10-23 — Essential root and main-only development

- After the user’s merge, keep main as the only branch. Preserve all removed branch histories in a local verified bundle, including the unmerged README badge commit from the old CI test branch.
- Keep six conventional root files; move container definitions into `infra/containers` with a Dockerfile-specific context allowlist. Remove duplicate root guides, environment examples and an unused Python patch-version pin.
- Keep agent TODO checklists local and ignored, with durable development plans in the nine canonical phase pages.
