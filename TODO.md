# TODO — Source layout and branch publication

Implementation phase spec: `knowledge/repository-cleanup.md#source-layout-follow-up`.

Baseline: branch `PatrizioAcquadro/audiovisual-policy` @ `dae723a147356193d0181d450927284e370c9d00`.
Pre-existing user changes to preserve:
- None; the tracked working tree was clean. Ignored local instructions, environment and evidence remain local.

Requested publication: push this feature branch after validation and verify the remote revision.

## Tasks

- [x] 0. Move runtime code, configuration and assets into one `src/vla_lego` package.
- [x] 1. Organize tests by subsystem and consolidate repository utilities under scripts.
- [x] 2. Update package installation, containers, CI and canonical documentation.
- [x] 3. Validate imports, installed wheel, native GPU/model/simulation behavior and software checks.
- [ ] 4. Create validated local commits for publication.
