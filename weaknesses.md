# Phase 0 Audit — Issues in `worldsim_project.zip` (as-is, before any change)

This document lists what is **not working** in the current zip **before any modifications**, mapped to the Phase 0 checklist requirements, with the **reason** each issue matters.

---

## A) Repo structure hygiene

### 1) Stray directories created by brace-expansion
**What exists in the tree:**
- `project/{sim,data,models,train,eval,configs,scripts,tests,docs,.github/`
- `project/configs/{model,trainer,data,cluster,logging}/`

**Why it matters:**
- Adds noise, confuses navigation and tooling, and is a clear repo-hygiene failure for Phase 0.

---

## B) Config-first execution (currently violated)

### 2) Training ignores `data` config and hardcodes datasets
**What happens:**
- `train/trainer.py` always constructs `DummyDataset(...)` and does not respect `cfg.data.dataset.name/path`.

**Why it matters:**
- Violates “config-first execution” (no hardcoded paths/hparams). The runtime does not follow versioned configs.

### 3) Dummy sample sizes are hardcoded in code
**What happens:**
- Code uses fixed sizes such as `num_samples=10000` (train) and `num_samples=500` (val).

**Why it matters:**
- Prevents consistent tiny runs and scalable experiments via config; undermines reproducibility and parameterization.

### 4) Defaults are not safe for local development
**What happens:**
- `configs/config.yaml` defaults include `cluster: gilbreth` and `data: default`.
- `data: default` points to a simulation dataset path that likely isn’t present locally.
- `cluster: gilbreth` enables DeepSpeed and assumes a cluster/GPU context.

**Why it matters:**
- The milestone requires “main stays runnable” and “one-cmd dry-run entrypoint.” Default config likely fails on local machines.

---

## C) Paths portability (local vs Gilbreth)

### 5) `paths.root` default is hardcoded to `/home/$USER/worldsim`
**What happens:**
- In `configs/config.yaml`, root defaults to `/home/${USER}/worldsim`.

**Why it matters:**
- Not portable across machines; not aligned with cluster best practices (scratch-first for logs/checkpoints).

### 6) Gilbreth config hardcodes `worldsim` in scratch paths
**What happens:**
- `configs/cluster/gilbreth.yaml` uses paths like `.../worldsim/checkpoints`.

**Why it matters:**
- Creates inconsistency once the project/repo is renamed; complicates running multiple projects or side-by-side experiments.

---

## D) Tests are currently broken (CI will fail)

### 7) Hydra config paths in tests are wrong
**What happens:**
- Several tests call Hydra with `initialize(config_path="../configs", ...)`.

**Why it matters:**
- When running pytest from the repo root (standard), `../configs` points outside the repo, so tests fail immediately.

### 8) `scripts/validate_configs.py` uses the same wrong config path
**What happens:**
- Also uses `initialize(config_path="../configs", ...)`.

**Why it matters:**
- The CI “Validate Configs” job will fail; prevents reliable gating and violates Phase 0 CI expectations.

### 9) `tests/test_data.py` expects shapes that don’t match `DummyDataset`
**What happens:**
- `DummyDataset` defaults are typically `seq_length=512`, `state_dim=256`.
- The test asserts shapes around `(128, 64)`.

**Why it matters:**
- Even after fixing Hydra config paths, unit tests will still fail due to incorrect expectations.

---

## E) CI workflow does not match policy (and is not “fast CPU-only”)

### 10) CI does not run on every push
**What happens:**
- `.github/workflows/ci.yml` triggers only on `main` and `develop`.

**Why it matters:**
- Phase 0 requires checks on **each push/PR**, including feature branches.

### 11) CI installs heavyweight deps during “fast checks”
**What happens:**
- CI does `pip install -e ".[dev]"`, while base deps include heavy/problematic packages (e.g., `deepspeed`, plus `wandb`, `accelerate`).

**Why it matters:**
- Breaks “fast CPU-only checks,” slows CI, and increases failure risk (especially cross-platform).

### 12) Smoke test is not a real dry-run
**What happens:**
- Smoke test runs `python -m train.trainer --help`, which does not execute the pipeline.

**Why it matters:**
- Does not satisfy milestone “one-cmd dry-run entrypoint” that exercises imports + config parsing + minimal execution.

### 13) CI “success” gate ignores some jobs
**What happens:**
- `ci-success` only checks lint/test/smoke outcomes, ignoring `typecheck` and `config-check`.

**Why it matters:**
- “Green” could still allow config regressions—exactly what Phase 0 is meant to prevent.

---

## F) CD/container reproducibility gaps (minor now, but will matter with your repo name)

### 14) GHCR image naming will break with `VLA-LEGO_Project` (uppercase)
**What happens:**
- `cd.yml` uses `${{ github.repository }}` for image naming.
- Repo name `VLA-LEGO_Project` includes uppercase, which is invalid for OCI image names.

**Why it matters:**
- CD will fail when pushing images to GHCR unless the image name is lowercased.

### 15) No explicit container hash export artifact
**What happens:**
- CD builds/pushes images and SBOM, but does not explicitly persist the image digest/hash in an artifact/release note.

**Why it matters:**
- Checklist includes “build container/hash export.” Persisting digests makes reproducibility auditable.

### 16) Container/Apptainer naming still uses `worldsim`
**What happens:**
- `cd.yml` builds `worldsim.sif`; docs/examples mention `worldsim`.

**Why it matters:**
- Not strictly blocking, but confusing/inconsistent once the project name is `VLA-LEGO_Project`.

---

## Priority summary

### Blocking issues (must fix to satisfy Phase 0)
- Broken Hydra config paths in tests and `scripts/validate_configs.py`
- Broken dataset shape expectations in `tests/test_data.py`
- Training pipeline violates config-first (hardcoded dummy logic + sizes)
- CI does not run on every push/PR (feature branches)
- Smoke test is not a real dry-run
- CI success gate does not require all critical jobs (including config-check)

### Cleanup / consistency (strongly recommended)
- Remove stray brace-expansion directories
- Replace hardcoded `/home/$USER/worldsim` defaults with portable workspace policy
- Remove hardcoded `worldsim` strings from Gilbreth paths/docs
- Lowercase GHCR image naming (required once repo name is `VLA-LEGO_Project`)
- Add explicit digest/hash export artifact
