# Phase 0.2 — MuJoCo Sim Stack + Visualization Workflow (Lab-first, Cluster-ready) (2–3 days)
**Goal:** Enable fast, visual, and reproducible MuJoCo development on the **lab workstation (Linux + monitor)**, while keeping the codebase **cluster-ready** for future large-scale rollouts and training.

**What is already done in Phase 0.1 (do NOT repeat):**
- Cluster access + SLURM job templates
- CUDA/cuDNN/PyTorch pinned stack + containerization strategy
- DeepSpeed ZeRO-1 multi-GPU training smoke tests
- W&B/MLflow tracking conventions
- Repo structure + CI/CD + config-first execution

**Phase 0.2 focus (new):**
- MuJoCo installation + rendering backends
- Interactive viewer workflow (lab PC)
- Headless rendering workflow (cluster-compatible)
- Simulation smoke tests + visual artifacts (videos/frames) integrated into the Phase 0.1 reproducibility contract

---

## 0.2.1 — MuJoCo Runtime Installation (Lab PC baseline)
### What we will do
Install MuJoCo (Python bindings) and validate basic simulation stepping on the **lab PC** in a pinned, reproducible environment.

### Why this matters
All Phase 1 work depends on a stable simulator. If MuJoCo import/rendering is flaky, you will lose days later debugging “not the robot, not the environment—just the stack”.

### Execution checklist
- Select the **canonical** MuJoCo Python binding and pin versions.
- Ensure system-level prerequisites are present on the lab PC (OpenGL libs, etc.).
- Confirm `import` works and a minimal MJCF loads.
- Validate deterministic stepping at a fixed timestep for N steps.
- Record environment metadata:
  - OS + GPU driver version
  - Python version
  - pinned deps hash / lockfile hash
  - git commit hash

### Milestone (minimum satisfiable)
- A minimal MuJoCo scene loads and steps reliably on the lab PC with no runtime errors.
- Setup is reproducible from docs in < 60 minutes.

---

## 0.2.2 — Interactive Viewer Workflow (Lab PC “Debug Mode”)
### What we will do
Enable an interactive viewer workflow on the lab PC to debug:
- robot asset loading (Alex MJCF/meshes),
- coordinate frames,
- contacts,
- camera placement.

### Why this matters
For contact-rich manipulation, visual debugging is the fastest path to correctness (frames, penetration, collisions, camera alignment). This is your “speed loop”.

### Execution checklist
- Validate windowed rendering works (viewer opens reliably).
- Create a standard “viewer launch” entrypoint (consistent args + config).
- Define a debug checklist you can run in minutes:
  - verify gravity and ground plane
  - verify joint axes directions visually
  - verify collision geoms vs visual meshes
  - show contact points / forces if your tooling supports it
- Confirm that viewer usage does not pollute training/runtime codepaths (separation of debug vs headless).

### Milestone (minimum satisfiable)
- You can open a MuJoCo scene and visually inspect motion/contacts for 2–5 minutes without crashes.

---

## 0.2.3 — Headless Offscreen Rendering (Lab PC “Batch Mode”)
### What we will do
Make sure offscreen rendering works without a GUI so you can:
- generate videos for debugging,
- later run the same pipeline headlessly on the cluster.

### Why this matters
Phase 1 requires multi-view logging. Offscreen rendering is non-negotiable for scale and for unattended regression tests.

### Execution checklist
- [x] Select and validate the headless rendering backend strategy (GPU-offscreen preferred).
- [x] Run an offscreen render smoke test that produces:
  - [x] RGB frames (required)
  - [x] depth frames (strongly recommended; needed soon)
  - [x] segmentation/instance IDs (recommended for debugging/evaluation)
- [x] Export a short video artifact (e.g., 5–10 seconds) + a few sample images.
- [x] Verify strict synchronization primitives exist conceptually:
  - [x] same sim step -> same rendered frame index (alignment contract)
- [x] Store artifacts with Phase 0.1 run naming conventions.

### Milestone (minimum satisfiable)
- [x] A headless run produces a short MP4 (or equivalent) + sample frames reliably on the lab PC.

### Implementation notes
- Core module: `sim/offscreen.py` — `RenderConfig`, `RenderedFrame`, `render_frame`, `render_trajectory`, `save_video`, `save_sample_frames`
- Validation: `python scripts/validate_offscreen.py` (6-check script)
- Tests: `pytest tests/test_offscreen.py -v` (24 tests)
- Dependency: `imageio[ffmpeg]>=2.31.0` added to `sim` and `dev` groups
- Named camera `overhead` added to `test_scene.xml` for reliable offscreen rendering
- Key detail: `mj_forward()` must be called before rendering to compute lighting

---

## 0.2.4 — “Sim Smoke Tests” Suite (Push-button, CI-friendly)
### What we will do
Add a minimal smoke test suite for simulation that is fast and repeatable.

### Why this matters
This protects you from subtle regressions when you change assets, contacts, or rendering. It also signals “startup-grade engineering discipline”.

### Execution checklist
- [x] Create 3 standardized tests:
  1) **Step test:** N steps with stable dt, no NaNs, no explosions
  2) **Render test:** produce RGB (+ depth if enabled) deterministically
  3) **I/O test:** writes logs/artifacts to the correct directory layout
- [x] Define pass/fail metrics:
  - max penetration threshold (5 cm)
  - energy bound (1000 J)
  - render returns correct shapes and non-empty frames
- [x] Integrate with Phase 0.1 conventions:
  - log seed + config snapshot
  - attach artifacts to W&B (opt-in via `WANDB_MODE=online`)

### Milestone (minimum satisfiable)
- [x] One command runs all smoke tests and generates artifacts + logs with consistent naming.

### Implementation notes
- Pytest suite: `pytest -m smoke -v` or `pytest tests/test_sim_smoke.py -v` (12 tests)
- Standalone script: `python scripts/validate_sim_smoke.py` (3 categories, artifacts to `logs/sim_smoke/`)
- Pass/fail thresholds: max penetration 5 cm, energy bound 1000 J, no NaN, finite qpos
- `smoke` marker registered in `tests/conftest.py`
- W&B: set `WANDB_MODE=online` to attach video/frames/metadata as artifacts
- Artifacts: `logs/sim_smoke/smoke_video.mp4`, `logs/sim_smoke/frames/*.png`, `logs/sim_smoke/sim_smoke_meta.json`

---

## 0.2.5 — Asset Pathing & Loader Contract (Pre-Phase 1.1 enabler)
### What we will do
Lock a strict contract for how assets (MJCF + meshes + textures) are stored and referenced, so loading the Alex model is predictable.

### Why this matters
The most common failure mode in robotics sim work is “broken mesh paths / wrong scaling / inconsistent frames”. A loader contract prevents repeated friction.

### Execution checklist
- Define a canonical asset directory layout (and keep it stable).
- Enforce relative paths inside MJCF (no machine-specific absolutes).
- Add an “asset linter” that checks:
  - missing files
  - invalid references
  - suspicious scaling factors
- Create a minimal “asset load test” that loads:
  - floor + light + one test body
  - then the Alex MJCF (in Phase 1.1)

### Milestone (minimum satisfiable)
- Assets are loadable via a single entrypoint with zero missing-path errors.

### Implementation notes
- Canonical layout: `sim/assets/scenes/` (MJCF scenes), `sim/assets/robots/<name>/` (robot models with meshes/textures), `sim/assets/shared/` (future shared materials)
- Core module: `sim/asset_loader.py` — `load_scene("test_scene")` as single entrypoint, `resolve_scene_path()`, `resolve_robot_path()`
- Linter module: `sim/asset_linter.py` — `lint_mjcf()` checks absolute paths, missing files, suspicious scales
- CLI: `vla-lint-assets` console script (`sim/asset_linter_cli.py`)
- Validation: `python scripts/validate_assets.py` (4-check script, artifacts to `logs/`)
- Tests: `pytest tests/test_asset_loader.py -v` (directory layout, loader, linter)
- `test_scene.xml` moved from `sim/assets/` to `sim/assets/scenes/`

---

## 0.2.6 — Cluster Compatibility Smoke (SSH/VS Code + optional ThinLinc)
### What we will do
Confirm the cluster can run the exact same headless simulation path for future scaling, without making the cluster the primary debug environment.

### Why this matters
You want lab-first iteration speed, but you also need confidence that scaling later won’t break due to rendering backend or missing system deps.

### Execution checklist
- Reuse Phase 0.1 job templates / container strategy (no new infra).
- Run one short SLURM job that:
  - steps sim headlessly
  - renders a short clip
  - writes artifacts to the cluster filesystem using your standard layout
- Decide ThinLinc usage policy:
  - **Default:** no GUI on cluster (headless artifacts)
  - **Fallback:** ThinLinc only if a visual bug cannot be diagnosed from saved videos/frames

### Milestone (minimum satisfiable)
- A cluster job produces the same type of artifacts as the lab PC headless test (video/frames + logs).

---

# Phase 0.2 — Completion Criteria (Definition of Done)
Phase 0.2 is complete when:
- Lab PC supports **interactive viewer** for fast debugging.
- Lab PC supports **headless offscreen rendering** and exports artifacts reliably.
- A **push-button sim smoke test suite** exists and aligns with Phase 0.1 reproducibility standards.
- Asset paths and loader contract are defined (ready to load IHMC Alex in Phase 1.1).
- Cluster can run a headless sim smoke job (render + save) using existing Phase 0.1 infrastructure.
