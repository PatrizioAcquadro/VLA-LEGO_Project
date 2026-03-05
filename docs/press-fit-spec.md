# Soft Real Press-Fit Specification (Phase 1.2.0)

## 1. Purpose and Scope

This document defines the **Soft Real Press-Fit** connection model for LEGO brick assembly in the VLA-LEGO MuJoCo simulation. It specifies geometry, contact physics, tolerances, and measurable success criteria that all subsequent Phase 1.2 subphases implement against.

**What this spec defines:**
- Exact simulation dimensions for studs, tubes, and brick shells
- MuJoCo contact parameter classes for LEGO physics
- Alignment tolerances and capture envelope
- Retention mechanism design
- Quantitative acceptance tests

**What this spec does NOT define (deferred to later subphases):**
- Procedural generation code (1.2.1)
- Contact parameter tuning methodology (1.2.2)
- Baseplate layout and workspace (1.2.3)
- Camera setup (1.2.4)
- Episode management (1.2.5)
- Assembly task logic (1.2.6)

**Core principle:** Studs and tubes have accurate geometry/topology, but physics is made robust via carefully designed compliance, tolerances, and contact conditioning — not hard interference or magic snap constraints.

---

## 2. Real LEGO Reference Dimensions

Standard LEGO brick dimensions (from published specifications and measurements):

| Dimension | Value (mm) | Source |
|-----------|-----------|--------|
| Stud pitch (center-to-center) | 8.0 | Standard |
| Stud outer diameter | 4.8 | Standard (measured: ~4.85-4.89) |
| Stud height (above brick top) | 1.7 | Standard (some sources: 1.6-1.8) |
| Tube outer diameter | 6.51 | Standard (for 2+ wide bricks) |
| Tube inner diameter | 4.8 | Nominal (creates interference with stud) |
| Standard brick height (body only) | 9.6 | = 3 plates |
| Standard brick height (with stud) | 11.3 | 9.6 + 1.7 |
| Plate height (body only) | 3.2 | = 1 plate |
| Wall thickness | 1.2 | Standard |
| Inter-brick gap (side-to-side) | 0.2 | 0.1 per side |
| Brick width (N studs) | N × 8.0 - 0.2 | Accounts for gap tolerance |
| Manufacturing tolerance | ±0.02 | Typical injection molding |
| Interference fit (stud-tube) | ~0.1-0.2 | Stud OD slightly > tube ID |

**Real insertion/retention forces (published estimates):**

| Metric | Value | Notes |
|--------|-------|-------|
| Insertion force per stud | 3-5 N | Varies with brick age/condition |
| Pull-off force per stud | 2-3 N | Vertical separation |
| Lateral shear per stud | 1-2 N | Before sliding |

**Material:** ABS (Acrylonitrile Butadiene Styrene)
- Density: ~1050 kg/m³
- Young's modulus: ~2.3 GPa
- Coefficient of friction (ABS-on-ABS): 0.3-0.5

---

## 3. Simulation Geometry Specification

### 3.1 Coordinate and Unit Conventions

- **Units:** meters (MuJoCo convention). All dimensions below given in both mm and m.
- **Coordinate frame:** Z-up, X-forward, Y-left (consistent with Phase 1.1 world frame)
- **Stud axis:** +Z (studs point up)
- **Brick origin:** center of bottom face (geometric center at z=0 of the shell bottom)
- **Stud grid origin:** first stud at `(-pitch*(nx-1)/2, -pitch*(ny-1)/2)` relative to brick origin, where `nx`, `ny` are stud counts

### 3.2 Stud Geometry (Collision)

Each stud is represented as a **MuJoCo cylinder primitive** (type="cylinder"):

| Parameter | Value (mm) | Value (m) | Rationale |
|-----------|-----------|-----------|-----------|
| Radius | 2.35 | 0.00235 | 50 µm undersize vs real 2.4 mm; provides clearance for tube capsule ring |
| Half-height | 0.85 | 0.00085 | Total height 1.7 mm (full stud height) |
| Z-offset (from brick top) | +0.85 | +0.00085 | Center of stud cylinder above brick top surface |

**Visual mesh:** Full stud cylinder at real dimensions (r=2.4 mm, h=1.7 mm) in a separate visual-only geom (group 0 or 1, contype=0, conaffinity=0).

**Contact class:** `lego/stud_tube` (see Section 4.1)

### 3.3 Tube/Socket Geometry (Collision)

MuJoCo has no hollow cylinder primitive. We represent each tube as a **ring of 8 capsule primitives** arranged in a circle:

```
    Top view of one tube (8 capsules):

         ○
       ○   ○        ○ = capsule cross-section (r=0.5 mm)
      ○  ●  ○       ● = tube center (stud insertion axis)
       ○   ○        Ring radius = 2.40 mm (capsule centers)
         ○
```

| Parameter | Value (mm) | Value (m) | Rationale |
|-----------|-----------|-----------|-----------|
| Ring radius (capsule centers) | 2.40 | 0.00240 | Matches real tube ID = 4.8 mm diameter |
| Capsule radius | 0.50 | 0.00050 | Creates inner opening of ~3.8 mm effective ID at narrowest |
| Capsule half-height | 0.80 | 0.00080 | Total capsule length 1.6 mm; slightly shorter than stud to allow engagement |
| Capsule count per tube | 8 | — | Provides near-rotational symmetry; tunable up to 12 |
| Angular spacing | 45° | — | 360° / 8 capsules |
| Z-offset (from brick bottom) | -0.80 | -0.00080 | Capsule centers inside brick shell bottom |

**Effective radial clearance:** stud radius (2.35 mm) vs tube effective opening. The gap between adjacent capsule edges (where the stud can pass) is wider than the capsule-to-center distance, creating a **funnel effect** that guides aligned studs while rejecting misaligned ones.

**Geometric interlock mechanism:** When a stud is inserted, it sits between the capsule ring. The capsules provide:
1. **Lateral constraint** — stud cannot slide more than ~0.5 mm before contacting a capsule
2. **Retention lip** — the capsule ring's top edge creates a slight overhang that resists pull-out via friction and geometry
3. **Rotational constraint** — 8-point contact ring resists torque (reinforced by condim=4)

**Contact class:** `lego/stud_tube` (see Section 4.1)

**Fallback strategy:** If the 8-capsule ring produces unacceptable jitter or uneven friction during Phase 1.2.2 tuning, increase to 12 capsules. If still insufficient, fall back to a **friction-only** model where tubes are represented by a single flat contact surface with high friction.

**Why not other approaches:**
- **Mesh collision:** thin-walled cylinder mesh creates degenerate triangles → solver instability
- **Box walls (4 boxes forming a square socket):** not rotationally symmetric → unrealistic insertion behavior
- **Single cylinder with contact conditioning only:** no geometric interlock → retention depends entirely on friction

### 3.4 Brick Shell Geometry

The brick body (hollow rectangular box) uses **box primitives** for collision:

| Parameter | Formula (mm) | Example: 2×4 (mm) | Example: 2×4 (m) |
|-----------|-------------|-------------------|-------------------|
| Shell half-X | (nx × 8.0 - 0.2) / 2 | 15.9 | 0.0159 |
| Shell half-Y | (ny × 8.0 - 0.2) / 2 | 7.9 | 0.0079 |
| Shell half-Z | brick_height / 2 | 4.8 | 0.0048 |
| Wall thickness | 1.2 | 1.2 | 0.0012 |

Where `nx`, `ny` are stud counts along X, Y; `brick_height` = 9.6 mm for standard bricks.

**Mass computation:** `mass = density × volume_solid` where `volume_solid` accounts for hollow interior:
- `V_outer = (nx × 8.0 - 0.2) × (ny × 8.0 - 0.2) × 9.6` mm³
- `V_inner = (V_outer_x - 2×1.2) × (V_outer_y - 2×1.2) × (9.6 - 1.0)` mm³ (1.0 mm top thickness)
- `V_studs = nx × ny × π × 2.4² × 1.7` mm³
- `V_tubes ≈ n_tubes × π × (3.255² - 2.4²) × 1.6` mm³ (approximate)
- `mass = 1.05 × 10⁻⁶ × (V_outer - V_inner + V_studs + V_tubes)` kg (density in kg/mm³)

Example masses (approximate):
| Brick | Mass (g) | Mass (kg) |
|-------|----------|-----------|
| 2×2 | ~2.3 | 0.0023 |
| 2×4 | ~4.5 | 0.0045 |
| 2×6 | ~6.8 | 0.0068 |

**Contact class:** `lego/brick_surface` for external faces (see Section 4.2)

**Inertia:** Computed via MuJoCo's automatic inertia from geom primitives, or explicitly set from box shell approximation. Preferably let MuJoCo compute via `<compiler inertiafromgeom="true"/>`.

### 3.5 Brick Type Parameters

| Brick Type | nx | ny | Height (mm) | Studs | Tubes |
|-----------|----|----|-------------|-------|-------|
| 2×2 | 2 | 2 | 9.6 | 4 | 1 (center) |
| 2×4 | 2 | 4 | 9.6 | 8 | 3 (row) |
| 2×6 | 2 | 6 | 9.6 | 12 | 5 (row) |

**Tube placement:** For 2-wide bricks, tubes are placed along the centerline (Y=0 relative to brick center), spaced at 8.0 mm pitch. A 2×N brick has (N-1) tubes.

### 3.6 Dimensional Summary

| Dimension | Real (mm) | Sim (mm) | Sim (m) | Δ from real | Rationale |
|-----------|-----------|----------|---------|-------------|-----------|
| Stud pitch | 8.0 | 8.0 | 0.008 | 0 | Exact |
| Stud radius (collision) | 2.4 | 2.35 | 0.00235 | -0.05 | Clearance for capsule ring |
| Stud radius (visual) | 2.4 | 2.4 | 0.0024 | 0 | Exact visual |
| Stud height | 1.7 | 1.7 | 0.0017 | 0 | Exact |
| Tube ring radius | ~2.4 | 2.40 | 0.0024 | 0 | Capsule centers match real tube ID |
| Tube capsule radius | N/A | 0.50 | 0.0005 | N/A | Sim-only parameter |
| Brick height (body) | 9.6 | 9.6 | 0.0096 | 0 | Exact |
| Wall thickness | 1.2 | 1.2 | 0.0012 | 0 | Visual only (shell is box) |
| Effective radial clearance | -0.1 (interference) | +0.05 | +0.00005 | +0.15 | Key sim adaptation |

---

## 4. Contact Physics Specification

### 4.1 Material Class: `lego/stud_tube`

Used for stud-to-tube capsule contacts. This is the critical class for press-fit behavior.

```xml
<default class="lego/stud_tube">
    <geom contype="2" conaffinity="2"
          solref="0.004 1.0"
          solimp="0.95 0.99 0.001 0.5 3"
          friction="0.5 0.005 0.005"
          condim="4"/>
</default>
```

**Parameter rationale:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `solref` | `0.004 1.0` | timeconst=4 ms (stiffer than robot 5 ms, modeling hard ABS); dampratio=1.0 (critically damped, no bounce). Satisfies constraint: timeconst ≥ 2×timestep = 4 ms. |
| `solimp` | `0.95 0.99 0.001 0.5 3` | dmin=0.95 (near-rigid even at light contact); dmax=0.99 (very rigid at full engagement); width=0.001 m = 1 mm (transition zone matches clearance scale); midpoint=0.5; power=3 (sharper impedance ramp → more distinct "snap" feel during insertion) |
| `friction` | `0.5 0.005 0.005` | Tangent=0.5 (upper range of ABS-on-ABS; provides retention); torsional=0.005 (prevents spin after insertion); rolling=0.005 (prevents roll-out) |
| `condim` | `4` | 3D sliding + torsional friction. Torsional friction is critical: prevents brick from rotating on studs after insertion. |
| `contype` | `2` | Bit 1 set: LEGO-internal contact channel |
| `conaffinity` | `2` | Bit 1 set: responds to LEGO contype=2 |

### 4.2 Material Class: `lego/brick_surface`

Used for external brick faces (top surface around studs, side walls, bottom surface around tubes). Governs brick-to-brick surface sliding and brick-to-table contact.

```xml
<default class="lego/brick_surface">
    <geom contype="2" conaffinity="3"
          solref="0.005 1.0"
          solimp="0.9 0.95 0.001 0.5 2"
          friction="0.4 0.005 0.002"
          condim="3"/>
</default>
```

**Parameter rationale:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `solref` | `0.005 1.0` | Same as robot collision defaults; brick surfaces don't need extra stiffness |
| `solimp` | `0.9 0.95 0.001 0.5 2` | Matches robot defaults; power=2 (gradual, no snap needed on flat surfaces) |
| `friction` | `0.4 0.005 0.002` | Lower than stud-tube (flat ABS sliding is smoother); enough to prevent ice-skating |
| `condim` | `3` | 3D sliding only; no torsional friction needed for flat contacts |
| `contype` | `2` | LEGO channel |
| `conaffinity` | `3` | Bits 0+1: responds to both robot (contype=1) and LEGO (contype=2) contacts |

### 4.3 Material Class: `lego/baseplate`

Used for baseplate surface and baseplate studs. Higher friction for stable anchoring.

```xml
<default class="lego/baseplate">
    <geom contype="2" conaffinity="3"
          solref="0.005 1.0"
          solimp="0.9 0.95 0.001 0.5 2"
          friction="0.6 0.01 0.005"
          condim="3"/>
</default>
```

**Parameter rationale:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `friction` | `0.6 0.01 0.005` | Higher than brick surface; baseplate is static → more friction helps anchor bricks |
| Others | Same as brick_surface | Baseplate doesn't need special stiffness/impedance |
| `conaffinity` | `3` | Responds to robot (table surface contact) and LEGO |

### 4.4 Contact Group Isolation Scheme

```
Robot geoms:     contype=1, conaffinity=1
LEGO stud/tube:  contype=2, conaffinity=2  (LEGO-to-LEGO only)
LEGO surface:    contype=2, conaffinity=3  (LEGO-to-LEGO + robot-to-LEGO)
Baseplate:       contype=2, conaffinity=3  (LEGO-to-LEGO + robot-to-LEGO)
Floor/table:     contype=1, conaffinity=1  (robot + LEGO surfaces contact floor)
```

**Why isolate stud/tube contacts?** Robot collision geoms should never directly contact stud/tube capsule internals (only the robot grips the brick shell). This prevents the robot's collision geometry from interfering with the delicate stud-tube press-fit contacts.

**Important:** Robot geom conaffinity must also include bit 1 to contact LEGO brick_surface geoms. Update robot defaults to `conaffinity="3"` (bits 0+1) so the EZGripper can grasp bricks.

### 4.5 Solver Requirements

| Parameter | Current (Phase 1.1) | Recommended (Phase 1.2) | Rationale |
|-----------|---------------------|------------------------|-----------|
| `iterations` | 50 | 80 | Multi-stud contacts increase constraint count; more iterations needed for convergence |
| `ls_iterations` | 10 | 10 | Adequate |
| `timestep` | 0.002 s | 0.002 s | Compatible with solref timeconst=0.004 s (≥ 2× timestep) |
| `integrator` | `implicitfast` | `implicitfast` | Unconditionally stable; best for contact-rich tasks |
| `noslip_iterations` | 0 (default) | 0 | May increase to 1-2 if sliding artifacts appear during 1.2.2 tuning |
| `cone` | `pyramidal` (default) | `pyramidal` | Elliptic cone is more accurate but slower; try pyramidal first |

**Contact count budget:** A 2×2 brick insertion involves up to 4 studs × 8 capsules = 32 stud-tube contact pairs. A 2×6 insertion: 12 studs × 8 = 96 pairs. Target: simulation remains real-time with <100 active contacts on a single CPU core.

---

## 5. Capture Envelope and Alignment Tolerances

### 5.1 Capture Envelope

The capture envelope defines the region where a brick can be successfully inserted:

| Tolerance | Value | Rationale |
|-----------|-------|-----------|
| Lateral (XY) | ±0.5 mm (±0.0005 m) | ~6× real LEGO tolerance; accounts for sim compliance and gripper precision |
| Angular (roll/pitch) | ±3° from vertical | ~2-3× real tolerance; allows for slight EZGripper misalignment |
| Vertical approach zone | starts 2.0 mm (0.002 m) above engagement | stud height is 1.7 mm; extra 0.3 mm margin |

### 5.2 Alignment Success/Failure Behavior

**Within envelope (aligned):**
- Studs find tube openings via compliance
- Gradual insertion with observable resistance
- Stud slides between capsule ring elements
- Final state: stud seated within capsule ring, held by friction + geometry

**Outside envelope (misaligned):**
- Stud contacts tube capsule tops or adjacent capsules
- Resistance increases rapidly → brick slides off or jams
- No "snap correction" or "teleportation" to aligned position
- Brick remains in contact but does not engage (resting on top surface)

### 5.3 Approach Velocity

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Nominal insertion speed | 1-5 mm/s (0.001-0.005 m/s) | Slow enough for stable contact; fast enough for practical episodes |
| Maximum insertion speed | 10 mm/s (0.01 m/s) | Beyond this, impact forces may cause bouncing or solver stress |

---

## 6. Retention Mechanism

### 6.1 Primary Retention: Friction

The tangent friction coefficient (0.5) on stud-tube contacts creates resistance to separation. With 8 contact points per stud (one per capsule), the normal forces from the capsule ring pressing inward on the stud generate friction that resists vertical pull-out.

**Expected retention force per stud:** 0.3-1.0 N (depends on capsule compression and friction coefficient)

### 6.2 Secondary Retention: Geometric Interlock

The capsule ring provides a slight geometric interlock:
- Capsule tops form a lip around the stud entry
- Once the stud passes the lip (widest point), it seats below the capsule tops
- Pull-out requires the stud to push capsules outward again (compliance resists this)

This is a **soft interlock** — not a rigid snap — consistent with the "soft real" philosophy.

### 6.3 Combined Retention

| Mechanism | Contribution | Notes |
|-----------|-------------|-------|
| Friction | Primary (~60-70%) | Tangent friction at capsule-stud contact |
| Geometric interlock | Secondary (~30-40%) | Capsule ring lip effect |
| Torsional friction | Anti-rotation | condim=4 prevents twist-off |
| Gravity | Supplementary | Brick weight helps maintain contact |

### 6.4 What We Explicitly Do NOT Model

- **Plastic deformation** (ABS flex during real insertion) — modeled as elastic compliance instead
- **Acoustic/tactile feedback** (click sound/feel) — not relevant for VLA training
- **Thermal effects** (friction heating) — negligible
- **Fatigue** (connection loosening over cycles) — all cycles are identical in sim
- **Asymmetric wear** (real LEGO bricks wear unevenly) — all bricks are pristine

---

## 7. Measurable Success Criteria

All tests below are scripted and deterministic. They constitute the acceptance criteria for Phase 1.2.2 (contact tuning) and serve as regression tests thereafter.

### 7.1 Insertion Tests

| Test | Setup | Pass Criterion |
|------|-------|---------------|
| **Single stud aligned** | Stud centered over tube, approach at 1 mm/s downward | Full engagement (stud z-center below tube lip) within 0.5 s |
| **Single stud near-miss (in envelope)** | 0.3 mm lateral offset | Still succeeds; engagement within 1.0 s |
| **Single stud miss (outside envelope)** | 1.0 mm lateral offset | Fails to engage; stud rests on surface or slides off |
| **2×2 aligned** | All 4 studs aligned, approach at 1 mm/s | All 4 studs engaged within 1.0 s |
| **2×4 aligned** | All 8 studs aligned, approach at 1 mm/s | All 8 studs engaged within 2.0 s |
| **Noisy insertion** | Gaussian noise σ=0.2 mm on XY position, 100 trials | Success rate ≥ 95% |
| **Angular tolerance** | 2° tilt from vertical, aligned XY | Succeeds (within angular envelope) |
| **Angular rejection** | 5° tilt from vertical, aligned XY | Fails or partial engagement only |

### 7.2 Retention Tests

| Test | Setup | Pass Criterion |
|------|-------|---------------|
| **Vertical pull-off (single stud)** | After successful insertion, apply upward force ramp | Resists ≥ 0.3 N before detaching |
| **Vertical pull-off (2×2)** | After successful 4-stud insertion | Resists ≥ 1.2 N (4 × 0.3 N) before detaching |
| **Lateral shear (single stud)** | After insertion, apply horizontal force ramp | Resists ≥ 0.2 N before sliding |
| **Static hold** | After insertion, no external force, 5 s wait | Drift < 0.1 mm in any axis |
| **Gravity only** | Insert 2×2 on inverted baseplate (studs down), release | Brick does NOT fall off (retention > gravity = ~0.023 N for 2.3 g brick) |

### 7.3 Stability Tests

| Test | Setup | Pass Criterion |
|------|-------|---------------|
| **Insert/remove cycles** | 10 consecutive insert-then-remove cycles | No solver divergence (no NaN in qpos/qvel) |
| **Penetration cap** | Monitor `data.contact[i].dist` during all insertions | Max penetration depth < 2 mm (0.002 m) |
| **Energy bound** | Monitor total energy during insertion | Energy < 500 J at all times |
| **Post-insertion jitter** | After insertion, measure position RMS over 1 s | < 0.05 mm (0.00005 m) RMS |
| **Multi-brick stack** | Stack 3 bricks (2×2) on baseplate | All connections stable for 5 s; no oscillation or collapse |

### 7.4 Performance Tests

| Test | Setup | Pass Criterion |
|------|-------|---------------|
| **Contact count** | 2×2 brick insertion | < 100 active contacts during insertion |
| **Solver time** | 2×4 brick insertion on single CPU core | Physics runs at ≥ 1× real-time |
| **Large-scene stability** | 6 bricks in scene (various sizes) | No solver timeout or divergence over 10 s |

---

## 8. Configuration Integration

All tunable parameters from this spec will be stored in `configs/sim/lego.yaml` (created in Phase 1.2.1+). The config structure:

```yaml
lego:
  # Geometry (frozen from spec)
  geometry:
    stud_pitch: 0.008          # m
    stud_radius: 0.00235       # m (collision)
    stud_visual_radius: 0.0024 # m
    stud_height: 0.0017        # m
    tube_ring_radius: 0.0024   # m (capsule centers)
    tube_capsule_radius: 0.0005 # m
    tube_capsule_count: 8
    tube_capsule_height: 0.0016 # m
    brick_height: 0.0096       # m (standard, no stud)
    wall_thickness: 0.0012     # m
    inter_brick_gap: 0.0002    # m
    density: 1050.0            # kg/m³ (ABS)

  # Contact classes
  contact:
    stud_tube:
      solref: [0.004, 1.0]
      solimp: [0.95, 0.99, 0.001, 0.5, 3.0]
      friction: [0.5, 0.005, 0.005]
      condim: 4
    brick_surface:
      solref: [0.005, 1.0]
      solimp: [0.9, 0.95, 0.001, 0.5, 2.0]
      friction: [0.4, 0.005, 0.002]
      condim: 3
    baseplate:
      solref: [0.005, 1.0]
      solimp: [0.9, 0.95, 0.001, 0.5, 2.0]
      friction: [0.6, 0.01, 0.005]
      condim: 3

  # Tolerances
  tolerances:
    lateral_mm: 0.5
    angular_deg: 3.0
    approach_zone_mm: 2.0
    max_insertion_speed_m_s: 0.01

  # Success thresholds
  thresholds:
    insertion_success_rate: 0.95
    min_retention_force_per_stud_N: 0.3
    min_lateral_shear_per_stud_N: 0.2
    max_penetration_m: 0.002
    max_jitter_rms_m: 0.00005
    max_drift_m: 0.0001
    max_energy_J: 500.0
    hold_duration_s: 5.0
    max_active_contacts: 100

  # Solver overrides for LEGO scenes
  solver:
    iterations: 80
    ls_iterations: 10
    noslip_iterations: 0
```

---

## 9. Frozen Decisions and Future Scope

### 9.1 Frozen for Phase 1.2

These values are **locked** and should not be changed without re-running all acceptance tests:

- Stud/tube collision dimensions (Section 3.6 table)
- Contact parameter classes: `lego/stud_tube`, `lego/brick_surface`, `lego/baseplate` (Section 4)
- Capture envelope tolerances (Section 5.1)
- Success criteria thresholds (Section 7)
- Tube geometry strategy: 8-capsule ring
- Brick set: {2×2, 2×4, 2×6} at standard height (9.6 mm)

### 9.2 Tunable During Phase 1.2.2

These may be adjusted during contact tuning without re-specifying:

- Capsule count per tube (8 → 12 if friction is uneven)
- Exact solref/solimp values (within ±20% of spec values)
- Friction coefficients (within ±30%)
- Solver iteration count (50-120 range)

### 9.3 Deferred to Future Phases

- Non-standard brick shapes (slopes, plates, technic, 1×N)
- Multi-material contacts (different brick materials)
- Acoustic/force feedback
- True interference modeling (not feasible in MuJoCo)
- Brick deformation / fatigue
- Brick-to-brick lateral slide connections (Technic pins, etc.)

---

## Appendix A: MuJoCo Contact Parameter Reference

### solref: Reference Acceleration

`solref = [timeconst, dampratio]` (when both positive)

Defines the constraint's target corrective motion as a spring-damper:
- **timeconst**: softness parameter. Lower = stiffer contact. Must be ≥ 2× simulation timestep.
- **dampratio**: 1.0 = critically damped (no oscillation). Values < 1.0 create bouncing.

The stiffness `k` and damping `b` are derived as:
- `b = 2 / (dwidth × timeconst)`
- `k = d(r) / (dwidth² × timeconst² × dampratio²)`

Where `dwidth` is the impedance at the contact surface and `d(r)` is the position-dependent impedance from solimp.

### solimp: Impedance Function

`solimp = [dmin, dmax, width, midpoint, power]`

Controls the constraint's ability to generate force as a function of penetration:
- **dmin**: impedance at zero penetration (0 = no force, 1 = full force)
- **dmax**: impedance at `width` penetration
- **width**: penetration distance over which impedance transitions from dmin to dmax (meters)
- **midpoint**: center of the sigmoid transition (0-1, typically 0.5)
- **power**: steepness of the transition (higher = sharper snap)

For stiff contacts (ABS plastic): use high dmin/dmax (0.95/0.99) so force is generated immediately on contact.

### condim: Contact Dimensionality

- `condim=1`: frictionless (normal force only)
- `condim=3`: 3D sliding friction (tangent plane)
- `condim=4`: sliding + torsional friction (resists rotation)
- `condim=6`: sliding + torsional + rolling friction

### friction: Friction Coefficients

`friction = [tangent, torsional, rolling]` (3 values for condim ≥ 3)

When two geoms with different friction values collide, MuJoCo uses element-wise maximum.

---

## Appendix B: Force Estimation Worksheet

### Approximate contact force from solref/solimp

For a stud-tube contact with `solref=[0.004, 1.0]` and `solimp=[0.95, 0.99, 0.001, 0.5, 3]`:

At penetration depth `r`:
1. Impedance `d(r)` transitions from 0.95 to 0.99 over 1 mm
2. Reference acceleration: `a_ref = -(b × v + k × r)` where `b` and `k` depend on `d(r)` and `solref`
3. Actual constraint force ≈ `d(r) × a_ref × effective_mass`

**Rough estimate for single stud insertion:**
- Effective mass of 2×2 brick: ~2.3 g
- At 0.05 mm penetration (our clearance gap): `d ≈ 0.95` (barely into transition)
- With timeconst=0.004 s and brick approaching at 1 mm/s:
  - Spring-like stiffness: order of `mass / timeconst²` ≈ 0.0023 / 0.000016 ≈ 144 N/m
  - Force at 0.05 mm: ~144 × 0.00005 ≈ 0.007 N per contact point
  - With 8 capsules: ~0.056 N total per stud

This is lower than target (0.5-2.0 N). The actual force will be higher because:
1. Multiple capsules compress simultaneously during insertion (contact count multiplier)
2. Friction forces add to insertion resistance
3. Geometric interlock creates additional normal forces as stud displaces capsules

**Tuning note:** If initial forces are too low during Phase 1.2.2, decrease timeconst (increase stiffness) or increase capsule radius (more geometric interference). The spec values are starting points; the acceptance tests (Section 7) define the success criteria.

---

## Appendix C: Brick Type Quick Reference

### 2×2 Brick
- Studs: 4 (2×2 grid)
- Tubes: 1 (center)
- Outer dimensions: 15.8 × 15.8 × 9.6 mm (body) + 1.7 mm (studs)
- Mass: ~2.3 g

### 2×4 Brick
- Studs: 8 (2×4 grid)
- Tubes: 3 (centerline row)
- Outer dimensions: 31.8 × 15.8 × 9.6 mm (body) + 1.7 mm (studs)
- Mass: ~4.5 g

### 2×6 Brick
- Studs: 12 (2×6 grid)
- Tubes: 5 (centerline row)
- Outer dimensions: 47.8 × 15.8 × 9.6 mm (body) + 1.7 mm (studs)
- Mass: ~6.8 g
