# Blocks and tasks

## Target task

Alex003 builds an **ordered five-block tower**, selecting each requested piece by color and size among distractors. Instructions specify bottom-to-top order; initial solvable tasks use stable support relationships and reachable pieces. Missing pieces or impossible arrangements belong to explicit stop cases.

Each nominal episode starts with loose pieces and an empty designated assembly area. The target is five distinct physical blocks in five vertical levels, with the base on the table and each later block supported by the preceding one. The policy receives the instruction and deployable observations, not target object IDs, simulator coordinates or a teacher-selected next piece. Human arrangement/reset between episodes is allowed; no human manipulation or task guidance occurs after the autonomous attempt begins.

The instruction must identify each requested piece by visible attributes, adding shape when color and size would otherwise be ambiguous. Visibly equivalent duplicate pieces are interchangeable; an invisible serial identity cannot determine success. Include distractors that distinguish color from size: at least one matches a requested color but has the wrong size, and another matches a requested size but has the wrong color. Phase 1 fixes the vocabulary, admitted dimensions and feasible orders; it does not silently sort every tower by footprint instead of following the instruction.

The two arms must cooperate. For the initial task, every successful episode includes at least one placement above the base in which one arm intentionally stabilizes the existing structure while the other aligns and places a block. Their task-relevant actions overlap; incidental contact or an idle arm touching the table does not count. Roles vary across episodes, but a role swap within every episode is not required. Phase 1 defines the contact/motion evidence for functional stabilization, and Phase 7 confirms its observable physical equivalent.

Success requires correct pieces and order, supported placements, complete release of all robot contact and stability through a calibrated dwell. The [qualification page](validation.md#assembly-qualification-targets) owns aggregate targets and campaign rules. Future arrangements include walls and bridges; large tolerant male–female insertions follow only after gravity assembly works, and LEGO is later still.

A recovery is complete only when the robot resumes and finishes that same task within its allowed budget. A dropped piece may be recovered only while it remains reachable under the declared scene rules; a human reset starts a new episode and cannot rescue the previous result. If an instruction is non-completable, the expected behavior is a justified bounded attempt to resolve uncertainty or a safe stop, not an invented replacement task. Non-completable episodes are scored separately from tower completion.

## Printable catalog

Use [DeepMind RGB-Objects](https://github.com/google-deepmind/dm_robotics/tree/main/py/manipulation/props/rgb_objects): 152 documented STL geometries derived from a cube, with shape/scale parameters and existing stacking research. The repository uses [Apache-2.0](https://github.com/google-deepmind/dm_robotics/blob/main/LICENSE); preserve applicable notices and source attribution when importing assets. Reuse the meshes, not its MuJoCo runtime.

Start with cubes and rectangular prisms, two graspable scales and six distinct colors. Assign color independently of geometry in simulation and the expanded physical catalog; do not inherit the source benchmark's fixed color/shape associations.

| Seed asset | Documented source scale | Intended use |
|---|---|---|
| [s0 cube STL](https://github.com/google-deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/assets/rgb_v1.3/meshes/test_triplets/s0_sds4_shr0_drf0_hlw0_shx0_shy0_scx50_scy50_scz50.stl) | 50 × 50 × 50 mm | Simple grasp and flat support |
| [f6 rectangular prism STL](https://github.com/google-deepmind/dm_robotics/blob/main/py/manipulation/props/rgb_objects/assets/rgb_v1.3/meshes/heldout/f6_sds4_shr0_drf0_hlw0_shx0_shy0_scx47_scy47_scz68.stl) | 47 × 47 × 68 mm | Different dimensions and support orientation |

Select two final print scales from gripper clearance and stable support, starting with a **12-piece calibration sample** covering both geometry families, both scales and all six colors. This small sample is not a full factorial dataset or sufficient evidence of generalization. Expand and cross color/shape/scale combinations before broad physical training or qualification, including multiple distinguishable target and distractor pieces.

The source includes 103 training meshes, 13 test-triplet meshes and 36 held-out meshes. Using its held-out cuboids as project training pieces is allowed by our task design, but does not preserve the original benchmark split. Define project splits explicitly and claim only the corresponding generalization.

## Physical calibration

These are lightweight, easily separable gravity blocks, without studs, sockets, locks, magnets, welds or force-assisted retention. STL geometry does not specify printed mass, friction, wall stiffness or grasp quality. Measure dimensions, mass, surface/slip behavior, compliance, gripper clearance and released stability on actual prints; represent relevant print-to-print variation in simulation.

The shared Alex measurements record about 70 mm fully open inner finger spacing. This is a geometric reference, not usable clearance for every approach or an established safe grasp envelope. Confirm actual fingertips, orientation and access. Treat print material, wall/infill settings and contact calibration as a consistent initial manufacturing choice before adding variation.

The two seed STLs have now been imported unchanged from upstream commit `59bebe7e`, with the source license and [asset provenance](../../src/vla_lego/isaac/assets/blocks/PROVENANCE.md). Both are exactly 12-triangle centered cuboids in meters; native Isaac uses analytically identical primitives. `vla_lego.isaac.catalog` crosses both geometries, scales 0.8/1.0 and all six colors (24 combinations). The provisional effective bulk density gives 25 g for a 40 mm cube and about 58.7 g for the large prism; these are assumptions, not printed measurements. No physical print or contact calibration is claimed. The earlier optional printing reference is replaced by this catalog; its unverified training terms are not a dependency.
