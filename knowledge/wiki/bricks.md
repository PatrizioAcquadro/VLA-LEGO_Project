# Blocks and tasks

## Target task

Alex003 builds an **ordered five-block tower**, selecting each requested piece by color and size among distractors. Instructions specify bottom-to-top order; initial solvable tasks use stable support relationships and reachable pieces. Missing pieces or impossible arrangements belong to explicit stop cases.

The two arms must cooperate: the initial behavior is one arm stabilizing the existing structure while the other places a piece, followed by full release. Roles vary across episodes. Alternating independent single-arm placements or moving the idle arm does not establish cooperation. Define observable task-relevant contributions from both arms and record when they occur.

Success requires correct pieces and order, supported placements, complete release of all robot contact and stability through a calibrated dwell. The [qualification page](validation.md#assembly-qualification-targets) owns aggregate targets and campaign rules. Future arrangements include walls and bridges; large tolerant male–female insertions follow only after gravity assembly works, and LEGO is later still.

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

No STL has been imported or printed by this revision. The earlier optional printing reference is replaced by this catalog; its unverified training terms are not a dependency.

## Existing MuJoCo behavior

Current code uses three provisional cuboids: 40 × 40 × 40 mm / 20 g, 60 × 30 × 30 mm / 18 g, and 90 × 30 × 30 mm / 25 g. Uniform-cuboid inertia, friction 0.6 and the current table geometry are uncalibrated defaults, not the new printing specification.

The existing tower goal orders pieces by footprint, not by a language instruction. The triangle goal is three separate planar locations. Seeded reset and support/release evaluation exist; robot assembly does not.

The current evaluator uses position tolerance of 10% of the shortest dimension, 5° yaw tolerance, speed bounds of 0.01 m/s and 0.1 rad/s, and six samples over a 0.25 s dwell. These are legacy provisional tolerances, not automatically the future physical acceptance criteria. [Simulation workflows](simulation.md) describe that implementation.
