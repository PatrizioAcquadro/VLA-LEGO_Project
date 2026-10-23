# Lightweight bricks and tasks

## Scope and physical model

Current bricks are lightweight, variable-size cuboids that can be picked, placed and stacked under gravity. They have no studs, sockets, baseplate locking, magnetic connection, weld or retention assistance. Actual LEGO is deferred to its [extension plan](implementation_phases/phase-8-transfer-handoff.md#future-extension-real-lego).

`BrickSpec` validates positive finite XYZ dimensions in metres, mass in kilograms and RGBA color. The initial catalog is [packaged YAML](../../configs/bricks/default.yaml):

| Piece | XYZ dimensions | Effective mass |
|---|---|---|
| Cube | 40×40×40 mm | 20 g |
| Short | 60×30×30 mm | 18 g |
| Long | 90×30×30 mm | 25 g |

Masses represent provisional hollow/low-infill prints, not solid ABS density. Inertia uses a uniform cuboid approximation. Contact friction starts at 0.6. The table top is at 0.77 m, centered at `(0.45,0)`, with a 0.50×0.60 m surface. None of these defaults qualifies a printable design or a robot grasp.

`scene_xml()` composes bricks with the maintained Alex scene; standalone exported MJCF is also available. `EpisodeManager` compiles once, resolves free-joint addresses by name, clears state/forces on reset, samples separated yawed bounding boxes from a seed and settles for 0.4 s. Custom catalogs are supplied through `load_bricks(path)` or the constructor; changing YAML alone does not prove grasp feasibility.

## Goals and evaluation

`make_goal(bricks, "tower")` orders pieces by footprint, centers them and records the supporting brick for each level. `"triangle"` requires exactly three pieces and creates three separated target positions in a plane. It does not form a filled triangle, an interlocking frame or triangular individual pieces.

`evaluate_goal()` checks position, yaw with footprint symmetries, positive support contact, low linear/angular velocity and absence of robot contact. Default position tolerance is 10% of the shortest piece dimension; yaw tolerance is 5°. Linear/angular speed bounds are 0.01 m/s and 0.1 rad/s. All six samples across a 0.25-s dwell must pass. These are simulation evaluation tolerances requiring calibration against printed pieces; they are not hard safety limits. A misplaced, unsupported, moving or still-held piece must fail even if it looks close in one image.

See [simulation](simulation.md) for executable examples and [2.1](implementation_phases/phase-2-data.md#21-robot-trajectory-generation) for actual assembly demonstrations.

## Printable prototype direction

A relevant public design is [Endless Design Blocks by EndlessDesignLab](https://cults3d.com/en/3d-model/art/endless-design-blocks-an-infinitely-scalable-building-block-set-for-geometric): a scalable geometric building set with STL/3MF files, assembled through geometry and gravity rather than snaps, magnets or locking. The pieces have a dual-sided cube-corner/triangular geometry, rather than the simple cuboids used here. The listing identifies CC BY-SA and also displays a “No AI” label. Treat it as a printing reference; resolve the additional label before any use in model training or synthetic-data generation, and retain required attribution. No third-party mesh or license-dependent asset has been imported into this repository.

Use the design as an optional physical reference, not a qualified Alex part. A simpler first prototype is a hollow cuboid matching the catalog, printed with enough wall stiffness for gripping. Measure actual mass, dimensions, grip opening and slip/stack behavior; then update the catalog and acceptance ranges from those measurements. Scale must account for gripper clearance and stable support area. Begin with a small set and one pick-and-place before expanding shapes or buying hardware.
