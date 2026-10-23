# RGB-Objects seed geometry

Unmodified binary STL files from Google DeepMind `dm_robotics`, commit
`59bebe7ef6e5050ec1b2443f59857836bb727742` (Apache-2.0).
The applicable upstream license is retained in `LICENSE_RGB_OBJECTS.txt`.

Upstream root: https://github.com/google-deepmind/dm_robotics/tree/59bebe7ef6e5050ec1b2443f59857836bb727742/py/manipulation/props/rgb_objects/assets/rgb_v1.3/meshes

- `cube_s0.stl`: `test_triplets/s0_sds4_shr0_drf0_hlw0_shx0_shy0_scx50_scy50_scz50.stl`.
- `prism_f6.stl`: `heldout/f6_sds4_shr0_drf0_hlw0_shx0_shy0_scx47_scy47_scz68.stl`.

Vertices are in meters. Each file has 12 triangles and is exactly a centered
cuboid: 50 x 50 x 50 mm and 47 x 47 x 68 mm, respectively. The Isaac task uses
analytically identical cuboid collision/render primitives at declared scales;
this imports geometry only, without the upstream MuJoCo runtime. Source
held-out membership is not preserved as this project's training split.

Color is assigned independently of shape and scale by this project. Mass,
friction and compliance are provisional simulation parameters, not supplied by
these STL files or established by physical print measurements.
