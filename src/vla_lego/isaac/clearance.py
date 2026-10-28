"""Native collision geometry for teacher arm and source-grasp planning."""

from __future__ import annotations


def _separation_distance(points, axes, obstacle_points, obstacle_axes):
    import torch

    directions = torch.cat((axes, obstacle_axes.expand(points.shape[0], -1, -1)), dim=1)
    a = points @ directions.transpose(1, 2)
    b = obstacle_points.expand(points.shape[0], -1, -1) @ directions.transpose(1, 2)
    return torch.maximum(a.amin(dim=1) - b.amax(dim=1), b.amin(dim=1) - a.amax(dim=1)).amax(dim=1)


def _separated(points, axes, obstacle_points, obstacle_axes):
    return _separation_distance(points, axes, obstacle_points, obstacle_axes) > 0


def _sphere_separated(points, axes, center, radius):
    """Prove separation using exact sphere projections, without a box enlargement."""
    import torch

    relative = points - center
    nearest = relative[
        torch.arange(len(points), device=points.device), relative.square().sum(-1).argmin(-1)
    ]
    directions = torch.cat((axes, nearest[:, None], relative.mean(1)[:, None]), dim=1)
    projected = relative @ directions.transpose(1, 2)
    reach = radius * directions.norm(dim=-1)
    return torch.maximum(projected.amin(1) - reach, -reach - projected.amax(1)).amax(1) > 0


def _unit_axes(vectors):
    import torch

    length = vectors.norm(dim=-1)
    axes = vectors[length > 1e-12] / length[length > 1e-12, None]
    return torch.unique(torch.round(axes * 1e6) / 1e6, dim=0)


def _mesh_axes(points, counts, indices):
    """Face and edge directions of the native convex collision mesh."""
    import torch

    triangles, start = [], 0
    for count in counts:
        face = indices[start : start + count]
        triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
        start += count
    faces = points[torch.tensor(triangles, device=points.device)]
    first, second = faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]
    return _unit_axes(torch.linalg.cross(first, second)), _unit_axes(
        torch.cat((first, second, faces[:, 2] - faces[:, 1]))
    )


def source_grasp_span(env, object_index: int, yaw_deg: float) -> float:
    """Observed horizontal block span along the top-down jaw closing axis."""
    import math

    from vla_lego.isaac.evaluation import _corners

    yaw = math.radians(yaw_deg)
    state = env.task_truth(object_index)
    outline = _corners(env.placements[object_index].spec.dimensions_m, state["block_pose_w"])
    projected = [-math.sin(yaw) * p[0] + math.cos(yaw) * p[1] for p in outline]
    return float(max(projected) - min(projected))


def _jaw_geometry(env, controller, side: str):
    """Native collision meshes expressed in the measured tool frame."""
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    tcp = controller.pose(side)[0]
    geometry = []
    for path in env.contact_robot_paths:
        if f"{side}_WSG32_" not in path:
            continue
        body = env.stage.GetPrimAtPath(path)
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        pose = env.robot.data.body_link_pose_w.torch[
            0, env.robot.data.body_names.index(body.GetName())
        ]
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not prim.IsA(UsdGeom.Mesh):
                raise RuntimeError("Unsupported native jaw collider")
            transform = cache.GetLocalToWorldTransform(prim) * inverse
            points = env.target.new_tensor(
                [
                    transform.Transform(Gf.Vec3d(*p))
                    for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()
                ]
            )
            world = quat_apply(pose[3:].expand(len(points), -1), points) + pose[:3]
            local = quat_apply_inverse(tcp[3:].expand(len(points), -1), world - tcp[:3])
            mesh = UsdGeom.Mesh(prim)
            normals, edges = _mesh_axes(
                local, mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
            )
            geometry.append((local, normals, edges))
    if not geometry:
        raise RuntimeError("Native jaw geometry unavailable")
    return geometry


def source_grasp_height(env, controller, side, orientation, object_height: float) -> float:
    """Keep the distal jaw tips above the payload bottom at supported placement."""
    from isaaclab.utils.math import quat_apply

    geometry = _jaw_geometry(env, controller, side)
    lowest = min(
        float(quat_apply(orientation.expand(len(points), -1), points)[:, 2].min())
        for points, _, _ in geometry
    )
    # The cooperative search may descend another 6 mm; native tracking admits 5 mm.
    return max(0.06, -object_height / 2 - lowest + 0.006 + 0.005)


def source_grasp_clearance(
    env, controller, side: str, object_index: int | None, *, other_hand: bool = False
):
    """Rank source approaches by open-jaw clearance from other visible pieces."""
    import torch
    from isaaclab.utils.math import matrix_from_quat, quat_apply

    from vla_lego.isaac.evaluation import _corners

    geometry = _jaw_geometry(env, controller, side)
    obstacles = []
    for index, placement in enumerate(env.placements):
        if index == object_index:
            continue
        pose = env.task_truth(index)["block_pose_w"]
        axes = matrix_from_quat(env.target.new_tensor(pose[3:])).T
        obstacles.append(
            (env.target.new_tensor(_corners(placement.spec.dimensions_m, pose)), axes, axes)
        )
    if other_hand:
        opposite = "right" if side == "left" else "left"
        pose = controller.pose(opposite)[0]
        rotation = matrix_from_quat(pose[3:])
        for points, normals, edges in _jaw_geometry(env, controller, opposite):
            obstacles.append(
                (points @ rotation.T + pose[:3], normals @ rotation.T, edges @ rotation.T)
            )

    def clear(pose):
        rotation = matrix_from_quat(pose[3:])
        distance = float("inf")
        for points, normals, edges in geometry:
            world = quat_apply(pose[3:].expand(len(points), -1), points) + pose[:3]
            for obstacle_points, obstacle_axes, obstacle_edges in obstacles:
                crossing = _unit_axes(
                    (edges @ rotation.T)[:, None, :]
                    .expand(-1, len(obstacle_edges), -1)
                    .cross(obstacle_edges[None].expand(len(edges), -1, -1), dim=-1)
                    .reshape(-1, 3)
                )
                axes = torch.cat((rotation.T, normals @ rotation.T, crossing))[None]
                distance = min(
                    distance,
                    float(
                        _separation_distance(
                            world[None], axes, obstacle_points, obstacle_axes
                        ).item()
                    ),
                )
        return distance

    return clear


def admit_empty_hand_path(env, controller, solver, side: str, goal, *, start=None):
    """Preflight every park target against native arms, pieces and the other jaws."""
    import math

    import torch
    from isaaclab.utils.math import quat_from_matrix

    _, ids, _ = controller.frames[side]
    if start is None:
        start = env.robot.data.joint_pos.torch[0, ids]
    if bool(((goal < solver.bounds[:, 0]) | (goal > solver.bounds[:, 1])).any()):
        raise RuntimeError("Empty-hand park exceeds native joint limits")
    ticks = max(1, math.ceil(float((goal - start).abs().max()) / 0.025))
    path = torch.lerp(
        start[None], goal[None], torch.linspace(0, 1, ticks + 1, device=start.device)[:, None]
    )
    if not bool(arm_clearance(env, solver, side)(path).all()):
        raise RuntimeError("Empty-hand park intersects native arm, head or table geometry")
    clear = source_grasp_clearance(env, controller, side, None, other_hand=True)
    transforms, _ = solver.forward(path)
    minimum = float("inf")
    for transform in transforms:
        pose = torch.cat((transform[:3, 3], quat_from_matrix(transform[:3, :3])))
        minimum = min(minimum, clear(pose))
        if minimum <= 0:
            raise RuntimeError("Empty-hand park intersects a piece or the other jaws")
    return minimum


def plan_empty_hand_path(env, controller, solver, side: str, goal):
    """Admit the whole retirement before moving, including a rearward fallback."""
    import torch
    from isaaclab.utils.math import quat_from_matrix

    try:
        return [goal], admit_empty_hand_path(env, controller, solver, side, goal)
    except RuntimeError:
        if bool(((goal < solver.bounds[:, 0]) | (goal > solver.bounds[:, 1])).any()):
            raise

    _, ids, _ = controller.frames[side]
    start = env.robot.data.joint_pos.torch[0, ids].clone()
    pose = controller.pose(side)[0].clone()
    pose[0] -= 0.08
    native = arm_clearance(env, solver, side)
    clear = source_grasp_clearance(env, controller, side, None, other_hand=True)

    def admissible(configurations):
        valid = native(configurations)
        transforms, _ = solver.forward(configurations)
        for i in valid.nonzero().flatten().tolist():
            transform = transforms[i]
            tool = torch.cat((transform[:3, 3], quat_from_matrix(transform[:3, :3])))
            valid[i] = clear(tool) > 0
        return valid

    previous = solver.configuration_admission
    solver.configuration_admission = admissible
    try:
        waypoint = solver.solve(pose, start)
    finally:
        solver.configuration_admission = previous
    first = admit_empty_hand_path(env, controller, solver, side, waypoint)
    second = admit_empty_hand_path(env, controller, solver, side, goal, start=waypoint)
    return [waypoint, goal], min(first, second)


def arm_clearance(env, solver, side: str):
    """Use separating face planes, retaining physical simulation as final authority."""
    import itertools

    import numpy as np
    import torch
    from isaaclab.utils.math import matrix_from_quat
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = env.stage
    transforms = UsdGeom.XformCache()

    def body_named(name):
        return next(
            (
                stage.GetPrimAtPath(p)
                for p in env.contact_robot_paths
                if p.rsplit("/", 1)[-1] == name
            ),
            None,
        )

    def parts(body, relative=None, required=True, spheres=None):
        if not body:
            raise RuntimeError("Native body unavailable for wrist clearance")
        result = []
        # Most shared robot collision meshes remain instance proxies.
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            owner = prim
            while owner != body and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if owner != body:
                continue
            matrix = transforms.GetLocalToWorldTransform(prim)
            if relative:
                matrix = matrix * transforms.GetLocalToWorldTransform(relative).GetInverse()
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                vertices = np.array(
                    [matrix.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()]
                )
                counts = mesh.GetFaceVertexCountsAttr().Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                triangles, start = [], 0
                for count in counts:
                    face = indices[start : start + count]
                    triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
                    start += count
                faces = vertices[np.asarray(triangles)]
                normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
            elif prim.IsA(UsdGeom.Cube):
                half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
                # Collision-only furniture can be invisible to a render BBoxCache.
                vertices = np.asarray(
                    [
                        matrix.Transform(Gf.Vec3d(*point))
                        for point in itertools.product((-half, half), repeat=3)
                    ]
                )
                normals = np.asarray([matrix.TransformDir(Gf.Vec3d(*axis)) for axis in np.eye(3)])
            elif prim.IsA(UsdGeom.Sphere) and spheres is not None:
                scales = [
                    Gf.Vec3d(matrix.TransformDir(Gf.Vec3d(*axis))).GetLength() for axis in np.eye(3)
                ]
                if max(scales) - min(scales) > 1e-6 * max(scales):
                    raise RuntimeError("Nonuniformly scaled native sphere is not spherical")
                spheres.append(
                    (
                        env.target.new_tensor(matrix.Transform(Gf.Vec3d(0, 0, 0))),
                        UsdGeom.Sphere(prim).GetRadiusAttr().Get() * scales[0],
                    )
                )
                continue
            else:
                raise RuntimeError(f"Unsupported native clearance primitive: {prim.GetTypeName()}")
            lengths = np.linalg.norm(normals, axis=1)
            normals = normals[lengths > 1e-12] / lengths[lengths > 1e-12, None]
            # Any separating direction proves disjointness; rounding only deduplicates axes.
            normals = np.unique(np.round(normals, 6), axis=0)
            result.append((env.target.new_tensor(vertices), env.target.new_tensor(normals)))
        if not result and required:
            raise RuntimeError("Native collision geometry unavailable for arm clearance")
        return result

    moving = {}
    for name in solver.link_names:
        body = body_named(name)
        if body:
            geometry = parts(body, relative=body, required=False)
            if geometry:
                moving[name] = geometry
    if not moving:
        raise RuntimeError("Native arm collision geometry unavailable")
    pedestal = stage.GetPrimAtPath("/World/envs/env_0/PurduePedestal")
    if not pedestal:
        raise RuntimeError("Native pedestal unavailable for arm clearance")
    furniture = [
        prim
        for prim in Usd.PrimRange(pedestal, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    obstacles = [
        (body.GetName(), points, axes)
        for body in (
            stage.GetPrimAtPath("/World/envs/env_0/Table"),
            body_named("PEDESTAL_LINK"),
            *furniture,
        )
        for points, axes in parts(body)
    ]
    spherical_obstacles: list[tuple] = []
    # The head is not in either arm chain, but can obstruct a high cross-body reach.
    for name in ("NECK_Z_LINK", "HEAD_LINK"):
        body = body_named(name)
        if body is None:
            raise RuntimeError(f"Native head collision body unavailable: {name}")
        pose = env.robot.data.body_link_pose_w.torch[0, env.robot.data.body_names.index(name)]
        rotation = matrix_from_quat(pose[3:])
        spheres: list = []
        for points, axes in parts(body, relative=body, required=False, spheres=spheres):
            obstacles.append((name, points @ rotation.T + pose[:3], axes @ rotation.T))
        spherical_obstacles.extend(
            (name, center @ rotation.T + pose[:3], radius) for center, radius in spheres
        )
    adjacent = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetCollisionEnabledAttr().Get():
            continue
        first, second = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if first and second:
            adjacent.add(frozenset((first[0].name, second[0].name)))

    def clear(q):
        transforms = solver.link_transforms(q, moving)
        admissible = torch.ones(q.shape[0], dtype=torch.bool, device=q.device)
        for name, geometry in moving.items():
            transform = transforms[name]
            for points, axes in geometry:
                world = points @ transform[:, :3, :3].transpose(1, 2) + transform[:, None, :3, 3]
                directions = axes @ transform[:, :3, :3].transpose(1, 2)
                for owner, obstacle_points, obstacle_axes in obstacles:
                    # PhysX excludes the directly connected shoulder/base pair.
                    if frozenset((name, owner)) in adjacent:
                        continue
                    admissible &= _separated(world, directions, obstacle_points, obstacle_axes)
                for owner, center, radius in spherical_obstacles:
                    if frozenset((name, owner)) not in adjacent:
                        admissible &= _sphere_separated(world, directions, center, radius)
        return admissible

    return clear
