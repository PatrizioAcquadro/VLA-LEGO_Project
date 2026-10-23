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


def source_grasp_clearance(env, controller, side: str, object_index: int):
    """Rank source approaches by open-jaw clearance from other visible pieces."""
    from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from vla_lego.isaac.evaluation import _corners

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
            geometry.append(quat_apply_inverse(tcp[3:].expand(len(points), -1), world - tcp[:3]))
    if not geometry:
        raise RuntimeError("Native jaw geometry unavailable")
    obstacles = []
    for index, placement in enumerate(env.placements):
        if index == object_index:
            continue
        pose = env.task_truth(index)["block_pose_w"]
        obstacles.append(
            (
                env.target.new_tensor(_corners(placement.spec.dimensions_m, pose)),
                matrix_from_quat(env.target.new_tensor(pose[3:])).T,
            )
        )

    def clear(pose):
        axes = matrix_from_quat(pose[3:]).T[None]
        distance = float("inf")
        for points in geometry:
            world = quat_apply(pose[3:].expand(len(points), -1), points) + pose[:3]
            for obstacle_points, obstacle_axes in obstacles:
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


def arm_clearance(env, solver, side: str):
    """Use separating face planes, retaining physical simulation as final authority."""
    import itertools

    import numpy as np
    import torch
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

    def parts(body, relative=None, required=True):
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
        return admissible

    return clear
