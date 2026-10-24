"""Bounded differential tool control; task choices belong to the caller."""

from __future__ import annotations

from typing import Any

from vla_lego.isaac.environment import ARM_JOINTS, DT
from vla_lego.isaac.kinematics import _bounded_dls_step


def _posture_step(jacobian, delta):
    """Project a seven-joint posture correction into the tool's redundant direction."""
    import torch

    _, _, vectors = torch.linalg.svd(jacobian, full_matrices=True)
    direction = vectors[:, -1]
    return direction * (direction * delta).sum(dim=-1, keepdim=True)


class ToolController:
    """Shared-asset TCP transforms and damped least-squares joint realization.

    This controller does not select pieces, stages, arm roles or recoveries.
    Physical contacts are resolved by PhysX, not by an object attachment.
    Collision-aware path admission is not yet implemented.
    """

    def __init__(self, env: Any):
        import torch
        from ihmc_alex_isaaclab.robots.purdue_frames import author_alex_purdue_frames
        from pxr import Gf, UsdGeom

        self.env = env
        self.frames = {}
        paths = author_alex_purdue_frames(
            env.stage, "/World/envs/env_0/Robot", end_effector="wsg32_umi_v1"
        )
        cache = UsdGeom.XformCache()
        for side in ("left", "right"):
            frame = env.stage.GetPrimAtPath(paths[f"{side.upper()}_WSG32_TCP_FRAME"])
            parent = frame
            body_name = f"{side.upper()}_GRIPPER_Y_LINK"
            while parent and parent.GetName() != body_name:
                parent = parent.GetParent()
            if not parent:
                raise RuntimeError(f"TCP is not anchored to {body_name}")
            local = Gf.Transform(
                cache.GetLocalToWorldTransform(frame)
                * cache.GetLocalToWorldTransform(parent).GetInverse()
            )
            quat = local.GetRotation().GetQuat()
            local_pose = torch.tensor(
                [[*local.GetTranslation(), *quat.GetImaginary(), quat.GetReal()]],
                device=env.robot.device,
                dtype=torch.float32,
            )
            joint_ids = [
                env.robot.data.joint_names.index(name)
                for name in ARM_JOINTS
                if name.startswith(side.upper())
            ]
            self.frames[side] = (
                env.robot.data.body_names.index(body_name),
                joint_ids,
                local_pose,
            )

    def pose(self, side: str):
        import torch
        from isaaclab.utils.math import combine_frame_transforms

        body_id, _, local = self.frames[side]
        body = self.env.robot.data.body_link_pose_w.torch[:, body_id]
        pos, quat = combine_frame_transforms(body[:, :3], body[:, 3:], local[:, :3], local[:, 3:])
        return torch.cat((pos, quat), dim=-1)

    def jacobian(self, side: str):
        from isaaclab.utils.math import quat_apply, skew_symmetric_matrix

        body_id, ids, local = self.frames[side]
        body = self.env.robot.data.body_link_pose_w.torch[:, body_id]
        # The fixed root is absent from PhysX's Jacobian body axis.
        jac = self.env.robot.data.body_link_jacobian_w.torch[:, body_id - 1, :, ids].clone()
        lever = quat_apply(body[:, 3:], local[:, :3])
        jac[:, :3] -= skew_symmetric_matrix(lever) @ jac[:, 3:]
        return jac

    def command(
        self,
        targets: dict[str, Any],
        *,
        period_s: float = 0.05,
        integrate_target: bool = False,
        advance: bool = True,
        postures: dict[str, Any] | None = None,
        twists: dict[str, Any] | None = None,
    ) -> dict:
        """Absolute TCP poses in meters/XYZW; bounded joint speed at each tick."""
        import torch
        from isaaclab.utils.math import compute_pose_error

        if not 0 < period_s <= 0.1:
            raise ValueError("Control period must be in (0, 0.1] seconds.")
        feedback = {}
        for side, target in targets.items():
            desired = self.env.target.new_tensor(target).reshape(1, 7)
            if not bool(torch.isfinite(desired).all()):
                raise ValueError("Tool pose must be finite.")
            quat_norm = desired[:, 3:].norm(dim=-1, keepdim=True)
            if not bool((quat_norm > 1e-6).all()):
                raise ValueError("Tool quaternion must be nonzero.")
            desired[:, 3:] /= quat_norm
            current = self.pose(side)
            position_error, rotation_error = compute_pose_error(
                current[:, :3],
                current[:, 3:],
                desired[:, :3],
                desired[:, 3:],
                rot_error_type="axis_angle",
            )
            error = torch.cat((position_error, rotation_error), dim=-1)
            jac = self.jacobian(side)
            _, ids, _ = self.frames[side]
            base = (
                self.env.target[:, ids]
                if integrate_target
                else self.env.robot.data.joint_pos.torch[:, ids]
            )
            limits = self.env.robot.data.joint_pos_limits.torch[:, ids]
            lower = (limits[:, :, 0] - base).clamp(min=-0.5 * period_s)
            upper = (limits[:, :, 1] - base).clamp(max=0.5 * period_s)
            if integrate_target:
                # Integrate a 2/s Cartesian velocity servo into the commanded
                # joint target, retaining the load-bearing PD offset. Replacing
                # it with measured q each tick creates a static payload error.
                error = error * (2.0 * period_s)
            if twists is not None and side in twists:
                velocity = self.env.target.new_tensor(twists[side]).reshape(1, 6)
                if not bool(torch.isfinite(velocity).all()):
                    raise ValueError("Tool velocity must be finite.")
                error += period_s * velocity
            dq = _bounded_dls_step(jac, error, 0.03, lower, upper)
            if postures is not None and side in postures:
                posture = torch.as_tensor(
                    postures[side], device=self.env.target.device, dtype=self.env.target.dtype
                ).reshape(1, len(ids))
                if not bool(torch.isfinite(posture).all()):
                    raise ValueError("Posture target must be finite.")
                delta = posture - self.env.robot.data.joint_pos.torch[:, ids]
                free = (dq > lower + 1e-7) & (dq < upper - 1e-7)
                dq += 0.5 * period_s * _posture_step(jac * free[:, None, :], delta) * free
            # 0.5 rad/s is a provisional pilot speed, below all asset motor limits.
            dq = dq.clamp(lower, upper)
            q = base + dq
            q = torch.maximum(torch.minimum(q, limits[:, :, 1]), limits[:, :, 0])
            self.env.set_joint_targets(
                dict(
                    zip(
                        (self.env.robot.data.joint_names[i] for i in ids),
                        q[0].tolist(),
                        strict=True,
                    )
                )
            )
            feedback[side] = {
                "position_error_m": float(position_error.norm()),
                "rotation_error_rad": float(rotation_error.norm()),
                "requested_pose": desired[0].tolist(),
                "joint_targets": q[0].tolist(),
            }
        if advance:
            self.env.step(round(period_s / DT))
        return feedback
