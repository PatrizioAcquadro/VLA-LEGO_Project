"""Small URDF-derived GPU arm kinematics for bounded multi-start IK."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _required(element: ET.Element, name: str) -> ET.Element:
    child = element.find(name)
    if child is None:
        raise ValueError(f"Missing {name} in URDF joint {element.get('name')}")
    return child


class ArmKinematics:
    """Use the shared URDF's chain and limits; no task or object coordinates."""

    def __init__(self, urdf_path: Path, side: str, root_pose, device: str):
        import torch
        from isaaclab.utils.math import matrix_from_quat

        robot = ET.parse(urdf_path).getroot()
        parents = {_required(j, "child").attrib["link"]: j for j in robot.findall("joint")}
        link = f"{side}_WSG32_TCP_LINK"
        joints = []
        while link in parents:
            joint = parents[link]
            joints.append(joint)
            link = _required(joint, "parent").attrib["link"]
        self.root = torch.eye(4, device=device)
        self.root[:3, :3] = matrix_from_quat(root_pose[3:])
        self.root[:3, 3] = root_pose[:3]
        self.chain = []
        self.link_names = []
        self.configuration_admission: Callable[[Any], Any] | None = None
        self.names = []
        self.limits = []
        for joint in reversed(joints):
            origin = joint.find("origin")
            if origin is None:
                origin = ET.Element("origin")
            xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            roll, pitch, yaw = [float(x) for x in origin.get("rpy", "0 0 0").split()]
            sr, cr = math.sin(roll), math.cos(roll)
            sp, cp = math.sin(pitch), math.cos(pitch)
            sy, cy = math.sin(yaw), math.cos(yaw)
            transform = torch.tensor(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, xyz[0]],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, xyz[1]],
                    [-sp, cp * sr, cp * cr, xyz[2]],
                    [0, 0, 0, 1],
                ],
                device=device,
            )
            axis = None
            if joint.get("type") != "fixed":
                if joint.get("type") != "revolute":
                    raise ValueError("Tool chain must contain only fixed/revolute joints.")
                axis = torch.tensor(
                    [float(x) for x in _required(joint, "axis").attrib["xyz"].split()],
                    device=device,
                )
                self.names.append(joint.attrib["name"])
                limit = _required(joint, "limit")
                self.limits.append([float(limit.attrib["lower"]), float(limit.attrib["upper"])])
            self.chain.append((transform, axis))
            self.link_names.append(_required(joint, "child").attrib["link"])
        if len(self.names) != 7:
            raise ValueError(f"Expected a seven-joint arm, got {self.names}")
        self.bounds = torch.tensor(self.limits, device=device)

    def forward(self, q):
        import torch
        from isaaclab.utils.math import matrix_from_quat, quat_from_angle_axis

        transform = self.root.expand(q.shape[0], -1, -1).clone()
        axes, pivots = [], []
        index = 0
        for origin, axis in self.chain:
            transform = transform @ origin
            if axis is not None:
                axes.append((transform[:, :3, :3] @ axis).clone())
                pivots.append(transform[:, :3, 3].clone())
                rotation = torch.eye(4, device=q.device).repeat(q.shape[0], 1, 1)
                rotation[:, :3, :3] = matrix_from_quat(
                    quat_from_angle_axis(q[:, index], axis.expand(q.shape[0], -1))
                )
                transform = transform @ rotation
                index += 1
        linear = torch.stack(
            [
                torch.linalg.cross(axis, transform[:, :3, 3] - pivot)
                for axis, pivot in zip(axes, pivots, strict=True)
            ],
            dim=-1,
        )
        return transform, torch.cat((linear, torch.stack(axes, dim=-1)), dim=1)

    def link_transforms(self, q, names):
        """URDF link poses for task-external geometric collision planning."""
        import torch
        from isaaclab.utils.math import matrix_from_quat, quat_from_angle_axis

        transform = self.root.expand(q.shape[0], -1, -1).clone()
        index = 0
        result = {}
        for name, (origin, axis) in zip(self.link_names, self.chain, strict=True):
            transform = transform @ origin
            if axis is not None:
                rotation = torch.eye(4, device=q.device).repeat(q.shape[0], 1, 1)
                rotation[:, :3, :3] = matrix_from_quat(
                    quat_from_angle_axis(q[:, index], axis.expand(q.shape[0], -1))
                )
                transform = transform @ rotation
                index += 1
            if name in names:
                result[name] = transform.clone()
        return result

    def solve(self, pose, current_q, *, starts: int = 48, iterations: int = 200):
        import torch
        from isaaclab.utils.math import compute_pose_error, quat_from_matrix

        if starts > 1:
            # Nearby Cartesian waypoints usually need only the current branch.
            # The same pose/limit checks apply; retain full multi-start on failure.
            try:
                return self.solve(pose, current_q, starts=1, iterations=24)
            except RuntimeError:
                pass
        generator = torch.Generator(device=current_q.device).manual_seed(20261023)
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        q = lower + torch.rand((starts, 7), generator=generator, device=current_q.device) * (
            upper - lower
        )
        q[0] = current_q
        for _ in range(iterations):
            tf, jac = self.forward(q)
            pos, rot = compute_pose_error(
                tf[:, :3, 3],
                quat_from_matrix(tf[:, :3, :3]),
                pose[:3].expand(starts, -1),
                pose[3:].expand(starts, -1),
            )
            error = torch.cat((pos, rot), dim=-1)
            if starts == 1 and bool((pos.norm(dim=-1) < 1e-6) & (rot.norm(dim=-1) < 1e-5)):
                break
            dq = (
                jac.transpose(1, 2)
                @ torch.linalg.solve(
                    jac @ jac.transpose(1, 2) + 0.02**2 * torch.eye(6, device=q.device),
                    error.unsqueeze(-1),
                )
            )[:, :, 0]
            q = (q + dq.clamp(-0.12, 0.12)).clamp(lower, upper)
        tf, _ = self.forward(q)
        pos, rot = compute_pose_error(
            tf[:, :3, 3],
            quat_from_matrix(tf[:, :3, :3]),
            pose[:3].expand(starts, -1),
            pose[3:].expand(starts, -1),
        )
        valid = (pos.norm(dim=-1) < 0.003) & (rot.norm(dim=-1) < 0.04)
        if self.configuration_admission is not None:
            valid &= self.configuration_admission(q)
        cost = (q - current_q).square().sum(dim=-1).masked_fill(~valid, torch.inf)
        if not bool(valid.any()):
            best = int((pos.norm(dim=-1) + rot.norm(dim=-1)).argmin())
            raise RuntimeError(
                f"No admissible IK solution: position={float(pos[best].norm()):.4f} m, rotation={float(rot[best].norm()):.4f} rad"
            )
        return q[int(cost.argmin())]
