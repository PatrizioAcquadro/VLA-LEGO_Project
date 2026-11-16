"""Native fixed-base Alex003 foundation, consuming the shared Alex package."""

from __future__ import annotations

import math
from pathlib import Path

from vla_lego.isaac.catalog import BlockPlacement, BlockSpec
from vla_lego.isaac.physics import (
    PHYSX,
    default_solver_iterations,
    physics_profile,
    steel_physics,
)
from vla_lego.isaac.recording import EpisodeRecorder

ARM_JOINTS = tuple(
    f"{side}_{joint}"
    for side in ("LEFT", "RIGHT")
    for joint in (
        "SHOULDER_Y",
        "SHOULDER_X",
        "SHOULDER_Z",
        "ELBOW_Y",
        "WRIST_Z",
        "WRIST_X",
        "GRIPPER_Y",
    )
)
JAW_JOINTS = ("left_WSG32_JAW_OPENING", "right_WSG32_JAW_OPENING")
DT = 1 / 200
TABLE_TOP = 0.80  # Provisional task furniture; not a physical measurement.
DIAGNOSTIC_BLOCK = BlockSpec("cube", "small", "red")
BLOCK_SIZE = DIAGNOSTIC_BLOCK.dimensions_m[0]


class AssemblyEnvironment:
    """One contact-only scene; task truth is exposed separately from observations."""

    @property
    def physics_cuda_graph(self):
        return self.steel and bool(self.sim.backend_manager.backend.graphs)

    def __init__(
        self,
        cache_dir: Path,
        *,
        placements: tuple[BlockPlacement, ...] | None = None,
        mass_scale: float = 1.0,
        friction: tuple[float, float] = (0.6, 0.5),
        assembly_xy_m: tuple[float, float] | None = None,
        solver_iterations: tuple[int, int] | None = None,
        measure_pose_motion: bool = False,
        appearance: str = "legacy",
        parked_start: bool = False,
    ):
        import isaaclab.sim as sim_utils
        import torch
        from ihmc_alex_isaaclab.robots.alex_purdue import make_alex_purdue_cfg
        from ihmc_alex_isaaclab.scenes.purdue_alex003 import (
            make_purdue_alex003_reference_scene_cfg,
        )
        from ihmc_alex_isaaclab.sensors.zed_x_mini import (
            author_alex_purdue_zed_x_mini_mount,
            make_zed_x_mini_cfgs,
        )
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene
        from isaaclab.sensors import Camera, ContactSensor, ContactSensorCfg
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_physx.sim.schemas import (
            PhysxArticulationRootPropertiesCfg,
            PhysxCollisionPropertiesCfg,
            PhysxRigidBodyPropertiesCfg,
        )
        from pxr import Usd, UsdGeom, UsdPhysics

        if not torch.cuda.is_available():
            raise RuntimeError("Native CUDA is required; CPU fallback is not supported.")
        self.placements = (
            (BlockPlacement(DIAGNOSTIC_BLOCK, (0.45, 0.15)),) if placements is None else placements
        )
        if not self.placements:
            raise ValueError("The task scene requires at least one loose block.")
        if not math.isfinite(mass_scale) or mass_scale <= 0:
            raise ValueError("Mass scale must be positive and finite.")
        if len(friction) != 2 or any(not math.isfinite(v) or v < 0 for v in friction):
            raise ValueError("Static/dynamic friction must be finite and non-negative.")
        self.physics_profile = physics_profile()
        self.steel = self.physics_profile != PHYSX
        defaults = default_solver_iterations(self.physics_profile)
        solver_iterations = defaults if solver_iterations is None else solver_iterations
        if self.steel and solver_iterations != defaults:
            raise ValueError("Native Steel uses 16 PGS sweeps and no velocity-only sweeps.")
        if (
            len(solver_iterations) != 2
            or any(not isinstance(v, int) for v in solver_iterations)
            or not 1 <= solver_iterations[0] <= 255
            or not 0 <= solver_iterations[1] <= 255
        ):
            raise ValueError("PhysX solver iterations require position 1..255 and velocity 0..255.")
        self.solver_iterations = solver_iterations
        if self.physics_profile == PHYSX:
            physics = PhysxCfg(
                solver_type=1,
                solve_articulation_contact_last=True,
                enable_external_forces_every_iteration=True,
                min_position_iteration_count=solver_iterations[0],
                min_velocity_iteration_count=solver_iterations[1],
            )
        else:
            physics = steel_physics(self.physics_profile)
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=DT, device="cuda:0", gravity=(0.0, 0.0, -9.81), physics=physics
            )
        )
        cfg = make_purdue_alex003_reference_scene_cfg()
        root_pos, root_rot = cfg.robot.init_state.pos, cfg.robot.init_state.rot
        cfg.robot = make_alex_purdue_cfg(end_effector="wsg32_umi_v1")
        cfg.robot.init_state.pos, cfg.robot.init_state.rot = root_pos, root_rot
        self.robot_urdf_path = Path(cfg.robot.spawn.asset_path)
        cfg.robot.spawn.usd_dir = str(cache_dir / "robot")
        cfg.pedestal.spawn.usd_dir = str(cache_dir / "pedestal")
        cfg.robot.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=solver_iterations[0],
            solver_velocity_iteration_count=solver_iterations[1],
        )
        cfg.robot.init_state.joint_pos = {
            **dict.fromkeys((*ARM_JOINTS, "NECK_Z", "NECK_Y"), 0.0),
            "NECK_Y": 0.40,
            ".*_WSG32_JAW_.*": 0.034,
        }
        if parked_start:
            for side in ("LEFT", "RIGHT"):
                values = (0.8, 0.35 if side == "LEFT" else -0.35, 0.0, -1.8, 0.0, 0.0, 0.0)
                names = [name for name in ARM_JOINTS if name.startswith(side)]
                cfg.robot.init_state.joint_pos.update(zip(names, values, strict=True))
        cfg.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(0.70, 1.10, 0.05),
                collision_props=PhysxCollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(
                        (0.72, 0.72, 0.70)
                        if appearance == "neutral_side"
                        else (0.67, 0.56, 0.43)
                        if appearance == "light_wood"
                        else (0.4, 0.35, 0.3)
                    ),
                    roughness=0.9,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=friction[0], dynamic_friction=friction[1], restitution=0.0
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.60, 0.0, TABLE_TOP - 0.025)),
        )
        if assembly_xy_m is not None:
            if any(not math.isfinite(value) for value in assembly_xy_m):
                raise ValueError("Assembly marker coordinates must be finite.")
            for index, (dx, dy, size) in enumerate(
                (
                    (-0.06, 0.0, (0.004, 0.12, 0.0001)),
                    (0.06, 0.0, (0.004, 0.12, 0.0001)),
                    (0.0, -0.06, (0.12, 0.004, 0.0001)),
                    (0.0, 0.06, (0.12, 0.004, 0.0001)),
                )
            ):
                setattr(
                    cfg,
                    f"marker_{index}",
                    AssetBaseCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/Marker_{index}",
                        spawn=sim_utils.CuboidCfg(
                            size=size,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.01, 0.01, 0.01)
                            ),
                        ),
                        init_state=AssetBaseCfg.InitialStateCfg(
                            pos=(assembly_xy_m[0] + dx, assembly_xy_m[1] + dy, TABLE_TOP + 0.0001)
                        ),
                    ),
                )
        self.block_names = tuple(
            "block" if i == 0 else f"block_{i}" for i in range(len(self.placements))
        )
        self.block_paths = tuple(
            f"/World/envs/env_0/Block{f'_{i}' if i else ''}" for i in range(len(self.placements))
        )
        for index, (name, placement) in enumerate(
            zip(self.block_names, self.placements, strict=True)
        ):
            if any(not math.isfinite(value) for value in placement.xy_m):
                raise ValueError("Block coordinates must be finite.")
            block_cfg = RigidObjectCfg(
                prim_path=self.block_paths[index].replace("/World/envs/env_0", "{ENV_REGEX_NS}"),
                spawn=sim_utils.CuboidCfg(
                    size=placement.spec.dimensions_m,
                    rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=False),
                    collision_props=PhysxCollisionPropertiesCfg(
                        contact_offset=0.001, rest_offset=0.0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=placement.spec.mass_kg * mass_scale
                    ),
                    activate_contact_sensors=True,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=placement.spec.rgb, roughness=0.9, metallic=0.0
                    ),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=friction[0], dynamic_friction=friction[1], restitution=0.0
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(*placement.xy_m, TABLE_TOP + placement.spec.dimensions_m[2] / 2 + 0.06),
                    rot=(
                        0.0,
                        0.0,
                        math.sin(math.radians(placement.yaw_deg) / 2),
                        math.cos(math.radians(placement.yaw_deg) / 2),
                    ),
                ),
            )
            setattr(cfg, name, block_cfg)
        zed_path = "{ENV_REGEX_NS}/Robot/ZED_X_Mini" if self.steel else "{ENV_REGEX_NS}/ZED_X_Mini"
        zed_model, camera_cfg, _ = make_zed_x_mini_cfgs(zed_path, resolution="SVGA")
        if self.steel:
            spawn_robot = cfg.robot.spawn.func

            def spawn_robot_with_zed(prim_path, cfg, **pose):
                robot = spawn_robot(prim_path, cfg, **pose)
                # Clone the head and fixed camera joint together.
                zed = robot.GetPath().AppendChild("ZED_X_Mini")
                zed_model.spawn.func(str(zed), zed_model.spawn)
                author_alex_purdue_zed_x_mini_mount(
                    robot.GetStage(), str(robot.GetPath()), str(zed)
                )
                return robot

            cfg.robot.spawn.func = spawn_robot_with_zed
        else:
            cfg.zed_model = zed_model
        camera_cfg.data_types = ["rgb"]
        cfg.key_light = AssetBaseCfg(
            prim_path="/World/KeyLight",
            spawn=sim_utils.DistantLightCfg(
                intensity=1000 if appearance == "neutral_side" else 1200
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                rot=(
                    (0.0, math.sin(math.radians(15)), 0.0, math.cos(math.radians(15)))
                    if appearance == "neutral_side"
                    else (0.0, 0.0, 0.0, 1.0)
                )
            ),
        )
        cfg.replicate_physics = False
        self.scene = InteractiveScene(cfg)
        self.robot = self.scene["robot"]
        self.blocks = tuple(self.scene[name] for name in self.block_names)
        self.block = self.blocks[0]
        self.stage = self.sim.stage
        if appearance != "legacy":
            # Equal pair coefficients keep PhysX material combination from defeating
            # the declared effective block/table/jaw friction profile.
            for prim in self.stage.Traverse():
                if prim.HasAPI(UsdPhysics.MaterialAPI):
                    material = UsdPhysics.MaterialAPI(prim)
                    material.CreateStaticFrictionAttr().Set(friction[0])
                    material.CreateDynamicFrictionAttr().Set(friction[1])
                    material.CreateRestitutionAttr().Set(0.0)
        zed_prim_path = zed_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        if not self.steel:
            author_alex_purdue_zed_x_mini_mount(
                self.stage, "/World/envs/env_0/Robot", zed_prim_path
            )
        self.contact_robot_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/envs/env_0/Robot"))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and not str(prim.GetPath()).startswith(zed_prim_path)
        ]
        # The generic helper stops at the first rigid ancestor; this articulated
        # URDF nests link prims, so each link must be activated explicitly.
        for body_path in self.contact_robot_paths:
            sim_utils.activate_contact_sensors(body_path, stage=self.stage)
        self.block_contacts = tuple(
            ContactSensor(
                ContactSensorCfg(
                    prim_path=path,
                    filter_prim_paths_expr=[*self.contact_robot_paths, *self.block_paths],
                    update_period=0.0,
                    history_length=1,
                )
            )
            for path in self.block_paths
        )
        if self.steel:
            self.robot_contacts = tuple(
                ContactSensor(ContactSensorCfg(prim_path=path, update_period=0.0))
                for path in self.contact_robot_paths
            )
        if self.block_contacts:
            self.block_contact = self.block_contacts[0]
        camera_cfg.prim_path = camera_cfg.prim_path.replace(
            "{ENV_REGEX_NS}", "/World/envs/env_0" if self.steel else self.scene.env_regex_ns
        )
        self.camera = Camera(camera_cfg)
        if self.steel:
            import numpy as np
            from scipy.spatial.transform import Rotation

            robot = self.stage.GetPrimAtPath("/World/envs/env_0/Robot")
            head = next(prim for prim in Usd.PrimRange(robot) if prim.GetName() == "HEAD_LINK")
            optical = self.stage.GetPrimAtPath(camera_cfg.prim_path)
            head_world, optical_world = (
                np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T.copy()
                for prim in (head, optical)
            )
            for transform in (head_world, optical_world):
                transform[:3, :3] /= np.linalg.norm(transform[:3, :3], axis=0)
            local = np.linalg.solve(head_world, optical_world)
            self.optical_offset = torch.tensor(
                local[:3, 3], device=self.sim.device, dtype=torch.float32
            )[None]
            self.optical_rotation = torch.tensor(
                Rotation.from_matrix(local[:3, :3]).as_quat(),
                device=self.sim.device,
                dtype=torch.float32,
            )[None]
        self.sim.reset()
        if self.steel:
            solver = self.sim.backend_manager.backend.solver
            self.solver_iterations = (solver.iterations, solver.velocity_iterations)
            self.head_index = self.robot.body_names.index("HEAD_LINK")
            names = [
                path.rsplit("/", 1)[1] for path in (*self.contact_robot_paths, *self.block_paths)
            ]
            self.contact_columns = []
            for sensor in self.block_contacts:
                actual = sensor.filter_object_names
                if len(names) != len(set(names)) or set(names) != set(actual):
                    raise RuntimeError("Steel contact partner names did not resolve uniquely.")
                self.contact_columns.append([actual.index(name) for name in names])
        else:
            self.robot_contact_view = self.sim.physics_sim_view.create_rigid_contact_view(
                self.contact_robot_paths, filter_patterns=[[] for _ in self.contact_robot_paths]
            )
            if self.robot_contact_view.sensor_count != len(self.contact_robot_paths):
                raise RuntimeError("Incomplete native robot contact reporting.")
        self.scene.reset()
        self.target = self.robot.data.default_joint_pos.torch.clone()
        self.requested_target = self.target.clone()
        self.robot.set_joint_position_target_index(target=self.target)
        self.jaw_ids = [self.robot.data.joint_names.index(name) for name in JAW_JOINTS]
        self.time = 0.0
        self.image_time: float | None = None
        self.physics_step = 0
        self.last_command_time = 0.0
        self.recorder: EpisodeRecorder | None = None
        self.motion = None
        if measure_pose_motion:
            from vla_lego.isaac.motion import PoseMotionWindow

            self.motion = PoseMotionWindow(
                torch.stack([block.data.root_pose_w.torch[0] for block in self.blocks])
            )

    def set_joint_targets(self, targets: dict[str, float]) -> None:
        """Named low-level targets, bounded by the shared robot's hard limits."""
        if self.physics_step % 10:
            raise RuntimeError("Joint target changes must occur on the 20 Hz control clock.")
        for name, value in targets.items():
            index = self.robot.data.joint_names.index(name)
            lower, upper = self.robot.data.joint_pos_limits.torch[0, index].tolist()
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Target outside joint limits: {name}={value}")
            if name.endswith("FOLLOWER"):
                raise ValueError("The shared mimic constraint owns follower targets.")
            self.requested_target[0, index] = value
            # One 20 Hz target tick, shared by teachers and autonomous policies.
            step = (0.4 if name in JAW_JOINTS else 0.5) * 0.05
            previous = float(self.target[0, index])
            self.target[0, index] = min(max(value, previous - step), previous + step)
        self.robot.set_joint_position_target_index(target=self.target)
        self.last_command_time = self.time

    def step(self, count: int = 1, *, render: bool = False) -> None:
        from ihmc_alex_isaaclab.robots.purdue_physics import ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS

        followers = {name for name, _ in ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS}
        driven = [i for i, name in enumerate(self.robot.data.joint_names) if name not in followers]
        for _ in range(count):
            if self.recorder is not None and self.physics_step % 10 == 0:
                self.capture()
                action_names = (*ARM_JOINTS, *JAW_JOINTS)
                action_ids = [self.robot.data.joint_names.index(name) for name in action_names]
                self.recorder.record(
                    self.observe(),
                    self.target[0, action_ids].detach().cpu().numpy(),
                    action_names,
                    [self.task_truth(i) for i in range(len(self.blocks))],
                    requested_action=self.requested_target[0, action_ids].detach().cpu().numpy(),
                    command_time_s=self.last_command_time,
                )
            if self.motion is not None and self.physics_step % 10 == 0:
                self.motion.reset()
            gravity = self.robot.data.gravity_compensation_forces.torch[
                :, self.robot.num_base_dofs :
            ][:, driven]
            self.robot.set_joint_effort_target_index(target=gravity, joint_ids=driven)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self.steel and self.physics_step == 0 and not self.physics_cuda_graph:
                raise RuntimeError("Steel CUDA Graph was not captured")
            self.scene.update(DT)
            for sensor in self.block_contacts:
                sensor.update(DT)
            if self.steel:
                for sensor in self.robot_contacts:
                    sensor.update(DT)
            if self.motion is not None:
                import torch

                self.motion.update(
                    torch.stack([block.data.root_pose_w.torch[0] for block in self.blocks]), DT
                )
            self.physics_step += 1
            self.time = self.physics_step * DT
        if render:
            self.capture()

    def capture(self) -> None:
        if self.steel:
            from isaaclab.utils.math import combine_frame_transforms

            position, rotation = combine_frame_transforms(
                self.robot.data.body_link_pos_w.torch[:, self.head_index],
                self.robot.data.body_link_quat_w.torch[:, self.head_index],
                self.optical_offset,
                self.optical_rotation,
            )
            self.camera.set_world_poses(position, rotation, convention="opengl")
        else:
            self.sim.forward()
        self.sim.render()
        elapsed = DT if self.image_time is None else max(DT, self.time - self.image_time)
        self.camera.update(elapsed, force_recompute=True)
        self.image_time = self.time

    def observe(self) -> dict:
        """Only deployable sensor/joint signals, never block poses or teacher stages."""
        names = (*ARM_JOINTS, "NECK_Z", "NECK_Y", *JAW_JOINTS)
        if self.image_time is None:
            raise RuntimeError("Capture a head frame before requesting observations.")
        indices = [self.robot.data.joint_names.index(name) for name in names]
        return {
            "timestamp_s": self.time,
            "image_timestamp_s": self.image_time,
            "head_rgb": self.camera.data.output["rgb"].torch[0, :, :, :3].cpu().numpy(),
            "joint_names": names,
            "joint_position": self.robot.data.joint_pos.torch[0, indices].detach().cpu().numpy(),
            "joint_velocity": self.robot.data.joint_vel.torch[0, indices].detach().cpu().numpy(),
        }

    def task_truth(self, block_index: int = 0) -> dict:
        """Privileged diagnostic/evaluator channel, excluded from observe()."""
        block = self.blocks[block_index]
        sensor = self.block_contacts[block_index]
        if self.steel:
            self._check_steel_contacts()
            matrix = sensor.data.normal_force_matrix_w.torch[
                0, 0, self.contact_columns[block_index]
            ]
            net = sensor.data.net_normal_forces_w.torch[0, 0]
        else:
            matrix = sensor.data.force_matrix_w.torch[0, 0]
            net = sensor.data.net_forces_w.torch[0, 0]
        robot_forces = matrix[: len(self.contact_robot_paths)]
        result = {
            "block_pose_w": block.data.root_pose_w.torch[0].tolist(),
            "block_velocity_w": block.data.root_vel_w.torch[0].tolist(),
            "robot_contact_force_n": float(robot_forces.norm(dim=-1).sum()),
            "robot_contact_forces_n": robot_forces.tolist(),
            "block_contact_forces_n": matrix[len(self.contact_robot_paths) :].tolist(),
            # PhysX GPU filters cannot target the static table. Retain the residual
            # as environmental contact, with geometry establishing table support.
            "nonrobot_contact_force_n": float((net - matrix.sum(dim=0)).norm()),
        }
        if self.motion is not None:
            result["motion_speed_max"] = self.motion.peaks[block_index].tolist()
            result["motion_window_s"] = self.motion.duration_s
        return result

    def _check_steel_contacts(self):
        backend = self.sim.backend_manager.backend
        backend.solver.check_constraint_capacity()
        if (
            int(backend.contacts.rigid_contact_count.numpy()[0])
            >= backend.contacts.rigid_contact_max
        ):
            raise RuntimeError("Steel contact buffer is saturated.")

    def robot_diagnostics(self) -> dict:
        """Control/contact evidence; never supplied as hidden task decisions."""
        if self.steel:
            import torch

            self._check_steel_contacts()
            forces = torch.stack(
                [s.data.net_normal_forces_w.torch[0, 0] for s in self.robot_contacts]
            )
        else:
            import warp as wp

            forces = wp.to_torch(self.robot_contact_view.get_net_contact_forces(dt=DT)).reshape(
                -1, 3
            )
        return {
            "joint_position": self.robot.data.joint_pos.torch[0].tolist(),
            "applied_joint_effort": self.robot.data.applied_torque.torch[0].tolist(),
            "robot_contact_paths": self.contact_robot_paths,
            "robot_contact_forces_n": forces.tolist(),
        }
