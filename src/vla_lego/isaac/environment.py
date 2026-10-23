"""Native fixed-base Alex003 foundation, consuming the shared Alex package."""

from __future__ import annotations

import math
from pathlib import Path

from vla_lego.isaac.catalog import BlockPlacement, BlockSpec
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

    def __init__(
        self,
        cache_dir: Path,
        *,
        placements: tuple[BlockPlacement, ...] | None = None,
        mass_scale: float = 1.0,
        friction: tuple[float, float] = (0.6, 0.5),
        assembly_xy_m: tuple[float, float] | None = None,
    ):
        import isaaclab.sim as sim_utils
        import omni.usd
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
        from pxr import Usd, UsdPhysics

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
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=DT,
                device="cuda:0",
                gravity=(0.0, 0.0, -9.81),
                physics=PhysxCfg(
                    solver_type=1,
                    solve_articulation_contact_last=True,
                    enable_external_forces_every_iteration=True,
                    min_position_iteration_count=8,
                    min_velocity_iteration_count=4,
                ),
            )
        )
        cfg = make_purdue_alex003_reference_scene_cfg()
        root_pos, root_rot = cfg.robot.init_state.pos, cfg.robot.init_state.rot
        cfg.robot = make_alex_purdue_cfg(end_effector="wsg32_umi_v1")
        cfg.robot.init_state.pos, cfg.robot.init_state.rot = root_pos, root_rot
        cfg.robot.spawn.usd_dir = str(cache_dir / "robot")
        cfg.pedestal.spawn.usd_dir = str(cache_dir / "pedestal")
        cfg.robot.spawn.articulation_props = PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        )
        cfg.robot.init_state.joint_pos = {
            **dict.fromkeys((*ARM_JOINTS, "NECK_Z", "NECK_Y"), 0.0),
            "NECK_Y": 0.40,
            ".*_WSG32_JAW_.*": 0.034,
        }
        cfg.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(0.70, 1.10, 0.05),
                collision_props=PhysxCollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.35, 0.3)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0
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
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=placement.spec.rgb),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=friction[0], dynamic_friction=friction[1], restitution=0.0
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(*placement.xy_m, TABLE_TOP + placement.spec.dimensions_m[2] / 2 + 0.06)
                ),
            )
            setattr(cfg, name, block_cfg)
        cfg.zed_model, camera_cfg, _ = make_zed_x_mini_cfgs(
            "{ENV_REGEX_NS}/ZED_X_Mini", resolution="SVGA"
        )
        camera_cfg.data_types = ["rgb"]
        cfg.key_light = AssetBaseCfg(
            prim_path="/World/KeyLight", spawn=sim_utils.DistantLightCfg(intensity=1200)
        )
        cfg.replicate_physics = False
        self.scene = InteractiveScene(cfg)
        self.robot = self.scene["robot"]
        self.blocks = tuple(self.scene[name] for name in self.block_names)
        self.block = self.blocks[0]
        self.stage = omni.usd.get_context().get_stage()
        self.contact_robot_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/envs/env_0/Robot"))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
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
        self.block_contact = self.block_contacts[0]
        author_alex_purdue_zed_x_mini_mount(
            self.stage, "/World/envs/env_0/Robot", "/World/envs/env_0/ZED_X_Mini"
        )
        camera_cfg.prim_path = camera_cfg.prim_path.replace(
            "{ENV_REGEX_NS}", self.scene.env_regex_ns
        )
        self.camera = Camera(camera_cfg)
        self.sim.reset()
        from isaaclab_physx.physics import PhysxManager

        self.robot_contact_view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
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
        for tick in range(count):
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
            gravity = self.robot.data.gravity_compensation_forces.torch[
                :, self.robot.num_base_dofs :
            ][:, driven]
            self.robot.set_joint_effort_target_index(target=gravity, joint_ids=driven)
            self.scene.write_data_to_sim()
            self.sim.step(render=render and tick == count - 1)
            self.scene.update(DT)
            for sensor in self.block_contacts:
                sensor.update(DT)
            self.physics_step += 1
            self.time = self.physics_step * DT
        if render:
            self.capture()

    def capture(self) -> None:
        self.sim.forward()
        self.sim.render_context.update_transforms(self.sim.get_physics_step_count())
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
        sensor = self.block_contacts[block_index]
        block = self.blocks[block_index]
        matrix = sensor.data.force_matrix_w.torch[0, 0]
        robot_forces = matrix[: len(self.contact_robot_paths)]
        return {
            "block_pose_w": block.data.root_pose_w.torch[0].tolist(),
            "block_velocity_w": block.data.root_vel_w.torch[0].tolist(),
            "robot_contact_force_n": float(robot_forces.norm(dim=-1).sum()),
            "robot_contact_forces_n": robot_forces.tolist(),
            "block_contact_forces_n": matrix[len(self.contact_robot_paths) :].tolist(),
            # PhysX GPU filters cannot target the static table. Retain the residual
            # as environmental contact, with geometry establishing table support.
            "nonrobot_contact_force_n": float(
                (sensor.data.net_forces_w.torch[0, 0] - matrix.sum(dim=0)).norm()
            ),
        }

    def robot_diagnostics(self) -> dict:
        """Control/contact evidence; never supplied as hidden task decisions."""
        import warp as wp

        forces = wp.to_torch(self.robot_contact_view.get_net_contact_forces(dt=DT)).reshape(-1, 3)
        return {
            "joint_position": self.robot.data.joint_pos.torch[0].tolist(),
            "applied_joint_effort": self.robot.data.applied_torque.torch[0].tolist(),
            "robot_contact_paths": self.contact_robot_paths,
            "robot_contact_forces_n": forces.tolist(),
        }
