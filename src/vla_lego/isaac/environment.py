"""Native fixed-base Alex003 foundation, consuming the shared Alex package."""

from __future__ import annotations

import math
from pathlib import Path

from vla_lego.isaac.catalog import BlockSpec

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

    def __init__(self, cache_dir: Path):
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
        cfg.block = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Block",
            spawn=sim_utils.CuboidCfg(
                size=DIAGNOSTIC_BLOCK.dimensions_m,
                rigid_props=PhysxRigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=DIAGNOSTIC_BLOCK.mass_kg),
                activate_contact_sensors=True,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DIAGNOSTIC_BLOCK.rgb),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.15, TABLE_TOP + 0.08)),
        )
        cfg.zed_model, camera_cfg, _ = make_zed_x_mini_cfgs(
            "{ENV_REGEX_NS}/ZED_X_Mini", resolution="SVGA"
        )
        camera_cfg.data_types = ["rgb"]
        cfg.key_light = AssetBaseCfg(
            prim_path="/World/KeyLight", spawn=sim_utils.DistantLightCfg(intensity=1200)
        )
        cfg.replicate_physics = False
        self.scene = InteractiveScene(cfg)
        self.robot, self.block = self.scene["robot"], self.scene["block"]
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
        self.block_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_0/Block",
                filter_prim_paths_expr=self.contact_robot_paths,
                update_period=0.0,
                history_length=1,
            )
        )
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
        self.robot.set_joint_position_target_index(target=self.target)
        self.jaw_ids = [self.robot.data.joint_names.index(name) for name in JAW_JOINTS]
        self.time = 0.0

    def set_joint_targets(self, targets: dict[str, float]) -> None:
        """Named low-level targets, bounded by the shared robot's hard limits."""
        for name, value in targets.items():
            index = self.robot.data.joint_names.index(name)
            lower, upper = self.robot.data.joint_pos_limits.torch[0, index].tolist()
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Target outside joint limits: {name}={value}")
            if name.endswith("FOLLOWER"):
                raise ValueError("The shared mimic constraint owns follower targets.")
            self.target[0, index] = value
        self.robot.set_joint_position_target_index(target=self.target)

    def step(self, count: int = 1, *, render: bool = False) -> None:
        from ihmc_alex_isaaclab.robots.purdue_physics import ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS

        followers = {name for name, _ in ALEX_PURDUE_WSG32_MIMIC_JOINT_PAIRS}
        driven = [i for i, name in enumerate(self.robot.data.joint_names) if name not in followers]
        for tick in range(count):
            gravity = self.robot.data.gravity_compensation_forces.torch[
                :, self.robot.num_base_dofs :
            ][:, driven]
            self.robot.set_joint_effort_target_index(target=gravity, joint_ids=driven)
            self.scene.write_data_to_sim()
            self.sim.step(render=render and tick == count - 1)
            self.scene.update(DT)
            self.block_contact.update(DT)
            self.time += DT
        if render:
            self.sim.forward()
            self.sim.render_context.update_transforms(self.sim.get_physics_step_count())
            self.camera.update(DT * count, force_recompute=True)

    def observe(self) -> dict:
        """Only deployable sensor/joint signals, never block poses or teacher stages."""
        names = (*ARM_JOINTS, "NECK_Z", "NECK_Y", *JAW_JOINTS)
        indices = [self.robot.data.joint_names.index(name) for name in names]
        return {
            "timestamp_s": self.time,
            "head_rgb": self.camera.data.output["rgb"].torch[0, :, :, :3].cpu().numpy(),
            "joint_names": names,
            "joint_position": self.robot.data.joint_pos.torch[0, indices].detach().cpu().numpy(),
            "joint_velocity": self.robot.data.joint_vel.torch[0, indices].detach().cpu().numpy(),
        }

    def task_truth(self) -> dict:
        """Privileged diagnostic/evaluator channel, excluded from observe()."""
        matrix = self.block_contact.data.force_matrix_w.torch[0, 0]
        return {
            "block_pose_w": self.block.data.root_pose_w.torch[0].tolist(),
            "block_velocity_w": self.block.data.root_vel_w.torch[0].tolist(),
            "robot_contact_force_n": float(matrix.norm(dim=-1).sum()),
            # PhysX GPU filters cannot target the static table. Retain the residual
            # as environmental contact, with geometry establishing table support.
            "nonrobot_contact_force_n": float(
                (self.block_contact.data.net_forces_w.torch[0, 0] - matrix.sum(dim=0)).norm()
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
