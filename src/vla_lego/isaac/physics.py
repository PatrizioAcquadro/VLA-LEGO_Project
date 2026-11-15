"""Select the physics engine for native VLA-LEGO runs."""

import os

PHYSX = "isaacsim_physx"
STEEL = "newton_steel_native"


def physics_profile() -> str:
    profile = os.environ.get("ISAAC_PHYSICS_PROFILE") or STEEL
    if profile == "newton_steel":
        return STEEL
    if profile not in (STEEL, PHYSX):
        raise ValueError(f"Unknown Isaac physics profile: {profile}")
    return profile


def default_solver_iterations(profile: str) -> tuple[int, int]:
    return (16, 0) if profile == STEEL else (8, 4)


def steel_physics():
    from isaaclab_newton.physics import NewtonCfg, SteelSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonShapeCfg

    return NewtonCfg(
        solver_cfg=SteelSolverCfg(
            iterations=default_solver_iterations(STEEL)[0],
            enable_joint_limits=True,
            enable_joint_velocity_limits=True,
            articulation_constraint_max=512,
        ),
        default_shape_cfg=NewtonShapeCfg(gap=0.02),
        use_cuda_graph=True,
    )


def launch_runtime():
    from isaaclab.app import launch_simulation
    from isaaclab_physx.physics import PhysxCfg

    physics = steel_physics() if physics_profile() == STEEL else PhysxCfg()
    runtime = launch_simulation(
        physics,
        {"headless": True, "device": "cuda:0", "enable_cameras": True, "require_kit": True},
    )
    runtime.__enter__()
    return runtime


def close_runtime(runtime, code):
    # Propagate the caller's status to Kit, which owns process shutdown.
    runtime.__exit__(SystemExit, SystemExit(code), None)
