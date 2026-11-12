"""Select the physics engine for native VLA-LEGO runs."""

import os

PHYSX = "isaacsim_physx"
STEEL = "newton_steel"


def physics_profile() -> str:
    profile = os.environ.get("ISAAC_PHYSICS_PROFILE") or STEEL
    if profile not in (STEEL, PHYSX):
        raise ValueError(f"Unknown Isaac physics profile: {profile}")
    return profile


def default_solver_iterations(profile: str) -> tuple[int, int]:
    return (16, 4) if profile == STEEL else (8, 4)
