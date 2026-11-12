#!/usr/bin/env bash
# Launch native Isaac with shared Alex assets and this task's source.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${ISAAC_SIM_DIR:?Set ISAAC_SIM_DIR to the Isaac Sim installation}"
: "${ISAAC_LAB_DIR:?Set ISAAC_LAB_DIR to the compatible Isaac Lab checkout}"
: "${ALEX_ASSETS_ROOT:?Set ALEX_ASSETS_ROOT to the shared Alex repository}"
export PYTHONPATH="$project_root/src:$ALEX_ASSETS_ROOT/src:$ISAAC_LAB_DIR/source/isaaclab:$ISAAC_LAB_DIR/source/isaaclab_physx:$ISAAC_LAB_DIR/source/isaaclab_ov:$ISAAC_LAB_DIR/source/isaaclab_ovphysx:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONEXE="${VLA_ISAAC_PYTHON:-$ISAAC_SIM_DIR/kit/python/bin/python3}"
export ISAAC_PHYSICS_PROFILE="${ISAAC_PHYSICS_PROFILE:-newton_steel}"
if [[ "$ISAAC_PHYSICS_PROFILE" == newton_steel ]]; then
    export ISAAC_PHYSICS_PROFILE_ROOT="${ISAAC_PHYSICS_PROFILE_ROOT:-$project_root/../AlexDoor-XAS}"
    export NEWTON_SITE="${NEWTON_SITE:-$ISAAC_PHYSICS_PROFILE_ROOT/outputs/generated/isaac-steel-runtime/site}"
    export PYTHONPATH="$ISAAC_PHYSICS_PROFILE_ROOT/src:$ISAAC_LAB_DIR/source/isaaclab_newton:$PYTHONPATH"
    source "$ISAAC_PHYSICS_PROFILE_ROOT/src/isaac_physics_profile/launch.sh"
    "$ISAAC_SIM_DIR/python.sh" "$@"
elif [[ "$ISAAC_PHYSICS_PROFILE" == isaacsim_physx ]]; then
    exec "$ISAAC_SIM_DIR/python.sh" "$@"
else
    printf 'Unknown Isaac physics profile: %s\n' "$ISAAC_PHYSICS_PROFILE" >&2
    exit 2
fi
