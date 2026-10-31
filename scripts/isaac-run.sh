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
exec "$ISAAC_SIM_DIR/python.sh" "$@"
