#!/usr/bin/env bash
# Launch the project's locked NVIDIA Isaac Lab runtime with Alex assets.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="${ISAAC_NATIVE_RUNTIME:-$HOME/.local/share/isaac-steel-rtx}"
alex="${ALEX_ASSETS_ROOT:-$root/../Alex}"
python="${VLA_ISAAC_PYTHON:-$runtime/.venv/bin/python}"
if [[ ! -x "$python" ]]; then
    printf 'Install the locked Isaac Lab runtime with the isaacsim extra (see the Isaac operations wiki).\n' >&2
    exit 2
fi
export PYTHONPATH="$root/src:$alex/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-$runtime/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$runtime/cache}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
exec "$python" "$@"
