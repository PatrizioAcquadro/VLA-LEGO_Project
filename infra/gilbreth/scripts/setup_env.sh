#!/usr/bin/env bash
# Run after loading a supported Python module; never changes the source checkout.
set -euo pipefail
: "${VLA_ENV_DIR:?Set VLA_ENV_DIR to the desired virtual environment path}"
task_repo=$(cd "$(dirname "$0")/../../.." && pwd)
task_python=${VLA_PYTHON:-python3}
if [[ ! -x "$VLA_ENV_DIR/bin/python" ]]; then
    "$task_python" -m venv "$VLA_ENV_DIR"
fi
"$VLA_ENV_DIR/bin/python" -m pip install -e "$task_repo[sim,vlm,tracking]"
echo "Activate with: source $VLA_ENV_DIR/bin/activate"
