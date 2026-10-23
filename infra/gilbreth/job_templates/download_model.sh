#!/usr/bin/env bash
#SBATCH --job-name=download_model
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=euge
#SBATCH --partition=shared
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${VLA_ENV_DIR:?Set VLA_ENV_DIR to the prepared Python environment}"
source "$VLA_ENV_DIR/bin/activate"
export HF_HOME=${HF_HOME:-${VLA_SCRATCH_ROOT:-/scratch/gilbreth/$USER/vla-lego}/cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export MUJOCO_GL=egl
# Download only; this CPU job never loads or runs model weights.
python - <<'PYCODE'
import os
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen3.5-4B", cache_dir=os.environ["HF_HUB_CACHE"],
                         allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
print(path)
PYCODE
