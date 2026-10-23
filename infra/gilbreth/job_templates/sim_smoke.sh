#!/usr/bin/env bash
#SBATCH --job-name=sim_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=euge
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${VLA_ENV_DIR:?Set VLA_ENV_DIR to the prepared Python environment}"
source "$VLA_ENV_DIR/bin/activate"
export HF_HOME=${HF_HOME:-${VLA_SCRATCH_ROOT:-/scratch/gilbreth/$USER/vla-lego}/cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export MUJOCO_GL=egl
python -m sim.demo --output-dir "outputs/sim-${SLURM_JOB_ID}"
