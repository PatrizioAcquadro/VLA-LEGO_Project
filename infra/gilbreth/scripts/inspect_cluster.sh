#!/usr/bin/env bash
# Read-only cluster inventory; run on Gilbreth before selecting resources.
set -euo pipefail
sinfo -o "%P %a %l %D %c %m %G"
sacctmgr show qos format=Name,MaxWall,MaxTRESPU,MaxJobsPU
module avail cuda 2>&1
module avail python 2>&1
df -h "${VLA_SCRATCH_ROOT:-$PWD}"
