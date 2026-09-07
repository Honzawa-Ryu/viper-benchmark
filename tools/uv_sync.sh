#!/bin/bash
#SBATCH --job-name=uv_sync
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --output=logs/uv-sync-log_%j.out
#SBATCH --error=logs/uv-sync-log_%j.out

set -uo pipefail

# Resolve absolute path of the project root
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Starting uv sync on compute node..."
echo "Project Root: ${PROJECT_ROOT}"
echo "=========================================="

# Execute uv sync inside the Apptainer container
# Note: 'cd' into PROJECT_ROOT inside the container to ensure .venv is created in the right place
# The exit code of `uv sync` is checked explicitly below: without this, a failing
# `uv sync` would still let the script reach the final echo and exit 0, making
# Slurm (and `make uv_sync`) report the job as COMPLETED even though the venv
# was never synced.
if apptainer exec --nv "${SIF_PATH}" bash -c "
    set -euo pipefail
    cd ${PROJECT_ROOT}
    uv sync
"; then
    echo "=========================================="
    echo "uv sync completed successfully."
    echo "=========================================="
else
    exit_code=$?
    echo "==========================================" >&2
    echo "❌ uv sync failed (exit code: ${exit_code})." >&2
    echo "==========================================" >&2
    exit "${exit_code}"
fi
