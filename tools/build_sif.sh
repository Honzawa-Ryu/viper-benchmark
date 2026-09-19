#!/bin/bash
#SBATCH --job-name=build_sif
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=logs/build-sif-log_%j.out
#SBATCH --error=logs/build-sif-log_%j.out

set -uo pipefail

# Resolve absolute path of the project root
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Building Apptainer image on compute node..."
echo "Project Root: ${PROJECT_ROOT}"
echo "=========================================="

# --fakeroot: no root on this cluster, relies on /etc/subuid /etc/subgid
# (already configured for this user). No --nv/GPU needed: the build itself
# only does apt-get installs + uv install, no CUDA compute.
if apptainer build --fakeroot "${PROJECT_ROOT}/env/env.sif" "${PROJECT_ROOT}/env/env.def"; then
    echo "=========================================="
    echo "apptainer build completed successfully."
    echo "=========================================="
else
    exit_code=$?
    echo "==========================================" >&2
    echo "❌ apptainer build failed (exit code: ${exit_code})." >&2
    echo "==========================================" >&2
    exit "${exit_code}"
fi
