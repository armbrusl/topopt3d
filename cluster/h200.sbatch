#!/bin/bash
#SBATCH --job-name=topopt3d
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=topopt3d-%j.out
set -euo pipefail

# jNO ships as a pinned submodule at vendor/jNO (branch feat/topopt-3d). The tetrahedral patch
# filter, the 3-D cell_angles and integrate(solver=) are not on jNO main. See README.
export JNO_DIR="${JNO_DIR:-${SLURM_SUBMIT_DIR}/vendor/jNO}"   # the pinned submodule

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # 141 GB is plenty; do not grab it all up front

# ---- what to run -----------------------------------------------------------------------------
# cuDSS is the backend to try FIRST here: it is not subject to the PARDISO blocker, and this
# problem is its ideal case (sparsity fixed across every iteration -> cached symbolic plan).
# Fall back to SOLVER="" (the default backend) if cuDSS is unavailable or misbehaves.
export SOLVER="${SOLVER:-cudss}"   # untested here; SOLVER="" is the safe fallback
export HMESH="${HMESH:-0.10}"       # 0.10 -> ~65k tets; go finer once a backend is proven
export PSTAR="${PSTAR:-0}"          # 0 = PROBE. Read the settled P, then rerun with ~0.76*P
export BUDGET="${BUDGET:-3000}"
export PENALP=0                     # SIMP continuation NaNs with the patch filter -- see README
export MOVING="${MOVING:-1}"
export TAG="${TAG:-h200}"

cd "${SLURM_SUBMIT_DIR}"
pixi run --manifest-path "$JNO_DIR/pyproject.toml" -e fem python scripts/run.py
