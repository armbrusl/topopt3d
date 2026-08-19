#!/bin/bash
#SBATCH --job-name=topopt3d
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=topopt3d-%j.out
set -euo pipefail

# jNO must be the feat/topopt-3d branch -- the tetrahedral patch filter, the 3-D cell_angles and
# integrate(solver=) are not on main. See README.
export JNO_DIR="${JNO_DIR:?set JNO_DIR to the jNO checkout}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # 141 GB is plenty; do not grab it all up front

# ---- what to run -----------------------------------------------------------------------------
# cuDSS is the backend to try FIRST here: it is not subject to the PARDISO blocker, and this
# problem is its ideal case (sparsity fixed across every iteration -> cached symbolic plan).
# Fall back to SOLVER="" (the default backend) if cuDSS is unavailable or misbehaves.
export SOLVER="${SOLVER:-cudss}"
export HMESH="${HMESH:-0.10}"       # 0.10 -> ~65k tets; go finer once a backend is proven
export PSTAR="${PSTAR:-0}"          # 0 = PROBE. Read the settled P, then rerun with ~0.76*P
export BUDGET="${BUDGET:-3000}"
export PENALP=0                     # SIMP continuation NaNs with the patch filter -- see README
export MOVING="${MOVING:-1}"
export TAG="${TAG:-h200}"

cd "$JNO_DIR"
pixi run -e fem python "${SLURM_SUBMIT_DIR}/scripts/run.py"
