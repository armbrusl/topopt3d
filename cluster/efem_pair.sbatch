#!/bin/bash
#SBATCH --job-name=efem-pair
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=efem-pair-%j.out
set -euo pipefail

# The Lagrange-vs-E-FEM comparison at a resolution this workstation cannot reach.
#
# WHY THE CLUSTER. E-FEM at h=0.25 (4,082 tets) peaked at 36 GB and was OOM-killed once on a 62 GB
# box. h=0.15 is 16,691 tets -- 4.1x -- so even at linear scaling it wants well over 100 GB, and the
# enriched space is not linear in the tet count. `--mem=200G` is sized from that, not guessed down.
#
# WHY IT MATTERS. Everything we currently claim about the two formulations rests on h=0.25, where the
# extracted solid is 99.3 % boundary nodes (662 nodes, 3 interior) and NEITHER refinement series
# converges. A second resolution is the difference between "measured" and "measured once, coarsely".
#
# WHAT TO EXPECT. At h=0.25 the corrected pair gave: Lagrange reported 4.8591 for a body whose
# refined solve is >= 15.6882, E-FEM reported 6.0357 for one whose refined solve is >= 5.9528. The
# question this job answers is whether that survives refinement.
#
#   sbatch cluster/efem_pair.sbatch                 # HMESH=0.15, both formulations
#   HMESH=0.20 sbatch cluster/efem_pair.sbatch      # cheaper first pass if 0.15 will not fit

export JNO_DIR="${JNO_DIR:-${SLURM_SUBMIT_DIR}/vendor/jNO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# The linear solve is PARDISO on the host here (scripts/bracket.py imports pardiso_solver directly),
# so the GPU is requested for the assembly/AD side only. Drop --gres if the queue is tight.
export SOLVER="${SOLVER:-custom}"

export HMESH="${HMESH:-0.15}"
export WASHER="${WASHER:-0.25}"     # reproduces the 1,084-tet passive set of the h=0.25 pair
export EYE="${EYE:-0.30}"           # ditto -- these two ARE the problem definition, not tuning
export BETA_H="${BETA_H:-0}"        # no Heaviside: the ramp needs >250 iterations to re-converge
export PSTAR="${PSTAR:-0}"          # no perimeter barrier, so the two runs differ in ONE thing
export ITERS="${ITERS:-250}"
export CHUNK="${CHUNK:-50}"         # CHUNK=10 stalls this problem -- see README

cd "${SLURM_SUBMIT_DIR}"
JRUN=(pixi run --manifest-path "$JNO_DIR/pyproject.toml" -e fem python -u)

for SPACE_ in Lagrange cover; do
    TAG_="h${HMESH/0./}_${SPACE_}"
    echo "=== ${TAG_} ($(date -Is)) ==================================================="
    SPACE="${SPACE_}" TAG="${TAG_}" "${JRUN[@]}" scripts/bracket.py

    # The acceptance test. A Lagrange run MUST reproduce its own reported compliance under an
    # independent solve; if it does not, the supports moved and every number in the run is void.
    # This is the check whose absence cost us a whole set of results -- do not skip it.
    RUN="topo3d_${TAG_}.npz" "${JRUN[@]}" scripts/reanalysis_boundary.py \
        > "runs/reb_${TAG_}.log" 2>&1 || echo "reanalysis FAILED for ${TAG_}"
    grep -E "formulation|DISTORTION|AT LEAST|understates" "runs/reb_${TAG_}.log" || true
done

echo "=== done $(date -Is). Compare with: grep -E 'reported|P1 read|AT LEAST' runs/reb_h*_*.log"
