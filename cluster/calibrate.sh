#!/bin/bash
#SBATCH --job-name=efem-calibrate
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=efem-calibrate-%j.out
set -uo pipefail

# Answers ONE question cheaply before spending a real slot on cluster/efem_pair.sbatch: on THIS
# machine, does anything actually speed up an E-FEM iteration?
#
# WHY THIS EXISTS. Measured on a 20-core workstation, PARDISO factorising a 13k-DOF cover problem:
# OMP_NUM_THREADS unset ran fastest (3.7 s/iter, ~6 cores used); forcing it to 1 cost 3.3x; forcing
# it to 4 or 20 cost ANOTHER 2x on top of that, with core usage flat at ~4 regardless of the
# setting -- XLA's own CPU pool and MKL's are contending, not sharing. That result does not
# necessarily transfer to a different core count or a different problem size, and it is cheap
# enough to just re-measure here rather than guess. A GPU-assembly run on a small consumer card
# also hit CUDA_ERROR_OUT_OF_MEMORY (RESOURCE_EXHAUSTED, too many live command-buffer graphs) --
# untested on real datacenter memory, so this checks that too.
#
# WHAT IT DOES NOT DO. It does not touch the linear SOLVE, which is where the larger half of an
# iteration goes (measured split ~43-57 % solve/assembly on the small pair; shifts toward assembly
# as the mesh grows). scripts/bracket.py hardcodes the solve through scripts/pardiso_solver.py --
# ONE pypardiso call on the host, unconditionally -- because jNO's own PARDISO backend aborts the
# process on this operator (see that file's docstring). There is currently no cuDSS lever for the
# solve without changing bracket.py's code, not just an env var, and doing that is untested against
# the same crash the hardcoded path exists to dodge. This script does not attempt it; it measures
# what the code as it stands gets from threading and from GPU-assisted assembly, which is real and
# free to test today.
#
#   bash cluster/calibrate.sh                          # HMESH=0.20, quick
#   HMESH=0.15 bash cluster/calibrate.sh                # closer to a real run, slower to calibrate

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export JNO_DIR="${JNO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/vendor/jNO}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
JRUN=(pixi run --manifest-path "$JNO_DIR/pyproject.toml" -e fem python -u)
mkdir -p runs

HMESH="${HMESH:-0.20}"
ITERS="${CAL_ITERS:-20}"
COMMON="HMESH=$HMESH WASHER=0.25 EYE=0.30 PSTAR=0 BETA_H=0 SPACE=cover ITERS=$ITERS CHUNK=$ITERS"

s_per_iter() {  # $1 = log path -> seconds/iteration, from the elapsed time at the final chunk.
    # bracket.py prints "(NNs)" at the end of every "  it" line. Includes mesh setup diluted over
    # ITERS iterations (a few seconds on a few dozen), so treat this as comparative, not absolute.
    local t
    t=$(grep -E '^  it' "$1" | tail -1 | grep -oE '\([0-9.]+s\)' | tr -d '()s')
    [ -n "$t" ] && awk -v t="$t" -v n="$ITERS" 'BEGIN{printf "%.2f", t/n}' || echo "FAILED"
}

echo "=== CPU: OMP_NUM_THREADS sweep, SPACE=cover, HMESH=$HMESH, $ITERS iterations each ==="
for T in unset 1 4 16 "${SLURM_CPUS_PER_TASK:-32}"; do
    TAG="cal_omp_${T}"
    if [ "$T" = "unset" ]; then
        env -u OMP_NUM_THREADS -u MKL_NUM_THREADS CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
            $COMMON TAG="$TAG" "${JRUN[@]}" scripts/bracket.py > "runs/${TAG}.log" 2>&1
    else
        env OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
            $COMMON TAG="$TAG" "${JRUN[@]}" scripts/bracket.py > "runs/${TAG}.log" 2>&1
    fi
    printf "  OMP_NUM_THREADS=%-6s  %s s/iter\n" "$T" "$(s_per_iter "runs/${TAG}.log")"
done

echo
echo "=== GPU-assisted assembly (SOLVER has no effect on bracket.py -- see the header) ==="
echo "    the linear solve still runs on the host via pypardiso either way; only assembly/AD move"
TAG="cal_gpu"
env CUDA_VISIBLE_DEVICES=0 $COMMON TAG="$TAG" "${JRUN[@]}" scripts/bracket.py > "runs/${TAG}.log" 2>&1
if grep -q RESOURCE_EXHAUSTED "runs/${TAG}.log"; then
    echo "  RESOURCE_EXHAUSTED -- see runs/${TAG}.log. On an 8 GB consumer card this was a CUDA"
    echo "  command-buffer allocation failure (29 live graphs), not the AD tangent itself; may not"
    echo "  recur on H200 memory. If it does, retry with XLA_FLAGS=--xla_gpu_enable_command_buffer="
else
    printf "  GPU-assisted assembly    %s s/iter\n" "$(s_per_iter "runs/${TAG}.log")"
fi
echo
echo "  No SOLVER=cudss row: bracket.py has no code path to it today (see the header comment)."
echo "  If GPU assembly above beats the best CPU row by a wide margin, wiring in a GPU solve"
echo "  (jno.fem(..., solver=lu(backend=\"cudss\")) in place of the hardcoded pardiso_lu import)"
echo "  is the next thing worth trying -- untested, and possibly hits the same crash pardiso_lu"
echo "  was written to avoid, so budget time to fall back if it does not work."

echo
echo "Compare the s/iter numbers above. Set the winning thread count explicitly in the real job"
echo "(SOLVER does not need setting -- it has no effect on bracket.py, see above):"
echo "  OMP_NUM_THREADS=<best> sbatch cluster/efem_pair.sbatch"
