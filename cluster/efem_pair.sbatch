#!/bin/bash
#SBATCH --job-name=efem-pair
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --output=efem-pair-%j.out
# NOT `set -e`. A queue job must never let one stage take the others down with it: the optimisation
# is the expensive part, and a reanalysis that runs out of memory must not destroy the design that
# preceded it. Every stage below reports its own exit status instead.
set -uo pipefail

# The Lagrange-vs-E-FEM comparison at a resolution this workstation cannot reach.
#
# WHY THE CLUSTER -- and why much less of it than you would think. E-FEM at h=0.25 used to peak at
# 36 GB and was OOM-killed once on a 62 GB box, which is what this job was originally sized for. That
# turned out to be a chunk-policy bug, not the method: jNO floored the element chunk at 8192 CELLS
# while the resource it bounds is BYTES, so a 48-DOF enriched element blew an 8 MiB budget by three
# orders of magnitude, and on a mesh smaller than the floor chunking was skipped outright. Fixed in
# `fem_utils.cell_chunk`. Measured peaks after the fix:
#
#     h=0.25   4,082 tets    36.2 GB -> 13.4 GB
#     h=0.20   7,615 tets              16.8 GB
#     h=0.15  16,691 tets              20.6 GB      <- now runs on the workstation
#
# So h=0.15 no longer needs a queue slot. This job is for h=0.10 (~65k tets) and finer.
#
# ONE GPU, not eight. Nothing here shards across devices -- jNO logs "mesh sharding" but we have not
# tested it, and requesting hardware we cannot drive only lengthens the queue wait. `FEMCHUNK` is
# likewise left at 0 on purpose: on a GPU the saturation floor is CORRECT (there are cores to starve)
# and 65k cover cells chunk to ~7 GB, which an H200 absorbs without help. Forcing the workstation's
# 128-cell chunk here would buy nothing and cost the measured ~2x.
#
# WHY IT MATTERS. Our claim about the two formulations rests on meshes where the extracted solid is
# almost entirely boundary -- 99.3 % of its nodes at h=0.25. A resolution where that is not true is
# the difference between "measured" and "measured once, coarsely".
#
# WHAT TO EXPECT. At h=0.25 the corrected pair gave: Lagrange reported 4.8591 for a body whose
# refined solve is >= 15.6882, E-FEM reported 6.0357 for one whose refined solve is >= 5.9528.
# COMPARE AT MATCHED REFINEMENT LEVEL, not headline to headline: a coarser reanalysis produces a
# weaker lower bound, so a less-refined case looks better purely for being less refined.
#
# RUNS WITH OR WITHOUT A SCHEDULER. The #SBATCH lines are comments to a plain shell, and every
# SLURM variable below has a fallback, so `bash cluster/efem_pair.sbatch` is a first-class way to
# run it. On a bare machine BUDGET is the only thing that stops it -- there is no wall clock.
#
#   sbatch cluster/efem_pair.sbatch                        # scheduler
#   bash   cluster/efem_pair.sbatch                        # bare machine
#   HMESH=0.12 bash cluster/efem_pair.sbatch               # cheaper first pass

export JNO_DIR="${JNO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/vendor/jNO}"
# NOT pinned to the core count. Measured on a 20-core workstation, PARDISO factorising a 13k-DOF
# cover problem: OMP_NUM_THREADS unset (MKL's own default) ran 50 iterations at 3.7 s/iter using
# ~6 cores; forcing OMP_NUM_THREADS=1 cost 3.3x (12.3 s/iter); forcing it to 4 or 20 cost another
# 2x on top of that (24.2-24.8 s/iter), with core USAGE flat at ~4 regardless of the setting --
# XLA's CPU thread pool and MKL's own are oversubscribing each other, not sharing work. A factorised
# system this size (1.5M nonzeros at h=0.25) does not have enough parallel work to profit from more
# threads, and forcing a count only adds contention. Leave it unset; do not extrapolate the workstation
# number to a different core count without re-measuring there.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# DEVICE=cpu by DEFAULT, and that is not a fallback -- it is the only path either formulation has
# been validated on. Measured: SPACE=cover dies on the GPU with
#   ValueError: Incompatible types for broadcasting: input=float64[67968], requested=float64[271872]
# where 271872 = 4 x 67968 and 4 is exactly the cover multiplier (1 value + 3 covers). It is not
# simply "cover on GPU" -- h=0.35 (1,818 tets) runs fine on one -- so it is size- or path-dependent
# and not yet diagnosed. Lagrange on GPU is fine; only the enriched space is affected.
#
# The CPU path costs nothing here: PARDISO is on the host anyway, and the chunk policy now derives
# its budget from host RAM, so a 1 TB machine gets a ~1.5 GB chunk and bounded memory for free.
export DEVICE="${DEVICE:-cpu}"
if [ "$DEVICE" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
    export JAX_PLATFORMS=cpu
else
    # One visible device on purpose: nothing shards, and jNO's mesh-sharding path keys off what JAX
    # can see, so 8 of them would silently pick an execution plan we have never tested.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

# NOTE: SOLVER here does NOTHING for bracket.py. It hardcodes `pardiso_lu` from
# scripts/pardiso_solver.py -- one `pure_callback` straight into pypardiso -- because jNO's own
# `lu(backend="pardiso")` aborts the process (glibc double free) on this exact operator, a P0
# density behind `rho.constrain(d.patch_filter())`; see that file's docstring. `SOLVER` is read
# only by scripts/run.py, a different script this job does not call. Left exported (harmless) in
# case a future problem script does read it, but do not expect it to change bracket.py's behaviour.
#
# CONSEQUENCE: the linear SOLVE always runs on the host via pypardiso, whatever GPU is attached.
# The GPU is used for assembly/AD only, automatically, whenever CUDA_VISIBLE_DEVICES makes one
# visible -- there is no separate switch for that either. Routing the solve itself through cuDSS
# would need bracket.py changed to call `fem.integrate(solver=...)` with a cuDSS backend instead of
# the hardcoded `pardiso_lu`, and that is UNTESTED against the same crash this file exists to avoid
# -- cuDSS may or may not hit it. cluster/calibrate.sh measures what the CURRENT code gets (host
# solve + GPU assembly) before anyone spends time wiring in an alternative solve path.
export SOLVER="${SOLVER:-custom}"

export HMESH="${HMESH:-0.10}"
export WASHER="${WASHER:-0.25}"     # reproduces the 1,084-tet passive set of the h=0.25 pair
export EYE="${EYE:-0.30}"           # ditto -- these two ARE the problem definition, not tuning
# Heaviside projection is now safe to turn on (jno.optimizers.heaviside_continuation actually
# reaches the gradient; the earlier hand-rolled ramp did not). Off by default here because the
# comparison this job runs is the reanalysis baseline, not the binary-design one -- turning it on
# changes what is being measured, not just how long it takes. BETA_H=16 BETA_AT=0.7 with a longer
# ITERS (750 measured C1 at h=0.25; scale up for h=0.10) reproduces the binary result.
export BETA_H="${BETA_H:-0}"
export PSTAR="${PSTAR:-0}"          # no perimeter barrier, so the two runs differ in ONE thing
export ITERS="${ITERS:-250}"
# BUDGET IS A HARD STOP, AND ITS DEFAULT IS 2 HOURS. `bracket.py` loops on
# `while time.time() - t0 < BUDGET and it < ITERS`, so an unset BUDGET truncates a long run and
# SAVES ANYWAY -- the .npz is indistinguishable from a finished one. Leaving it unset here would
# have stopped E-FEM at h=0.10 near iteration 80 and burned the rest of the slot for nothing.
# Sized to the SLURM limit with an hour of headroom for the reanalyses.
export BUDGET="${BUDGET:-72000}"
# Reanalysis refinement. Level k subdivides the extracted surface 4^k times, so cost is savage at a
# fine design: at h=0.10 level 2 is a ~1M-tet refill and level 3 will not fit. Start at 1 and raise
# it deliberately -- the bound is one-sided either way, just looser.
export REFINE="${REFINE:-2}"
# FEMCHUNK overrides jNO's element-chunk policy by hand. Unset is right -- the policy is fixed now
# and derives the chunk from the machine. Set it only to reproduce a pre-fix measurement.
export FEMCHUNK="${FEMCHUNK:-0}"
export CHUNK="${CHUNK:-50}"         # CHUNK=10 stalls this problem -- see README

cd "${SLURM_SUBMIT_DIR:-$PWD}"
JRUN=(pixi run --manifest-path "$JNO_DIR/pyproject.toml" -e fem python -u)
mkdir -p runs
# The reanalysis writes a .msh per solve. On a fine design those are large and purely intermediate,
# so keep them off the shared filesystem when the node offers somewhere local.
export SCRATCH="${SCRATCH:-${SLURM_TMPDIR:-${TMPDIR:-$PWD/runs}}}"
TAGS=""

# REANALYSE_ONLY=1 skips the optimisation and deep-refines designs that already exist. That is the
# cheapest way to spend a big machine on our actual weak point: every "truth >= X" we report is a
# one-sided bound that was still climbing at the finest level we could afford. Raising REFINE on an
# EXISTING design tightens it without re-optimising anything.
#
#   REANALYSE_ONLY=1 TAGLIST=h25c_Lagrange,h25c_cover REFINE=4 sbatch cluster/efem_pair.sbatch
for SPACE_ in Lagrange cover; do
    TAG_="h${HMESH/0./}_${SPACE_}"
    TAGS="${TAGS:+$TAGS,}${TAG_}"
done
[ -n "${TAGLIST:-}" ] && TAGS="$TAGLIST"

if [ "${REANALYSE_ONLY:-0}" != "1" ]; then
  for SPACE_ in Lagrange cover; do
    TAG_="h${HMESH/0./}_${SPACE_}"
    if [ -f "runs/topo3d_${TAG_}.npz" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "!! runs/topo3d_${TAG_}.npz exists -- refusing to overwrite. FORCE=1 to replace."
        continue
    fi
    echo "=== ${TAG_} optimisation ($(date -Is)) ==========================================="
    # tee, not >: the job log keeps the live output AND runs/<tag>.log gets the copy the report
    # needs. Without that file `cluster_report.py` cannot tell a truncated run from a finished one,
    # which it flagged on the first smoke run.
    SPACE="${SPACE_}" TAG="${TAG_}" "${JRUN[@]}" scripts/bracket.py 2>&1 | tee "runs/${TAG_}.log"
    echo "--- ${TAG_} optimisation exit=${PIPESTATUS[0]} ($(date -Is))"
  done
fi

# Reanalysis runs in a SECOND pass, after both designs are safely on disk. If it dies -- and it is
# the memory-hungry stage -- the expensive optimisations survive and it can be rerun on its own.
for TAG_ in ${TAGS//,/ }; do
    [ -f "runs/topo3d_${TAG_}.npz" ] || { echo "skip reanalysis ${TAG_}: no design"; continue; }
    echo "=== ${TAG_} reanalysis, REFINE=${REFINE} ($(date -Is)) ==========================="
    RUN="topo3d_${TAG_}.npz" "${JRUN[@]}" scripts/reanalysis_boundary.py > "runs/reb_${TAG_}.log" 2>&1
    echo "--- ${TAG_} reanalysis exit=$? ($(date -Is))"
    tail -25 "runs/reb_${TAG_}.log"
done

# One report, checked for the failures that are otherwise silent: a run truncated by BUDGET, a blank
# whose volume moved, a support set that drifted, and the acceptance test itself.
echo "=== report ($(date -Is)) ===================================================="
REPORT="cluster_report_${SLURM_JOB_ID:-local}.json" "${JRUN[@]}" scripts/cluster_report.py ${TAGS//,/ }
echo
echo "Bring back: runs/topo3d_h*.npz, runs/h*_*.log, runs/reb_h*_*.log, runs/cluster_report_*.json"
echo "The .npz files are self-contained -- every figure and every reanalysis can be redone from them."
