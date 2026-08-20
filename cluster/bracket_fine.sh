#!/bin/bash
# A long, fine rope-bracket run for another machine. Two phases, because one setting cannot be
# carried over from a coarser mesh:
#
#   1. PROBE   -- a short run with no barrier, to MEASURE the natural perimeter at this resolution.
#   2. DESIGN  -- the real run, with the barrier target set to a fraction of what the probe measured.
#
# The perimeter functional (Haber, Jog & Bendsoe, *Struct. Optim.* 11 (1996) 1-12) sums the density
# jump across every interior facet weighted by its AREA in 3-D, and its natural value GROWS with
# refinement -- measured 51 at h = 0.15 and 65 at h = 0.10 on this same bracket, because a finer mesh
# can resolve finer features. Copying a coarse mesh's PSTAR onto a fine one therefore does not hold
# the feature size fixed, it over-constrains. Hence the probe.
#
# Everything checkpoints every chunk to runs/topo3d_<TAG>.npz, so a wall-clock stop, a queue kill or
# a crash still leaves a usable design and its whole history.
#
#   bash cluster/bracket_fine.sh                 # defaults below
#   HMESH=0.05 HOURS=10 bash cluster/bracket_fine.sh
set -euo pipefail
cd "$(dirname "$0")/.."

HMESH="${HMESH:-0.05}"          # ~450k tets. Drop to 0.06 (~260k) or 0.07 (~165k) if memory binds.
ITERS="${ITERS:-140}"           # 140 is enough: on this problem C was flat from ~120 at h = 0.10
HOURS="${HOURS:-10}"
PSTAR_FRAC="${PSTAR_FRAC:-0.86}"   # target / natural, the ratio used at h = 0.15 and h = 0.10
PROBE_ITERS="${PROBE_ITERS:-15}"
CHUNK="${CHUNK:-10}"
TAG="${TAG:-bracket_fine}"
SOLVER="${SOLVER:-custom}"      # MKL PARDISO through scripts/pardiso_solver.py; see README
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

BUDGET=$(python3 -c "print(int($HOURS * 3600))")
RUN="pixi run --manifest-path vendor/jNO/pyproject.toml -e fem python -u"
COMMON="HMESH=$HMESH WASHER=0.18 EYE=0.20 MOVE_FRAC=0.5 VOLFRAC=0.30 BETA_H=16 BETA_AT=0.8 SOLVER=$SOLVER"

echo "=============================================================================="
echo " bracket_fine   h=$HMESH   iters=$ITERS   budget=${HOURS}h   solver=$SOLVER"
echo " expect roughly 56.3/h^3 tets: $(python3 -c "print(f'{56.3/($HMESH**3):,.0f}')")"
echo "=============================================================================="

# ---- 1. probe: no barrier, measure the natural perimeter area ---------------------------------
echo "[1/2] probing the natural perimeter at h=$HMESH ..."
env $COMMON PSTAR=0 ITERS=$PROBE_ITERS CHUNK=$PROBE_ITERS BUDGET=$BUDGET TAG=${TAG}_probe \
    $RUN scripts/bracket.py 2>&1 | tee "runs/${TAG}_probe.log" | grep -E "BRACKET|tets=|load |^  it " || true

PNAT=$(grep -oE "P= *[0-9.]+" "runs/${TAG}_probe.log" | tail -1 | grep -oE "[0-9.]+")
if [ -z "${PNAT:-}" ]; then
    echo "probe produced no perimeter -- refusing to guess a target. See runs/${TAG}_probe.log" >&2
    exit 1
fi
PSTAR=$(python3 -c "print(f'{$PNAT * $PSTAR_FRAC:.1f}')")
echo "[1/2] natural perimeter area = $PNAT  ->  PSTAR = $PSTAR  (x $PSTAR_FRAC)"

# ---- 2. the design run -------------------------------------------------------------------------
echo "[2/2] design run: $ITERS iterations, ${HOURS}h ceiling, checkpointing every $CHUNK ..."
env $COMMON PSTAR=$PSTAR ITERS=$ITERS CHUNK=$CHUNK BUDGET=$BUDGET TAG=$TAG \
    $RUN scripts/bracket.py 2>&1 | tee "runs/${TAG}.log"

echo
echo "done -> runs/topo3d_${TAG}.npz"
echo "visualise with:"
echo "  LX=4 LY=3 LZ=1.5 HMESH=$HMESH VOLFRAC=0.30 RUN=topo3d_${TAG}.npz TAG=$TAG FRAME_STRIDE=2 \\"
echo "      $RUN scripts/export_console.py"
echo
echo "NOTE ON SIZE. Measured on an i5-13600K, 250 iterations: 16,691 tets took 9.0 min and 52,967"
echo "took 25.9 min -- 2.16 and 6.22 s/iteration. A LINEAR extrapolation says h=0.05 (~450k tets)"
echo "would be ~50 s/iteration, i.e. ~2 h for 140. Treat that as a FLOOR, not an estimate: a sparse"
echo "direct factorisation on a 3-D mesh fills in faster than linearly, and the h=0.10 run already"
echo "came in 60 % above the linear prediction. Memory binds before wall clock does -- at ~285k DOFs"
echo "the factor is several GB. If it thrashes or OOMs, drop to h=0.07 (~165k tets) or switch to"
echo "SOLVER=cudss on a GPU node, which is the case cuDSS is built for (the sparsity pattern is"
echo "constant across every iteration, so its symbolic plan is computed once)."
