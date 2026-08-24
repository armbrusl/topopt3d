"""A rope bracket: bolted flat by two bolts, pulled by a rope that does NOT pull in-plane.

The industrial companion to the torsion bar. Two things make it a real 3-D case rather than a
thick 2-D one:

* **The pull is off-axis.** A rope almost never pulls in the plane of its bracket, and an
  out-of-plane component means the plate is pried and twisted as well as bent -- there is no planar
  truss to extrude. The sideways component also breaks left/right symmetry, so the two bolts carry
  different loads and the structure is not mirror-symmetric either.
* **The blank is not a box.** A tapered hexagonal prism with two holes through it, so the optimiser
  works inside a shape with real boundaries and re-entrant material around the holes.

    PSTAR=0 python scripts/bracket.py      # probe: no barrier, measure the natural perimeter AREA
    PSTAR=<measured> python scripts/bracket.py

Method: Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403 -- eq. (17)-(19) patch
filter (3-D: edge fans), eq. (21)/(24) dihedral bounds, eq. (26)/(28) density-coupled element-volume
bounds, eq. (38)-(41) perimeter barrier. Heaviside projection after Wang, Lazarov & Sigmund (2011).
"""

import os
import resource
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pardiso_solver import pardiso_lu  # noqa: E402

inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
OUT = os.environ.get("OUTDIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"
)
os.makedirs(OUT, exist_ok=True)

LX, LY, LZ = 4.0, 3.0, 1.5  # width, height, thickness
SHOULDER, TOPL, TOPR = (
    1.0,
    1.4,
    2.6,
)  # where the taper starts, and the flat top's extent
BX1, BX2, BY, RB = 0.8, 3.2, 0.6, 0.35  # bolt centres and radius
H_MESH = float(os.environ.get("HMESH", "0.15"))
E0, NU, PENAL = 1.0, 0.3, 3.0
EMIN = float(os.environ.get("EMIN", "1e-4"))
LAM, MU = E0 * NU / ((1 + NU) * (1 - 2 * NU)), E0 / (2 * (1 + NU))
VOLFRAC = float(os.environ.get("VOLFRAC", "0.30"))
# The paper uses theta_min = 20 deg on TRIANGLE interior angles (ideal 60, so a third of ideal).
# In 3-D `cell_angles()` returns the six DIHEDRALS, whose ideal is 70.53 deg, and the same ratio
# would be 23.5. But gmsh hands us a mesh whose worst dihedral is already 15.23 deg, so any bound
# above that is violated from iteration 0 -- and their own Fig. 6c shows the constraint only
# marginally active at convergence, not violated at the start. 15 deg is the tightest bound this
# initial mesh satisfies (0.00% of cells below it, against 0.91% below 20 deg).
THETA_MIN, PN, TOL = np.deg2rad(float(os.environ.get("THETA_MIN_DEG", "15.0"))), 50.0, 1e-9
# Loosening the aspect bound is how you find out whether it, rather than MOVE_FRAC, is what
# limits node migration -- the two are easy to confuse because both show up as a pinned mesh.
ASPECT_MAX = float(os.environ.get("ASPECT_MAX", "8.0"))
MOVE_BOUND = float(os.environ.get("MOVE_FRAC", "1.0")) * H_MESH
RELOCATE = int(os.environ.get("RELOCATE", "1"))
# Read here, not at the point of use: the CONFIG banner below prints it, and a knob that the
# banner reports has to exist by the time the banner runs.
SPACE = os.environ.get("SPACE", "Lagrange")
WASHER = float(
    os.environ.get("WASHER", "0.45")
)  # non-design ring around each bolt hole
EYE = float(os.environ.get("EYE", "0.35"))  # non-design depth under the loaded top face
PSTAR = float(os.environ.get("PSTAR", "0"))
BETA, GAMMA = 1e-3, 0.997
BETA_MIN = float(os.environ.get("BETA_MIN", "5e-4"))
BETA_H = float(os.environ.get("BETA_H", "16"))
ETA_H, RHO_MIN = 0.5, 1e-3
# Fraction of the run over which the Heaviside beta ramps to BETA_H. Full projection
# needs the topology to have formed first, so it must not arrive too early.
BETA_AT = float(os.environ.get("BETA_AT", "0.8"))
ITERS = int(os.environ.get("ITERS", "250"))
CHUNK = int(os.environ.get("CHUNK", "10"))
BUDGET = float(os.environ.get("BUDGET", "7200"))
TAG = os.environ.get("TAG", "bracket")

# The rope's pull: down, sideways, and OUT of the plate. The z component is what makes this a 3-D
# problem; without it the answer is a planar truss and a reviewer can say "sweep the 2-D result".
PULL = np.array(
    [
        float(os.environ.get("PULL_X", "0.30")),
        float(os.environ.get("PULL_Y", "-0.85")),
        float(os.environ.get("PULL_Z", "0.43")),
    ]
)
PULL = PULL / np.linalg.norm(PULL)

blank = jno.Shape.polygon(
    [(0.0, 0.0), (LX, 0.0), (LX, SHOULDER), (TOPR, LY), (TOPL, LY), (0.0, SHOULDER)],
    size=H_MESH,
).extrude(LZ)
# Subtracted one at a time: `+` on shapes builds a named multi-material REGION set, not a union.
_bolt = lambda bx: jno.Shape.cylinder(bx, BY, -0.1, 0.0, 0.0, LZ + 0.2, RB, size=H_MESH)  # noqa: E731
d = (blank - _bolt(BX1) - _bolt(BX2)).domain()
cells = np.asarray(d._cells_p1())

# WARMSTART=<npz> continues from a previous design: its density seeds `rho` and its converged mesh
# becomes this run's starting geometry. Nothing is transferred or resampled -- the blank, the
# connectivity and the cell order are identical, so the two runs share a design vector exactly.
# That is what makes a cheap-then-expensive schedule possible: solve in the conventional space,
# then hand the SAME variables to the enriched one.
#
# The mesh must be moved here, before `pts0` is read and before any coordinate is promoted:
# `.trainable()` seeds each coordinate parameter from the mesh as it stands, so a later assignment
# would leave the design variables pointing at the original geometry.
_WARM = os.environ.get("WARMSTART", "")
_warm = None
if _WARM:
    _warm = np.load(os.path.join(OUT, _WARM))
    assert _warm["cells"].shape == cells.shape, (
        f"{_WARM} has {_warm['cells'].shape[0]} cells, this blank has {cells.shape[0]} -- a warm "
        "start needs the identical mesh, not merely a similar one"
    )
    assert np.array_equal(np.asarray(_warm["cells"]), cells), (
        f"{_WARM} has the same cell COUNT but a different connectivity; the design vectors would "
        "not correspond element for element"
    )
    d.mesh.points[:, :3] = _warm["pts_f"]
    print(f"warm start: {_WARM}  (C={float(np.asarray(_warm['C']).reshape(-1)[-1]):.4f}, "
          f"M_nd={float(4 * (_warm['rho_f'] * (1 - _warm['rho_f'])).mean()):.4f})", flush=True)

pts0 = np.asarray(d.mesh.points)[:, :3]
NT = cells.shape[0]
print("=" * 92, flush=True)
print(
    f"  BRACKET  {LX}x{LY}x{LZ} tapered, 2 holes r={RB}  h={H_MESH}  tets={NT}  dofs={3 * pts0.shape[0]}",
    flush=True,
)
print(
    f"           volfrac={VOLFRAC}  pull={np.round(PULL, 3)}  PSTAR={PSTAR}  BETA_H={BETA_H}  TAG={TAG}\n"
    # BUDGET is a HARD stop: the loop is `while elapsed < BUDGET and it < ITERS`, so a run that hits
    # the clock saves a partial design that looks complete. Printing both is what lets anything
    # downstream tell the two apart.
    f"           ITERS={ITERS}  BUDGET={BUDGET:.0f}s  CHUNK={CHUNK}  SPACE={SPACE}  RELOCATE={RELOCATE}",
    flush=True,
)
print("=" * 92, flush=True)

# --- design variables 2-4: interior nodal coordinates -------------------------------------------
# "Interior" here means not on ANY boundary of this blank -- including the hole walls and the sloped
# faces, which a bounding-box test would miss entirely and let the optimiser chew the outline away.
_bnd = np.zeros(pts0.shape[0], dtype=bool)
_LF = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
_seen: dict = {}
for _t in range(NT):
    _c = cells[_t]
    for _f in _LF:
        _k = tuple(sorted(int(_c[i]) for i in _f))
        _seen.pop(_k, None) if _k in _seen else _seen.setdefault(_k, _t)
for _k in _seen:
    _bnd[list(_k)] = True
_interior_ids = np.flatnonzero(~_bnd)

# What may move: everything EXCEPT the surfaces a boundary condition is anchored to. This used to be
# a bounding box over the interior nodes, which is not the interior of anything -- it promoted 175
# BOUNDARY nodes (the hole walls and the sloped faces) on top of the 277 interior ones, exactly what
# the comment above warns against.
#
# It fails silently, and that is the dangerous part. The "boltwalls" tag is materialised ONCE from
# the initial coordinates and clamps those node IDS for the whole run, so a node that drifts off the
# hole wall stays clamped while any independent re-solve of the saved mesh clamps whatever is
# geometrically on the wall instead. Measured on h25_Lagrange: 146 clamped nodes at setup, 106 still
# inside the clamp band at convergence, and the reported compliance 4.6412 against 5.2876 from an
# independent P1 solve of the same mesh and the same density -- a 13.9 % gap that grows with every
# iteration. With the nodes pinned the same check agrees to 0.00 % at every frame.
#
# The margin has to apply to BOUNDARY nodes only. A boundary node is what a Dirichlet facet is made
# of, so only a boundary node can wander into the clamp band and change the support set; an interior
# node never can, because connectivity is fixed and it can never become a boundary node. Freezing a
# solid cylinder instead -- interior nodes included -- is what broke the first attempt: it captured
# the elements that violate the element-volume bound, g_vmx became unsatisfiable by ANY nodal motion,
# and MMA spent the run in feasibility restoration (nodes moved 0.045 instead of 0.407, vmx never
# left 1.734, compliance ended at 1.65e3 against 4.07).
#
# With the split, the margins can be big enough to be airtight. A node moves at most MOVE_BOUND per
# axis, so its radial excursion in x-y is at most sqrt(2) * MOVE_BOUND: no free boundary node can
# reach the band, and no frozen one can leave it. `record()` still checks, so a miss is loud.
_RMARGIN = float(np.sqrt(2.0)) * MOVE_BOUND
_YMARGIN = MOVE_BOUND
_BND_XYZ = pts0[_bnd]


def _bc_zone_np(X, Y, Z):
    """The BC-carrying surfaces, grown by the margin: bolt clamp band and the loaded rope patch."""
    return (
        (np.hypot(X - BX1, Y - BY) < RB + 0.10 + _RMARGIN)
        | (np.hypot(X - BX2, Y - BY) < RB + 0.10 + _RMARGIN)
        | ((Y > LY - _YMARGIN) & (X > TOPL - _RMARGIN) & (X < TOPR + _RMARGIN))
    )


def _on_boundary_np(X, Y, Z):
    """Is each point one of the blank's initial BOUNDARY nodes? Exact, by coordinate match."""
    P = np.stack([X, Y, Z], axis=-1)
    onb = np.zeros(P.shape[0], dtype=bool)
    for i in range(0, P.shape[0], 1024):  # chunked: the pairwise block is O(chunk * n_boundary)
        blk = P[i : i + 1024]
        onb[i : i + 1024] = (((blk[:, None, :] - _BND_XYZ[None, :, :]) ** 2).sum(-1).min(axis=1)) < 1e-18
    return onb


# --- tangential sliding -------------------------------------------------------------------------
# The design domain is GIVEN data, so a boundary node may slide WITHIN its own surface but never off
# it. The boundary is then the same point set at every iteration and the blank's volume is exactly
# invariant. Letting the outline move freely instead is not a milder version of this, it is a
# different problem: with the material budget written as a fraction of the CURRENT cell volumes, the
# optimiser pushes the outline outwards and is handed more material for doing so. Measured, blank
# 12.72 -> 18.05 (+42 %), material 3.81 -> 5.39, and a compliance of 3.11 against 4.07 -- bought, not
# designed. (The budget is now absolute as well, so neither half of that can recur.)
#
# Sliding is expressed per AXIS because a coordinate design variable is a box-bounded scalar: a node
# on an axis-aligned face simply never gets its normal component promoted. Exact for the flat faces.
# The two tapered faces are not axis-aligned, so their in-plane freedom is not a box -- but they are
# extruded in z, so a taper node still slides in z and only x and y are withheld. A node on a
# non-axis-aligned EDGE (taper meeting a flat face) is frozen: 1-D freedom along it is not a box
# either, and guessing here would move the outline.
_STOL = 1e-7


def _taper_plane(pa, pb):
    """(n, c) of the z-extruded plane through two polygon corners, with n.(x, y) = c."""
    n = np.array([pb[1] - pa[1], pa[0] - pb[0]], dtype=float)
    n /= np.linalg.norm(n)
    return n, float(n @ np.asarray(pa, dtype=float))


_TAPERS = (_taper_plane((LX, SHOULDER), (TOPR, LY)), _taper_plane((TOPL, LY), (0.0, SHOULDER)))


def _blocked_axes(X, Y, Z):
    """Per-axis motion a boundary node may NOT take without leaving its own surface."""
    on_x = (np.abs(X) < _STOL) | (np.abs(X - LX) < _STOL)
    on_y = (np.abs(Y) < _STOL) | (np.abs(Y - LY) < _STOL)
    on_z = (np.abs(Z) < _STOL) | (np.abs(Z - LZ) < _STOL)
    on_t = np.zeros(X.shape, dtype=bool)
    for n, c in _TAPERS:
        on_t |= np.abs(n[0] * X + n[1] * Y - c) < _STOL
    return (on_x | on_t), (on_y | on_t), on_z


def _free_axis_where(axis):
    """Predicate for 'this node may move along `axis`'. Interior nodes always may."""

    def where(x, y, z):
        X, Y, Z = (np.asarray(v).reshape(-1) for v in (x, y, z))
        onb = _on_boundary_np(X, Y, Z)
        blocked = _blocked_axes(X, Y, Z)[axis]
        return ~(onb & (_bc_zone_np(X, Y, Z) | blocked))

    return where


def _clamped_np(P):
    """The bolt clamp, on plain numpy -- for checking the set has not drifted.

    Restricted to BOUNDARY nodes: a Dirichlet condition is applied on boundary facets, so only a
    boundary node can join or leave the support. Counting interior nodes here reports a drift that
    does not exist -- measured, 6-8 phantom nodes per frame while the independent re-solve agreed
    to 0.00 %.
    """
    return (
        (np.hypot(P[:, 0] - BX1, P[:, 1] - BY) < RB + 0.10)
        | (np.hypot(P[:, 0] - BX2, P[:, 1] - BY) < RB + 0.10)
    ) & _bnd


_CLAMP0 = _clamped_np(pts0)
xm = d.variable("mvx", where=_free_axis_where(0), split=True)[0]
ym = d.variable("mvy", where=_free_axis_where(1), split=True)[1]
zm = d.variable("mvz", where=_free_axis_where(2), split=True)[2]
# RELOCATE=0 leaves the nodes where gmsh put them -- the ablation that separates "what the density
# field bought" from "what moving the mesh bought". It has to skip `.trainable()` entirely: pinning
# by squeezing the MMA box to zero width instead makes the coordinate subproblem degenerate and the
# joint solve walks off (measured: C jumps to 2.4e3 at iteration 10 and freezes there).
if RELOCATE:
    xm.trainable(name="mesh_x"), ym.trainable(name="mesh_y"), zm.trainable(name="mesh_z")
    # Each axis carries its OWN id set now -- a node on the top face is free in x and y but not z --
    # so nothing here may reuse spec[0]'s ids for the others.
    _per = {int(sp["axis"]): np.asarray(sp["ids"], dtype=int) for sp in d._trainable_coords}
    _any = np.unique(np.concatenate([v for v in _per.values()])) if _per else np.array([], dtype=int)
    print(
        f"movable: {_any.size} of {pts0.shape[0]} nodes "
        f"({int((~_bnd[_any]).sum())} interior, {int(_bnd[_any].sum())} sliding on the surface); "
        f"free dofs x/y/z = {_per[0].size}/{_per[1].size}/{_per[2].size} of {3 * pts0.shape[0]}; "
        f"{int(_CLAMP0.sum())} clamped; BC boundary frozen, margins r={_RMARGIN:.3f} y={_YMARGIN:.3f}",
        flush=True,
    )
    for spec in d._trainable_coords:
        c0 = pts0[np.asarray(spec["ids"], dtype=int), int(spec["axis"])]
        spec["expr"].optimizer(
            jno.optimizers.mma(
                move=0.02,
                lower=c0 - MOVE_BOUND,
                upper=c0 + MOVE_BOUND,
                move_gamma=0.9985,
                move_min=0.001,
            )
        )
else:
    print("RELOCATE=0: nodes are pinned, this is a density-only run", flush=True)

# --- design variable 1 -----------------------------------------------------------------------
# SPACE=cover turns on E-FEM (the paper's Sec. 2.2, eq. 6-11). It is not cosmetic. Measured on the
# corrected h=0.25 pair -- identical mesh, load, supports, material budget and iteration count, the
# formulation the only difference -- against an independent solve of the geometry each design
# actually produced:
#
#     Lagrange   reported 4.8591,  its extracted solid refines to >= 17.0401   (understates >= 251 %)
#     E-FEM      reported 6.0357,  its extracted solid refines to >=  6.1108   (understates >=   1.2 %)
#
# Refined to 4M tets, where the series' increments finally settle to a ratio of ~0.5 -- O(h^1), the
# rate a faceted body with re-entrant edges gives, not the O(h^2) a smooth one would. Extrapolating
# from that: Lagrange ~18.6, E-FEM ~6.26.
#
# The mechanism is not element distortion (the equal-count control gives +3.67 % and -0.09 %), it is
# material: at rho>0.5 the Lagrange design retains only 3.3836 of its 3.8159, so 11.3 % of it is grey
# that thresholding destroys, against 1.8 % for E-FEM. Their 2-D Fig. 7 is the same shape: +17.6 %
# without it, -0.5 % with.
#
# Both bounds are one-sided and still climbing, so read them as ">=" and not as converged values.
# An earlier "+29.0 % attributable" claim measured here is WITHDRAWN -- it predates the support-drift
# fix and was computed against a compliance the optimiser could not reproduce.
u, phi = d.fem_symbols(value_shape=(3,), **({"space": "cover"} if SPACE == "cover" else {}))
_r, s = d.fem_symbols(space="P0", names=("r", "s"))
rho = jno.np.parameter(s, name="rho")
rho.dtype(jnp.float64)
if _warm is not None:
    _rho0 = jnp.asarray(np.asarray(_warm["rho_f"], dtype=float))
    rho.initialize(lambda k, sh, dtype=None, _v=_rho0: _v.reshape(sh))
else:
    rho.initialize(jax.nn.initializers.constant(VOLFRAC))
rho.optimizer(
    jno.optimizers.mma(
        move=0.15, lower=1e-3, upper=1.0, move_gamma=0.9985, move_min=0.0075
    )
)

cen0 = pts0[cells].mean(axis=1)
_rb1 = np.hypot(cen0[:, 0] - BX1, cen0[:, 1] - BY)
_rb2 = np.hypot(cen0[:, 0] - BX2, cen0[:, 1] - BY)
solid = (
    ((_rb1 < RB + WASHER) | (_rb2 < RB + WASHER))  # washer footprints under the bolts
    | ((cen0[:, 1] > LY - EYE) & (cen0[:, 0] > TOPL) & (cen0[:, 0] < TOPR))
)  # the rope eye
pmask = jnp.asarray(solid)
print(f"non-design solid: {int(solid.sum())} of {NT} tets", flush=True)

patch = d.patch_filter()

# The filter reparameterises and the projection composes, because one is non-local and the other is
# pointwise. That split is not stylistic: a `constrain()` transform lives in the static half of the
# partition, so a `beta` read there is a trace-time CONSTANT. Mutating it to ramp the sharpness --
# which is what this script used to do -- changes nothing the step-program cache can see, so the
# gradient stays at the starting beta while `crux.eval` recompiles and reports the new one. Measured:
# a 250-iteration run at BETA_H=16 ended with V=0.7734, 23 % of its material unspent, because the
# design was optimised at beta=1 and only REPORTED at 16.
def filtered(r):
    """Patch filter, with the non-design region forced solid. Non-local, so it reparameterises."""
    return jnp.where(pmask, 1.0, patch(r))


rho.constrain(filtered)

# No clip. `E(r) = EMIN + r**PENAL * (E0 - EMIN)` is already regular at r = 0, so clipping buys no
# conditioning -- what it buys is a floor under the physical density, and therefore under
# M_nd = 4*mean(r(1-r)), which is the number this run exists to drive down. The old clip at 1e-3 is
# visible in the results as an extraction that had to threshold at rho > 0.0031 to recover its own
# material. The void stiffness lives in EMIN, once.
# Named apart from the perimeter barrier's `beta_p`, which is rebound further down; both are
# scheduled scalars and a shared name here would work only by accident of ordering.
beta_h_p = jno.np.parameter((1,), name="beta_h")
beta_h_p.dtype(jnp.float64)
beta_h_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, 1.0))
beta_h_p.optimizer(optax.sgd(1.0))

# `rho` is the FILTERED design; `rho_phys` is what the physics, the volume and every diagnostic see.
# Unindexed: the scheduled scalar arrives 0-dimensional here, so `beta_h_p[0]` over-indexes it.
# A scalar broadcasts against the per-element density either way, so nothing is lost.
rho_phys = rho.project(beta_h_p, eta=ETA_H) if BETA_H > 0 else rho

# --- supports and load ---------------------------------------------------------------------------
# The bolts clamp the HOLE WALLS -- a cylindrical surface, not a coordinate plane. Names avoid jNO's
# own face tags ("top"/"bottom"/"left"), which silently override a `where=` predicate.
xi, yi, zi, _ = d.variable("interior", split=True)
bolts = d.variable(
    "boltwalls",
    where=lambda x, y, z: (
        (jnp.hypot(x - BX1, y - BY) < RB + 0.10)
        | (jnp.hypot(x - BX2, y - BY) < RB + 0.10)
    ),
    split=True,
)[:3]
eye = d.variable(
    "ropepatch",
    where=lambda x, y, z: (y > LY - TOL) & (x > TOPL - TOL) & (x < TOPR + TOL),
    split=True,
)[:3]
eps = lambda w: symgrad(w, [xi, yi, zi])  # noqa: E731
a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731
E = lambda r: EMIN + r**PENAL * (E0 - EMIN)  # noqa: E731


def _facet_area(keep):
    """Area of the boundary facets whose three nodes all satisfy ``keep``.

    Geometric rather than assemble-then-rescale: a second ``jno.fem(...)`` on the same domain
    returns the FIRST operator (measured: tractions 1.0 and 10.0 both give |f| = 0.9692), so a
    calibrate-and-rebuild round trip is silently a no-op.
    """
    tri = np.array(list(_seen.keys()), dtype=int)
    tri = tri[keep(pts0[tri]).all(axis=1)]
    v = pts0[tri]
    return float(
        np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum()
        / 2.0
    )


A_eye = _facet_area(
    lambda P: (
        (P[..., 1] > LY - 1e-6) & (P[..., 0] > TOPL - 1e-6) & (P[..., 0] < TOPR + 1e-6)
    )
)
assert A_eye > 1e-12, "the rope patch has no boundary facets"
print(f"rope patch area {A_eye:.4f} (nominal {(TOPR - TOPL) * LZ:.4f})", flush=True)

# jNO's element-loop chunking is what caps the batched AD intermediate, but its policy floors the
# chunk at 8192 CELLS and then disables chunking entirely when the mesh is smaller than that. Our
# mesh is 4,082 tets, so it never engages -- and the per-cell cost is n_test * n_local**2, which the
# cover space raises 64x (48**3 against 12**3). Measured peak RSS at 4,082 tets: Lagrange 2.5 GB,
# cover 13.1 GB, cover + trainable coordinates 36.2 GB. FEMCHUNK caps it by hand.
# Default ON for the enriched space, because jNO's own policy cannot protect it: the chunk is
# floored at 8192 CELLS while the thing it bounds is BYTES, so a 48-DOF element blows the 8 MB budget
# by three orders of magnitude and, on a mesh smaller than the floor, chunking is skipped entirely.
# Measured peak RSS with trainable coordinates: 4,082 tets 36.2 GB -> 13.4 GB, and h=0.15 (16,691
# tets) fits in 20.6 GB, which is LESS than the coarse mesh used before. Costs ~6 % runtime, and
# iteration 0 is identical to every printed digit, so it is exact. Lagrange needs none of this -- its
# per-cell cost is 64x smaller (12**3 against 48**3) and jNO's policy handles it.
_FEMCHUNK = int(os.environ.get("FEMCHUNK", "128" if SPACE == "cover" else "0"))
fem = jno.fem(
    [
        E(rho_phys) * a(eps(u), eps(phi)),
        u(*bolts) - (0.0, 0.0, 0.0),
        -1.0 * inner(jnp.asarray(PULL / A_eye), phi.bind(**dict(zip("xyz", eye))), 1),
    ],
    quad_degree=2,
    **({"chunk": _FEMCHUNK} if _FEMCHUNK > 0 else {}),
)

# Every runtime parameter the form reads has to be supplied here, not just the density: an unset
# `beta_h` arrives as 0, where the projection's numerator and denominator both vanish, and the load
# probe below then measures a NaN operator and reports that the patch assembled nothing.
_probe = {"rho": jnp.full(NT, VOLFRAC)}
if BETA_H > 0:
    _probe["beta_h"] = jnp.asarray(1.0)
_A_op, _b = fem.operator.evaluate(_probe)
_f = np.asarray(jnp.asarray(_b)).reshape(-1, 3)
_res = _f.sum(axis=0)
print(
    f"load |f| = {np.abs(_f).sum():.4f}   resultant = ({_res[0]:+.4f}, {_res[1]:+.4f}, {_res[2]:+.4f})",
    flush=True,
)
print(
    f"           unit resultant = {np.round(_res / max(np.linalg.norm(_res), 1e-300), 3)}  "
    f"want {np.round(PULL, 3)}",
    flush=True,
)
assert np.linalg.norm(_res) > 1e-9, "the rope patch assembled nothing"
_cos = float(_res @ PULL / (np.linalg.norm(_res) * np.linalg.norm(PULL)))
assert _cos > 0.999, (
    f"the assembled resultant is not along the intended pull (cos = {_cos:.4f})"
)
assert abs(np.linalg.norm(_res) - 1.0) < 0.02, (
    f"the resultant should be unit, got {np.linalg.norm(_res):.4f}"
)

fixed_nodes = np.asarray(
    (np.hypot(pts0[:, 0] - BX1, pts0[:, 1] - BY) < RB + 0.10)
    | (np.hypot(pts0[:, 0] - BX2, pts0[:, 1] - BY) < RB + 0.10)
)
print(f"clamped nodes: {int(fixed_nodes.sum())}", flush=True)

# --- objective and constraints ---------------------------------------------------------------
compliance = (E(rho_phys) * a(eps(u), eps(u))).integrate(fem, solver=pardiso_lu)
cellv = d.cell_volume()
# Against the INITIAL blank, not the current one. `cellv.sum` is a functional of the moving mesh, so
# a fraction of it is a moving target: any outward motion of the outline raises the denominator and
# buys material. Measured before this was pinned down: +42 % blank, +42 % material, and a compliance
# that improved for neither reason. Tangential sliding already holds the blank fixed; this makes the
# budget independent of the mesh regardless.
_V0_BLANK = float(
    np.abs(
        np.linalg.det(np.stack([pts0[cells][:, i + 1] - pts0[cells][:, 0] for i in range(3)], -1))
    ).sum()
    / 6.0
)
print(f"material budget: {VOLFRAC} x {_V0_BLANK:.4f} = {VOLFRAC * _V0_BLANK:.4f} (fixed)", flush=True)
volume = (rho_phys.reshape(-1) * cellv).sum / (VOLFRAC * _V0_BLANK)
g_ang = (
    ((2 * np.pi - d.cell_angles()) / (2 * np.pi - THETA_MIN))
    .reshape(-1)
    .pnorm(PN, normalize=True)
)
g_asp = (d.cell_aspect() / ASPECT_MAX).pnorm(PN, normalize=True)
_v0 = (
    np.abs(
        np.linalg.det(
            np.stack([pts0[cells][:, i + 1] - pts0[cells][:, 0] for i in range(3)], -1)
        )
    )
    / 6.0
)
# eq. (25)/(27) use [0.05, 0.75] against a NOMINAL unit-edge element (a triangle of area 0.433),
# i.e. [0.115, 1.732] x nominal. That ratio is what transfers, not the numbers -- and it has to be
# taken against the mesh's OWN median, because gmsh's `size=h` does not produce ideal elements: our
# median tet is 1.79x the regular tet of edge h. Measured on the h=0.15 bracket, these bounds leave
# 0.00% of cells under V0MIN and 0.05% over the ACTIVE cap (2-rho)V0MAX at the starting density, so
# the run begins feasible. The previous mesh-percentile rule was 1.72x looser on the cap.
_v0_med = float(np.median(_v0))
V0MAX = float(os.environ.get("V0MAX_FRAC", "1.732")) * _v0_med
V0MIN = float(os.environ.get("V0MIN_FRAC", "0.115")) * _v0_med
rho_e = rho_phys.reshape(-1)
g_vmx = (cellv / ((2.0 - rho_e) * V0MAX)).pnorm(PN, normalize=True)
g_vmn = ((2 * V0MAX - cellv) / (2 * V0MAX - V0MIN)).pnorm(PN, normalize=True)
perim = rho_phys.perimeter(zeta=0.1)

# --- surface smoothness ------------------------------------------------------------------------
# The perimeter is a GLOBAL budget and converged designs exploit that: measured on this bracket at
# h=0.15 with the barrier active and satisfied, the angle between adjacent surface facets had a
# median of 33.5 deg, a p99 of 88.9, and 27 % of adjacent pairs above 60 deg. Both cheap levers
# were tried and both failed -- raising MOVE_FRAC un-pinned the nodes (27.3 % -> 6.2 % at the move
# bound) and barely moved the roughness (33.5 -> 32.0 deg), and tightening PSTAR to 31 bought
# 26.9 deg at 91 % worse compliance. Neither is a local measure of bending, so this is.
# `curvature()` hands back one value per RIDGE, so the aggregate is a design choice, not a detail.
# SMOOTH_P=0 sums them, which buys down the bulk of the boundary; SMOOTH_P>0 takes a p-norm, which
# goes after the WORST folds instead. That distinction is the one that matters here: measured on the
# converged pair, the summed penalty took the ISO-SURFACE median from 13.9 to 11.8 deg but left 9 %
# of its edges folded past 60 deg -- against 0 % for a smooth sphere on this same mesh, which is the
# floor a perfect design would hit. The bulk was never the problem; the tail is.
curv_r = rho_phys.curvature(zeta=0.1)
SMOOTH = float(os.environ.get("SMOOTH", "0"))
SMOOTH_P = float(os.environ.get("SMOOTH_P", "0"))
curv = curv_r.pnorm(SMOOTH_P) if SMOOTH_P > 0 else curv_r.sum
if SMOOTH > 0:
    # A weight in compliance units needs a scale, and the design cannot supply one: at iteration 0
    # the density is uniform and the bending is EXACTLY zero. So take the scale from the mesh --
    # what a single flat cut through it already costs, purely from its own faceting. That floor is
    # design-independent and does not track refinement (measured 50.7 at h=0.5 against 52.4 at
    # h=0.35 on a test box), so SMOOTH stays comparable across meshes.
    _coord0 = [
        jnp.asarray(pts0[np.asarray(sp["ids"], dtype=int), int(sp["axis"])])
        for sp in (getattr(d, "_trainable_coords", []) or [])
    ]
    _cut = (pts0[cells].mean(axis=1)[:, 1] > 0.5 * (pts0[:, 1].min() + pts0[:, 1].max())).astype(float)
    _flat = np.asarray(curv_r.fn(jnp.asarray(_cut), *_coord0))
    # curvature() hands back one value per RIDGE. A jNO old enough to return the SUM instead makes
    # `.pnorm(p)` a no-op on a scalar, so a p-norm run silently reproduces the summed one and the
    # only visible symptom is a reference that matches the sum's. Assert the shape, not the value.
    assert _flat.ndim == 1 and _flat.size > len(cells), (
        f"curvature() returned {_flat.shape}; expected one value per ridge (> {len(cells)} of them). "
        "vendor/jNO is behind -- `.pnorm()` on a scalar is the identity and the run would be a lie."
    )
    # The SAME aggregate the term uses, so the weight means the same thing either way.
    S_REF = float((_flat.astype(np.float64) ** SMOOTH_P).sum() ** (1.0 / SMOOTH_P)) if SMOOTH_P > 0 else float(_flat.sum())
    assert S_REF > 0.0, "a flat cut through an unstructured mesh has nonzero bending; got zero"
    print(f"smoothness: SMOOTH={SMOOTH} aggregate={'p-norm p=' + str(SMOOTH_P) if SMOOTH_P > 0 else 'sum'}  "
          f"flat-cut reference S_ref={S_REF:.4g}  weight={SMOOTH / S_REF:.4e}", flush=True)

terms, callbacks = [compliance], []
if BETA_H > 0:
    # Keyed to the run through `over`, and counted by the callback itself rather than read off
    # `epoch` -- `epoch` is per-solve()-call, so a schedule reading it makes CHUNK a physics knob
    # (measured: beta 10.8 at CHUNK=10 against 1.6 at CHUNK=50, M_nd 0.008 against 0.078).
    _hv = jno.optimizers.heaviside_continuation(
        beta_h_p, start=1.0, maximum=BETA_H, over=max(1, int(BETA_AT * ITERS))
    )
    callbacks.append(_hv)
if SMOOTH > 0:
    terms.append(((SMOOTH / S_REF) * curv).name("S"))
if PSTAR > 0:
    beta_p = jno.np.parameter((1,), name="beta")
    beta_p.dtype(jnp.float64)
    beta_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, BETA))
    beta_p.optimizer(optax.sgd(1.0))
    terms.append((beta_p[0] * perim.log_barrier(PSTAR)).name("R"))
    callbacks.append(
        jno.optimizers.geometric_decay(beta_p, GAMMA, start=BETA, minimum=BETA_MIN)
    )
terms += [
    jno.le(volume, 1.0),
    jno.le(g_ang, 1.0),
    jno.le(g_asp, 1.0),
    jno.le(g_vmx, 1.0),
    jno.le(g_vmn, 1.0),
]
crux = jno.core(terms, domain=jno.domain.from_array({"_": np.zeros((1, 1))}))

coord_specs = list(getattr(d, "_trainable_coords", []) or [])
coord_nodes = [sp["expr"] for sp in coord_specs]
hist = {k: [] for k in ("it", "C", "V", "P", "S", "mnd", "ang", "asp", "vmx", "vmn")}
frames_rho, frames_pts = [], []
ev = lambda n: float(np.asarray(crux.eval([n])).reshape(-1)[0])  # noqa: E731
DST = os.path.join(OUT, f"topo3d_{TAG}.npz")


def checkpoint():
    tmp = DST + ".tmp.npz"
    np.savez(
        tmp,
        cells=cells,
        pts0=pts0,
        pts_f=frames_pts[-1],
        rho_f=frames_rho[-1],
        frames_rho=np.array(frames_rho),
        frames_pts=np.array(frames_pts),
        solid=solid,
        load_f=_f,
        fixed=fixed_nodes,
        **{k: np.asarray(v) for k, v in hist.items()},
    )
    os.replace(tmp, DST)


def record(it_now):
    """Snapshot the design, the mesh and the history at ``it_now``, and checkpoint."""
    global it
    it = it_now
    # The PROJECTED field -- what the physics saw and what the reanalysis will extract.
    r = np.asarray(crux.eval([rho_phys])).reshape(-1)
    p = pts0.copy()
    for sp, node in zip(coord_specs, coord_nodes):
        p[np.asarray(sp["ids"], dtype=int), int(sp["axis"])] = np.asarray(
            crux.eval([node])
        ).reshape(-1)
    for k, v in (
        ("it", it),
        ("C", ev(compliance)),
        ("V", ev(volume)),
        ("P", ev(perim)),
        ("S", ev(curv_r.sum)),
        ("mnd", float(4.0 * np.mean(r * (1.0 - r)))),
        ("ang", ev(g_ang)),
        ("asp", ev(g_asp)),
        ("vmx", ev(g_vmx)),
        ("vmn", ev(g_vmn)),
    ):
        hist[k].append(v)
    # The failure this guards against is silent by construction: the clamp is bound to node IDs at
    # setup, so a drifting support still "works" and only an independent re-solve ever notices.
    _cf = _clamped_np(p)
    if int((_cf != _CLAMP0).sum()):
        print(
            f"  WARNING: the clamped set drifted, {int(_CLAMP0.sum())} -> {int(_cf.sum())} nodes "
            f"({int((_cf != _CLAMP0).sum())} changed). The reported compliance is no longer the "
            f"compliance of the saved mesh.",
            flush=True,
        )
    frames_rho.append(r.copy())
    frames_pts.append(p.copy())
    print(
        f"  it {it:4d}  C={hist['C'][-1]:10.4f}  V={hist['V'][-1]:.4f}  P={hist['P'][-1]:8.2f}  "
        f"S={hist['S'][-1]:8.2f}  M_nd={hist['mnd'][-1]:.4f}  beta_H={(_hv.value if BETA_H > 0 else 0.0):.2f}  g_ang={hist['ang'][-1]:.3f} "
        f"asp={hist['asp'][-1]:.3f} vmx={hist['vmx'][-1]:.3f}  ({time.time() - t0:.0f}s)",
        flush=True,
    )
    checkpoint()


it, t0 = 0, time.time()
record(0)
while time.time() - t0 < BUDGET and it < ITERS:
    crux.solve(CHUNK, callbacks=callbacks)
    record(it + CHUNK)


print(
    f"saved topo3d_{TAG}.npz  ({len(frames_rho)} frames, {it} iterations, {(time.time() - t0) / 60:.1f} min)",
    flush=True,
)
_ru, _rc = (
    resource.getrusage(resource.RUSAGE_SELF),
    resource.getrusage(resource.RUSAGE_CHILDREN),
)
_cpu = _ru.ru_utime + _ru.ru_stime + _rc.ru_utime + _rc.ru_stime
print(
    f"cpu {_cpu:.0f} s over {time.time() - t0:.0f} s wall -> {_cpu / (time.time() - t0):.1f} cores",
    flush=True,
)
