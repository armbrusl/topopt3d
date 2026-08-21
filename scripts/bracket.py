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
pts0 = np.asarray(d.mesh.points)[:, :3]
NT = cells.shape[0]
print("=" * 92, flush=True)
print(
    f"  BRACKET  {LX}x{LY}x{LZ} tapered, 2 holes r={RB}  h={H_MESH}  tets={NT}  dofs={3 * pts0.shape[0]}",
    flush=True,
)
print(
    f"           volfrac={VOLFRAC}  pull={np.round(PULL, 3)}  PSTAR={PSTAR}  BETA_H={BETA_H}  TAG={TAG}",
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
print(f"movable interior nodes: {_interior_ids.size} of {pts0.shape[0]}", flush=True)
_lo, _hi = pts0[_interior_ids].min(axis=0), pts0[_interior_ids].max(axis=0)
xm, ym, zm, _ = d.variable(
    "mv",
    where=lambda x, y, z: (
        (x > _lo[0] - 1e-9)
        & (x < _hi[0] + 1e-9)
        & (y > _lo[1] - 1e-9)
        & (y < _hi[1] + 1e-9)
        & (z > _lo[2] - 1e-9)
        & (z < _hi[2] + 1e-9)
    ),
    split=True,
)
xm.trainable(name="mesh_x"), ym.trainable(name="mesh_y"), zm.trainable(name="mesh_z")
ids = np.asarray(d._trainable_coords[0]["ids"], dtype=int)
for spec in d._trainable_coords:
    c0 = pts0[ids, int(spec["axis"])]
    spec["expr"].optimizer(
        jno.optimizers.mma(
            move=0.02,
            lower=c0 - MOVE_BOUND,
            upper=c0 + MOVE_BOUND,
            move_gamma=0.9985,
            move_min=0.001,
        )
    )

# --- design variable 1 -----------------------------------------------------------------------
# SPACE=cover turns on E-FEM (the paper's Sec. 2.2, eq. 6-11). It is not cosmetic: with the
# conventional formulation the converged 3-D design reanalyses 29.0% stiffer than it really is,
# because nodal movement WITHIN the geometric constraints still buys artificial stiffness. The
# paper's own words: geometric constraints and the enriched approximation "play complementary
# roles". Their 2-D Fig. 7 measures +17.6% without it and -0.5% with.
SPACE = os.environ.get("SPACE", "Lagrange")
u, phi = d.fem_symbols(value_shape=(3,), **({"space": "cover"} if SPACE == "cover" else {}))
_r, s = d.fem_symbols(space="P0", names=("r", "s"))
rho = jno.np.parameter(s, name="rho")
rho.dtype(jnp.float64)
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
_beta_now = [1.0]


def physical(r):
    x = jnp.where(pmask, 1.0, patch(r))
    if BETA_H <= 0:
        return jnp.clip(x, RHO_MIN, 1.0)
    tb = jnp.tanh(_beta_now[0] * ETA_H)
    proj = (tb + jnp.tanh(_beta_now[0] * (x - ETA_H))) / (
        tb + jnp.tanh(_beta_now[0] * (1.0 - ETA_H))
    )
    return jnp.clip(proj, RHO_MIN, 1.0)


rho.constrain(physical)

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

fem = jno.fem(
    [
        E(rho) * a(eps(u), eps(phi)),
        u(*bolts) - (0.0, 0.0, 0.0),
        -1.0 * inner(jnp.asarray(PULL / A_eye), phi.bind(**dict(zip("xyz", eye))), 1),
    ],
    quad_degree=2,
)

_A_op, _b = fem.operator.evaluate({"rho": jnp.full(NT, VOLFRAC)})
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
compliance = (E(rho) * a(eps(u), eps(u))).integrate(fem, solver=pardiso_lu)
cellv = d.cell_volume()
volume = (rho.reshape(-1) * cellv).sum / (VOLFRAC * cellv.sum)
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
rho_e = rho.reshape(-1)
g_vmx = (cellv / ((2.0 - rho_e) * V0MAX)).pnorm(PN, normalize=True)
g_vmn = ((2 * V0MAX - cellv) / (2 * V0MAX - V0MIN)).pnorm(PN, normalize=True)
perim = rho.perimeter(zeta=0.1)

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
curv_r = rho.curvature(zeta=0.1)
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
    r = np.asarray(crux.eval([rho])).reshape(-1)
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
    frames_rho.append(r.copy())
    frames_pts.append(p.copy())
    print(
        f"  it {it:4d}  C={hist['C'][-1]:10.4f}  V={hist['V'][-1]:.4f}  P={hist['P'][-1]:8.2f}  "
        f"S={hist['S'][-1]:8.2f}  M_nd={hist['mnd'][-1]:.4f}  beta_H={_beta_now[0]:.1f}  g_ang={hist['ang'][-1]:.3f} "
        f"asp={hist['asp'][-1]:.3f} vmx={hist['vmx'][-1]:.3f}  ({time.time() - t0:.0f}s)",
        flush=True,
    )
    checkpoint()


it, t0 = 0, time.time()
record(0)
while time.time() - t0 < BUDGET and it < ITERS:
    if BETA_H > 0:
        # Tied to a FRACTION OF THE RUN, not to a fixed per-iteration rate and not to how the
        # run is chunked. A fixed rate silently couples the projection to the iteration budget:
        # 1.10 per 10 iterations reaches beta = 10.8 by iteration 250 but only 3.8 by 140, so a
        # shorter run is not the same optimisation stopped early -- it is a different one, barely
        # projected. Ramping once per `crux.solve()` call was worse still and made CHUNK a physics
        # knob (beta 10.8 at CHUNK=10 against 1.6 at CHUNK=50, M_nd 0.008 against 0.078).
        _beta_now[0] = min(BETA_H, float(BETA_H) ** min(1.0, it / max(BETA_AT * ITERS, 1.0)))
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
