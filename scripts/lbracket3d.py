"""The L-bracket, in 3-D -- the paper's second benchmark, which has no 3-D answer anywhere.

Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403, Sec. 3.2 runs an L-bracket in 2-D
(their Fig. 9: L = 40, 2,320 elements, volfrac 0.35). Their Sec. 2.3.2 leaves the tetrahedral case
open, so extending THIS benchmark is the point of the script -- not reproducing their 2-D numbers,
which cannot be done anyway: compliance scales as the square of the applied load and the paper
never states its load magnitude (see `scripts/paper2d.py` for that measurement).

**What makes it 3-D rather than a thick 2-D plate.** An extruded L under an in-plane load is a
plane-stress problem wearing a third dimension: every section carries the same thing and the
optimiser has nothing to do through the thickness. The load here has an out-of-plane component, so
the horizontal arm is twisted as well as bent and the re-entrant corner carries a genuinely 3-D
stress state. That is the same reason `bracket.py` pulls off-axis.

**Why the re-entrant corner is the interesting part.** It is the classic stress concentration in
this benchmark, and it is exactly where a fixed mesh cannot put resolution without refining
globally. A deformable mesh can migrate nodes into it. That is the mechanism the method claims, so
a benchmark whose whole difficulty sits at one corner is a fair test of it.

    python scripts/lbracket3d.py                       # E-FEM, the proposed formulation
    SPACE=Lagrange python scripts/lbracket3d.py        # the conventional comparison
    RELOCATE=0 python scripts/lbracket3d.py            # fixed-mesh SIMP baseline

This is the first script in the repo built on `scripts/topopt/` rather than on its own copy of the
coordinate, budget and constraint logic -- see that package's STATUS note. The five older scripts
still carry their own copies.
"""

import os
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
from topopt import constraints as C  # noqa: E402
from topopt import coords as CO  # noqa: E402
from topopt import mesh as M  # noqa: E402

inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace

OUT = os.environ.get("OUTDIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"
)
os.makedirs(OUT, exist_ok=True)

# --- geometry ---------------------------------------------------------------------------------
# Their Fig. 9a proportions: the full L x L square minus its upper-right 0.6L x 0.6L block,
# extruded by THICK. Scaled to L = 4 so h = 0.25 gives ~3.3k tets, the size `bracket.py` runs at.
L = float(os.environ.get("LSIZE", "4.0"))
THICK = float(os.environ.get("THICK", "1.0"))
ARM = 0.4 * L  # width of the vertical arm and height of the horizontal one
H_MESH = float(os.environ.get("HMESH", "0.25"))
SPACE = os.environ.get("SPACE", "cover")
RELOCATE = int(os.environ.get("RELOCATE", "1"))
TAG = os.environ.get("TAG", f"lb3_{SPACE}")

E0, NU, PENAL = 1.0, 0.3, 3.0
EMIN = float(os.environ.get("EMIN", "1e-4"))
LAM, MU = E0 * NU / ((1 + NU) * (1 - 2 * NU)), E0 / (2 * (1 + NU))  # 3-D, not plane stress
VOLFRAC = float(os.environ.get("VOLFRAC", "0.35"))  # theirs, Fig. 9
# 0.5 x h, following the torsion measurement: at 2 x h the optimiser walks the aspect constraint
# to 1.33, at 0.5 x h it converges feasible for ~0.7 % more compliance. The paper's +/-2 is in
# UNITS of its own h = 1, i.e. 2 x h, which does not transfer to tets.
MOVE_BOUND = float(os.environ.get("MOVE_FRAC", "0.5")) * H_MESH
# Out of plane. A purely in-plane pull makes this a thick 2-D problem; this twists the arm too.
PULL = np.asarray([float(v) for v in os.environ.get("PULL", "0,-1,0.45").split(",")])
PULL = PULL / np.linalg.norm(PULL)
PSTAR = float(os.environ.get("PSTAR", "0"))  # perimeter AREA target; 0 = probe it first
BETA, GAMMA, BETA_MIN = 2e-4, 0.997, 1e-4
BETA_H = float(os.environ.get("BETA_H", "16"))
BETA_AT = float(os.environ.get("BETA_AT", "0.7"))
ITERS = int(os.environ.get("ITERS", "250"))
CHUNK = int(os.environ.get("CHUNK", "25"))  # NOT bookkeeping -- see torsion.py's MOVE_FRAC note
BUDGET = float(os.environ.get("BUDGET", "7200"))

shape = jno.Shape.box(0, 0, 0, L, L, THICK, size=H_MESH) - jno.Shape.box(
    ARM, ARM, -THICK, L, L, 2 * THICK, size=H_MESH
)
d = shape.domain()
pts0 = np.asarray(d.mesh.points)
cells = np.asarray(d._cells_p1())
NT, NN = cells.shape[0], pts0.shape[0]
TOL = 1e-7 * L

# WARMSTART=<npz> continues a previous run: its density seeds `rho` and its converged mesh becomes
# this run's starting geometry. The point here is resilience, not tuning -- with per-chunk
# checkpointing a killed run leaves a complete design, and this is what lets the next attempt pick
# it up instead of starting from the blank. Mirrors `bracket.py`'s mechanism, including its two
# asserts: a warm start onto a DIFFERENT mesh silently reinterprets one cell's density as another
# cell's, which no downstream check would catch.
_WARM = os.environ.get("WARMSTART", "")
_warm = None
if _WARM:
    _warm = np.load(os.path.join(OUT, _WARM))
    assert _warm["cells"].shape == cells.shape, (
        f"{_WARM} has {_warm['cells'].shape[0]} cells, this blank has {cells.shape[0]} -- a warm "
        f"start only means anything on the identical mesh")
    assert np.array_equal(np.asarray(_warm["cells"]), cells), (
        f"{_WARM} has the same number of cells but a different connectivity -- its densities do "
        f"not correspond to this blank's elements")
    d.mesh.points[:, :3] = _warm["pts_f"]
    pts0 = np.asarray(d.mesh.points)
    _wC = np.asarray(_warm["C"]).reshape(-1)
    print(f"warm start: {_WARM}  (C={float(_wC[-1]):.4f}, "
          f"M_nd={float(4 * (_warm['rho_f'] * (1 - _warm['rho_f'])).mean()):.4f})", flush=True)

# --- boundary conditions ------------------------------------------------------------------------
# Clamp: the TOP face of the vertical arm (their Fig. 9a hatching).
# Load:  the free END face of the horizontal arm, the opposite corner of the L.
clamped = lambda X, Y, Z: Y > L - TOL  # noqa: E731
loaded = lambda X, Y, Z: (X > L - TOL) & (Y < ARM + TOL)  # noqa: E731

_bnd = M.boundary_mask(cells, NN)
_nc = int((clamped(*pts0.T) & _bnd).sum())
_nl = int((loaded(*pts0.T) & _bnd).sum())
if _nc == 0 or _nl == 0:
    raise SystemExit(f"clamp caught {_nc} nodes and the load face {_nl}: one of them is empty")

# `split=True` yields a trailing element past the three coordinates, so take [:3] / unpack the
# extra -- the idiom bracket.py uses.
clamp_v = d.variable("clamp", where=lambda x, y, z: clamped(x, y, z), split=True)[:3]
end_v = d.variable("endface", where=lambda x, y, z: loaded(x, y, z), split=True)[:3]
xi, yi, zi, _ = d.variable("interior", split=True)

# --- design variables 2 and 3: the movable coordinates -------------------------------------------
# The BC-carrying surfaces are frozen with a margin, because a boundary node that drifts INTO the
# clamp band silently changes the support set and the reported compliance stops describing the
# geometry -- the 13.9 % error this repo retracted once already. `topopt.coords` owns that rule.
_ZONES = (
    lambda X, Y, Z: Y > L - ARM * 0.0 - TOL,          # the clamped top face
    lambda X, Y, Z: (X > L - TOL) & (Y < ARM + TOL),  # the loaded end face
)
if RELOCATE:
    CO.register(
        d, pts0, cells, zones=_ZONES, margin=float(np.sqrt(3.0)) * MOVE_BOUND,
        move_bound=MOVE_BOUND,
        mma=lambda lo, hi: jno.optimizers.mma(
            move=0.02, lower=lo, upper=hi, move_gamma=0.9985, move_min=0.001
        ),
    )
else:
    print("RELOCATE=0: nodes pinned, this is a density-only run", flush=True)

# --- design variable 1: one density per tet ------------------------------------------------------
u, phi = d.fem_symbols(value_shape=(3,), **({"space": "cover"} if SPACE == "cover" else {}))
_r, s = d.fem_symbols(space="P0", names=("r", "s"))
rho = jno.np.parameter(s, name="rho")
rho.dtype(jnp.float64)
if _warm is not None:
    _rho0 = jnp.asarray(np.asarray(_warm["rho_f"], dtype=float))
    rho.initialize(lambda k, sh, dtype=None, _v=_rho0: _v.reshape(sh))
else:
    rho.initialize(jax.nn.initializers.constant(VOLFRAC))
rho.optimizer(jno.optimizers.mma(move=0.15, lower=1e-3, upper=1.0, move_gamma=0.9985, move_min=0.0075))

# Solid, non-designable material under the load and at the clamp: without it the optimiser empties
# the loaded elements and the compliance measures that artefact rather than the structure.
cen0 = pts0[cells].mean(axis=1)
solid = ((cen0[:, 0] > L - 2 * H_MESH) & (cen0[:, 1] < ARM)) | (cen0[:, 1] > L - 2 * H_MESH)
pmask = jnp.asarray(solid)
print(f"non-design solid: {int(solid.sum())} of {NT} tets", flush=True)

patch = d.patch_filter()  # eq. (17)-(19), edge fans in 3-D


def filtered(r):
    """Patch filter, with the non-design region forced solid. Non-local, so it reparameterises."""
    return jnp.where(pmask, 1.0, patch(r))


rho.constrain(filtered)

# The filter reparameterises (non-local); the projection composes in the trace (pointwise) so that
# `beta` is a TRACED argument. Reading beta inside `constrain` instead makes it a trace-time
# constant and the ramp becomes decorative -- the failure `bracket.py` documents at length.
beta_h_p = jno.np.parameter((1,), name="beta_h")
beta_h_p.dtype(jnp.float64)
beta_h_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, 1.0))
beta_h_p.optimizer(optax.sgd(1.0))
rho_phys = rho.project(beta_h_p, eta=0.5) if BETA_H > 0 else rho

# --- the elasticity problem ----------------------------------------------------------------------
eps = lambda w: symgrad(w, [xi, yi, zi])  # noqa: E731
a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731
E = lambda r: EMIN + r**PENAL * (E0 - EMIN)  # noqa: E731

# The load coefficient is scaled so the ASSEMBLED resultant is exactly unit -- measured, not
# assumed. A band narrower than the mesh spacing contains no whole facet and integrates to exactly
# zero, silently; and a coefficient guessed from a node count is wrong because the term is an
# integral (that mistake put a 7x load error between two 2-D benchmarks -- see paper2d.py).
_end_facets = M.boundary_facets(cells)
# `keep` is a predicate on the (n, 3, 3) facet-vertex array returning (n, 3): a facet is kept
# only when ALL THREE of its nodes are on the loaded face, which is what "a whole facet" means.
_A_end = M.facet_area(pts0, _end_facets, keep=lambda V: loaded(V[..., 0], V[..., 1], V[..., 2]))
if not np.isfinite(_A_end) or _A_end <= 0:
    raise SystemExit(f"the loaded end face has area {_A_end}: nothing to push on")
print(f"load face area {_A_end:.4f}", flush=True)

fem = jno.fem(
    [
        E(rho_phys) * a(eps(u), eps(phi)),
        u(*clamp_v) - (0.0, 0.0, 0.0),
        -1.0 * inner(jnp.asarray(PULL / _A_end), phi.bind(**dict(zip("xyz", end_v))), 1),
    ],
    quad_degree=2,
)

_probe = {"rho": jnp.full(NT, VOLFRAC)}
if BETA_H > 0:
    _probe["beta_h"] = jnp.asarray(1.0)
_Aop, _b = fem.operator.evaluate(_probe)
_f = np.asarray(jnp.asarray(_b)).reshape(-1, 3)
_res = _f.sum(axis=0)
_mag = float(np.linalg.norm(_res))
if not np.isfinite(_mag) or _mag < 1e-12:
    raise SystemExit(f"the load band assembled a zero resultant {_res} -- it caught no whole facet")
print(f"load |f| = {_mag:.4f}   resultant = {np.round(_res, 4)}   "
      f"unit = {np.round(_res / _mag, 3)}   want {np.round(PULL, 3)}", flush=True)

# --- objective and constraints -------------------------------------------------------------------
cellv = d.cell_volume()
compliance = (E(rho_phys) * a(eps(u), eps(u))).integrate(fem, solver=pardiso_lu).name("C")
volume = C.material_budget(rho_phys, cellv, pts0, cells, VOLFRAC).name("V")
G = C.mesh_quality(d, pts0, cells, theta_min_deg=15.0, aspect_max=8.0, rho=rho_phys.reshape(-1))
perim = rho_phys.perimeter(zeta=0.1)

terms, callbacks = [compliance], []
if BETA_H > 0:
    callbacks.append(
        jno.optimizers.heaviside_continuation(
            beta_h_p, start=1.0, maximum=BETA_H, over=max(1, int(BETA_AT * ITERS))
        )
    )
if PSTAR > 0:
    beta_p = jno.np.parameter((1,), name="beta")
    beta_p.dtype(jnp.float64)
    beta_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, BETA))
    beta_p.optimizer(optax.sgd(1.0))
    terms.append((beta_p[0] * perim.log_barrier(PSTAR)).name("R"))
    callbacks.append(jno.optimizers.geometric_decay(beta_p, GAMMA, start=BETA, minimum=BETA_MIN))
terms.append(jno.le(volume, 1.0))
if RELOCATE:
    # On a pinned mesh these are constants of the mesh (and g_vmx is infeasible on one) -- the
    # trap bracket.py documents. A fixed-mesh baseline carries the volume budget alone.
    terms += [jno.le(g, 1.0) for g in (G["g_ang"], G["g_asp"], G["g_vmx"], G["g_vmn"])]
else:
    print("RELOCATE=0: the 4 element-quality rows are dropped -- constants of a fixed mesh",
          flush=True)

crux = jno.core(terms, domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
ev = lambda n: float(np.asarray(crux.eval([n])).reshape(-1)[0])  # noqa: E731

print("=" * 92, flush=True)
print(f"  L-BRACKET 3-D  {L}x{L}x{THICK}, arm {ARM}  h={H_MESH}  tets={NT}  dofs={fem.dofs}", flush=True)
print(f"        volfrac={VOLFRAC}  SPACE={SPACE}  RELOCATE={RELOCATE}  pull={np.round(PULL,3)}  "
      f"PSTAR={PSTAR}  BETA_H={BETA_H}  ITERS={ITERS}  TAG={TAG}", flush=True)
print("=" * 92, flush=True)

DST = os.path.join(OUT, f"topo3d_{TAG}.npz")
# Mandatory. Catches a boundary node drifting INTO a BC zone, which changes the support set with
# no exception and no visibly wrong number -- only an independent re-solve disagrees.
guard = CO.DriftGuard(pts0, cells, _ZONES) if RELOCATE else None


def coords_now():
    """The mesh as it stands. Rebuilt per checkpoint, not cached -- the nodes are design vars."""
    p = pts0.copy()
    for spec in getattr(d, "_trainable_coords", []) or []:
        p[np.asarray(spec["ids"], dtype=int), int(spec["axis"])] = np.asarray(
            crux.eval([spec["expr"]])
        ).reshape(-1)
    return p


def checkpoint(rho_f, pts_f, drift, hist, extra):
    """Write the design atomically, every chunk.

    Every chunk, not once at the end, for two reasons measured the hard way. A queue kill or an
    OOM used to lose the whole run -- this script wrote its npz exactly once, after the loop, so a
    23-hour job that died at hour 22 left nothing. And the BC-drift check used to `SystemExit`
    BEFORE the save, so the one failure mode most worth inspecting destroyed its own evidence.
    Now the design is on disk first and stamped `bc_drift`, and the caller decides whether to stop.

    `tmp` + `os.replace` because a kill DURING the write would otherwise corrupt the previous
    good checkpoint too -- `os.replace` is atomic within a filesystem, so the file on disk is
    always one complete checkpoint or the other, never a half of one.
    """
    H = np.array(hist)
    tmp = DST + ".tmp.npz"
    np.savez_compressed(
        tmp,
        cells=cells, pts0=pts0, pts_f=pts_f, rho_f=rho_f, solid=solid,
        hist=H,                       # kept: the lb3p_* npz already on disk are read with this
        # Canonical per-iteration columns, the layout `cluster_report.py` and `export_console.py`
        # expect from `bracket.py`. Same numbers, addressable by name.
        it=H[:, 0], C=H[:, 1], V=H[:, 2], mnd=H[:, 3],
        P=np.asarray(extra["P"]), ang=np.asarray(extra["ang"]),
        asp=np.asarray(extra["asp"]), vmx=np.asarray(extra["vmx"]),
        load_f=_res, space=SPACE, pull=PULL,
        # Names the problem so `reanalysis_boundary.py` takes the supports and the load from the
        # design rather than its own defaults -- the desync topopt/__init__.py flags as live.
        problem="lbracket3d",
        bc_drift=np.asarray(drift, dtype=int),
    )
    os.replace(tmp, DST)


t0, it, hist = time.time(), 0, []
extra = {k: [] for k in ("P", "ang", "asp", "vmx")}
while time.time() - t0 < BUDGET and it < ITERS:
    crux.solve(CHUNK, callbacks=callbacks)
    it += CHUNK
    rf = np.asarray(crux.eval([rho_phys])).reshape(-1)
    hist.append((it, ev(compliance), ev(volume), float(np.mean(4 * rf * (1 - rf)))))
    # Evaluated for the print line anyway; recording them costs nothing and is what lets the
    # report and the console read an L-bracket run at all.
    for k, node in (("P", perim), ("ang", G["g_ang"]), ("asp", G["g_asp"]), ("vmx", G["g_vmx"])):
        extra[k].append(ev(node))
    print(f"  it {it:4d}  C={hist[-1][1]:10.4f}  V={hist[-1][2]:.4f}  M_nd={hist[-1][3]:.4f}  "
          f"P={extra['P'][-1]:8.2f}  g_ang={extra['ang'][-1]:.3f} asp={extra['asp'][-1]:.3f} "
          f"vmx={extra['vmx'][-1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    pts_f = coords_now()
    drift = guard.check(pts_f) if guard is not None else (0, 0, 0)
    checkpoint(rf, pts_f, drift, hist, extra)
    if drift[2]:
        # Checked every chunk, not once at the end: this finds the fault ~10 chunks earlier, and
        # the design that was still valid up to this chunk survives on disk with the flag on it.
        print(f"BC drift at it {it}: {drift[0]} nodes in the BC zones at setup, {drift[1]} now, "
              f"{drift[2]} changed", flush=True)
        raise SystemExit(
            f"{drift[2]} boundary node(s) entered or left a BC zone during the run, so the "
            f"supports no longer describe the geometry their predicate names -- the reported "
            f"compliance is not trustworthy. The checkpoint IS saved and stamped bc_drift, so the "
            f"design can be inspected. Widen the freeze margin (currently "
            f"{float(np.sqrt(3.0)) * MOVE_BOUND:.4f})."
        )

print(f"BC drift: {drift[0]} boundary nodes in the BC zones at setup, {drift[1]} at convergence, "
      f"{drift[2]} changed", flush=True)
print(f"saved topo3d_{TAG}.npz  ({it} iterations, {(time.time() - t0) / 60:.1f} min)", flush=True)
