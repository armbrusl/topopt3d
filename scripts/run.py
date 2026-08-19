"""The 30-minute 3-D topology-optimisation run: 64,816 tetrahedra, deformable mesh, PARDISO.

Design vector is [rho, d_x, d_y, d_z]. Length-scale control is ON -- a perimeter (in 3-D an AREA)
barrier plus SIMP penalisation continuation -- because without a length-scale restriction the
continuum problem is not well posed and the design fragments at any resolution.

PSTAR=0 runs the PROBE: no barrier, so the natural perimeter can be measured and the target set
from it rather than guessed. The 2-D tutorial's 650 is a LENGTH and means nothing here.

Budgeted by wall clock, not by iteration count, so the run lands inside the budget whatever the
machine does.
"""

import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace

H_MESH = float(os.environ.get("HMESH", "0.10"))
PSTAR = float(os.environ.get("PSTAR", "0"))
BUDGET = float(os.environ.get("BUDGET", "1680"))          # seconds of design loop (28 min)
CHUNK = int(os.environ.get("CHUNK", "10"))
TAG = os.environ.get("TAG", "final")

L, H, W = 4.0, 2.0, 2.0
E0, EMIN, NU, PENAL, VOLFRAC = 1.0, 1e-4, 0.3, 3.0, 0.20
LAM, MU = E0 * NU / ((1 + NU) * (1 - 2 * NU)), E0 / (2 * (1 + NU))
THETA_MIN, ASPECT_MAX, PN, TOL = np.deg2rad(5.0), 8.0, 50.0, 1e-9
MOVE_BOUND = 0.30 * H_MESH
SPAN = 0.6                                                 # loaded band; also the solid pmask depth
BETA, GAMMA, BETA_MIN = 2e-4, 0.997, 1e-4

d = jno.Shape.box(0, 0, 0, L, H, W, size=H_MESH).domain()
cells = np.asarray(d._cells_p1())
pts0 = np.asarray(d.mesh.points)[:, :3]
NT = cells.shape[0]
print(f"tets={NT}  nodes={pts0.shape[0]}  dofs={3 * pts0.shape[0]}  PSTAR={PSTAR}", flush=True)

# --- design variables 2-4: interior nodal coordinates, per-node box bounds --------------------
MOVING = os.environ.get("MOVING", "1") == "1"
if MOVING:
  xm, ym, zm, _ = d.variable("mv", where=lambda x, y, z: (
      (x > TOL) & (x < L - TOL) & (y > TOL) & (y < H - TOL) & (z > TOL) & (z < W - TOL)), split=True)
  xm.trainable(name="mesh_x"), ym.trainable(name="mesh_y"), zm.trainable(name="mesh_z")
  ids = np.asarray(d._trainable_coords[0]["ids"], dtype=int)
  for spec in d._trainable_coords:
      c0 = pts0[ids, int(spec["axis"])]
      spec["expr"].optimizer(jno.optimizers.mma(
          move=0.02, lower=c0 - MOVE_BOUND, upper=c0 + MOVE_BOUND, move_gamma=0.9985, move_min=0.001))

# --- design variable 1: one density per tetrahedron -------------------------------------------
u, phi = d.fem_symbols(value_shape=(3,))
_r, s = d.fem_symbols(space="P0", names=("r", "s"))
rho = jno.np.parameter(s, name="rho")
rho.dtype(jnp.float64)
rho.initialize(jax.nn.initializers.constant(VOLFRAC))
rho.optimizer(jno.optimizers.mma(move=0.15, lower=1e-3, upper=1.0, move_gamma=0.9985, move_min=0.0075))

# penal as a RUNTIME parameter, so the continuation raises it without a recompile
USE_PP = os.environ.get("PENALP", "1") == "1"
if USE_PP:
    penal_p = jno.np.parameter((1,), name="penal")
    penal_p.dtype(jnp.float64)
    penal_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, PENAL))
    penal_p.optimizer(optax.sgd(1.0))

# Solid, non-designable material under the load. Without it the optimiser empties the loaded
# elements and the compliance measures that artefact rather than the structure.
cen0 = pts0[cells].mean(axis=1)
pmask = jnp.asarray((cen0[:, 0] > L - SPAN) & (cen0[:, 2] < SPAN))
patch = d.patch_filter()
# The physical density is FLOORED at RHO_MIN. With `penal` a differentiable parameter the SIMP
# exponent contributes d/dp (rho^p) = rho^p * ln(rho), which is 0 * (-inf) = NaN wherever the
# physical density is exactly zero -- and the patch filter reaches exactly zero (a fan whose
# product vanishes, or an element with no fan of three). Measured: without this floor the very
# first solve assembles a stiffness containing NaN, at every mesh size, whenever `penal` is a
# parameter; with a literal exponent the same design is fine, because there is no derivative in p.
RHO_MIN = 1e-3
_phys = (lambda r: jnp.where(pmask, 1.0, patch(r))) if os.environ.get("PMASK", "1") == "1" else patch
rho.constrain(lambda r: jnp.clip(_phys(r), RHO_MIN, 1.0))
print(f"pmask: {int(np.asarray(pmask).sum())} solid non-designable tets", flush=True)

xi, yi, zi, _ = d.variable("interior", split=True)
root = d.variable("root", where=lambda x, y, z: x < TOL, split=True)[:3]
tip = d.variable("tip", where=lambda x, y, z: (x > L - TOL) & (z < SPAN), split=True)[:3]
eps = lambda w: symgrad(w, [xi, yi, zi])  # noqa: E731
a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731
USE_PM = os.environ.get("PMASK", "1") == "1"
E = (lambda r: EMIN + r**penal_p * (E0 - EMIN)) if USE_PP else (lambda r: EMIN + r**PENAL * (E0 - EMIN))  # noqa: E731
fem = jno.fem([E(rho) * a(eps(u), eps(phi)), u(*root) - (0.0, 0.0, 0.0),
               -1.0 * inner(jnp.array([0.0, 0.0, -1.0]), phi.bind(**dict(zip("xyz", tip))), 1)],
              quad_degree=2)
_A, _b = fem.operator.evaluate({"rho": jnp.full(NT, VOLFRAC), "penal": jnp.array([PENAL])})
_load = float(np.abs(np.asarray(jnp.asarray(_b)).reshape(-1)).sum())
assert _load > 1e-9, f"traction band assembled |f| = {_load:.2e}; widen SPAN past the mesh spacing"
print(f"load |f| = {_load:.4f}", flush=True)

cellv = d.cell_volume()
_BK = os.environ.get("SOLVER", "pardiso")
compliance = (E(rho) * a(eps(u), eps(u))).integrate(fem, solver=(jno.solve.lu(backend=_BK) if _BK else None))
volume = (rho.reshape(-1) * cellv).sum / (VOLFRAC * cellv.sum)
g_ang = ((2 * np.pi - d.cell_angles()) / (2 * np.pi - THETA_MIN)).reshape(-1).pnorm(PN, normalize=True)
g_asp = (d.cell_aspect() / ASPECT_MAX).pnorm(PN, normalize=True)
perim = rho.perimeter(zeta=0.1)

terms = [compliance]
callbacks = [jno.optimizers.simp_continuation(penal_p, rho, physical=patch, every=25, watch=0, patience=6)] \
    if USE_PP else []   # the callback drives penal_p; without it in the graph it has no layer to write
if PSTAR > 0:
    beta_p = jno.np.parameter((1,), name="beta")
    beta_p.dtype(jnp.float64)
    beta_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, BETA))
    beta_p.optimizer(optax.sgd(1.0))
    terms.append((beta_p[0] * perim.log_barrier(PSTAR)).name("R"))
    callbacks.append(jno.optimizers.geometric_decay(beta_p, GAMMA, start=BETA, minimum=BETA_MIN))
terms += [jno.le(volume, 1.0)] + ([jno.le(g_ang, 1.0), jno.le(g_asp, 1.0)] if MOVING else [])
crux = jno.core(terms, domain=jno.domain.from_array({"_": np.zeros((1, 1))}))

coord_specs = list(getattr(d, "_trainable_coords", []) or [])
coord_nodes = [sp["expr"] for sp in coord_specs]
hist = {k: [] for k in ("it", "C", "V", "P", "mnd", "ang", "asp", "penal")}
frames_rho, frames_pts = [], []
ev = lambda n: float(np.asarray(crux.eval([n])).reshape(-1)[0])  # noqa: E731

it, t0 = 0, time.time()
while time.time() - t0 < BUDGET:
    crux.solve(CHUNK, callbacks=callbacks)
    it += CHUNK
    r = np.asarray(crux.eval([rho])).reshape(-1)
    p = pts0.copy()
    for sp, node in zip(coord_specs, coord_nodes):
        p[np.asarray(sp["ids"], dtype=int), int(sp["axis"])] = np.asarray(crux.eval([node])).reshape(-1)
    for k, v in (("it", it), ("C", ev(compliance)), ("V", ev(volume)), ("P", ev(perim)),
                 ("mnd", float(4.0 * np.mean(r * (1.0 - r)))), ("ang", ev(g_ang)),
                 ("asp", ev(g_asp)), ("penal", float(np.asarray(crux.eval([penal_p])).reshape(-1)[0]) if USE_PP else PENAL)):
        hist[k].append(v)
    frames_rho.append(r.copy())
    frames_pts.append(p.copy())
    print(f"  it {it:4d}  C={hist['C'][-1]:9.3f}  V={hist['V'][-1]:.4f}  P={hist['P'][-1]:9.2f}  "
          f"M_nd={hist['mnd'][-1]:.4f}  penal={hist['penal'][-1]:.2f}  "
          f"g_ang={hist['ang'][-1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)

np.savez(os.path.join(OUT, f"topo3d_{TAG}.npz"), cells=cells, pts0=pts0,
         pts_f=frames_pts[-1], rho_f=frames_rho[-1],
         frames_rho=np.array(frames_rho), frames_pts=np.array(frames_pts),
         **{k: np.asarray(v) for k, v in hist.items()})
print(f"saved topo3d_{TAG}.npz  ({len(frames_rho)} frames, {it} iterations, "
      f"{(time.time() - t0) / 60:.1f} min)", flush=True)
