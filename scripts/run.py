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
import resource
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")
os.makedirs(OUT, exist_ok=True)
inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace

H_MESH = float(os.environ.get("HMESH", "0.10"))
PSTAR = float(os.environ.get("PSTAR", "0"))
BUDGET = float(os.environ.get("BUDGET", "1680"))          # seconds of design loop (28 min)
# ITERS caps the iteration count as well, for a run that has to be COMPARABLE rather than merely fit
# a wall-clock box: budgeting alone means a faster machine silently runs a different number of design
# steps. 0 leaves the run budget-bound as before; whichever limit is hit first ends it.
ITERS = int(os.environ.get("ITERS", "0"))
CHUNK = int(os.environ.get("CHUNK", "10"))
TAG = os.environ.get("TAG", "final")

# Geometry and load direction are configurable so the SAME code can run a genuine 3-D box or a
# PSEUDO-2D slab -- a domain a couple of elements thick with its z-faces traction-free, which is
# plane-stress in all but name. That case matters because its answer is known: it must reproduce the
# classic 2-D cantilever truss, so it separates "3-D is hard" from "the machinery is wrong".
L = float(os.environ.get("LX", "4.0"))
H = float(os.environ.get("LY", "2.0"))
W = float(os.environ.get("LZ", "2.0"))
LOAD_AXIS = int(os.environ.get("LOAD_AXIS", "2"))   # 0=x, 1=y, 2=z; the 2-D benchmark loads -y
E0, NU, PENAL = 1.0, 0.3, 3.0
EMIN = float(os.environ.get("EMIN", "1e-4"))
VOLFRAC = float(os.environ.get("VOLFRAC", "0.20"))
LAM, MU = E0 * NU / ((1 + NU) * (1 - 2 * NU)), E0 / (2 * (1 + NU))
THETA_MIN, ASPECT_MAX, PN, TOL = np.deg2rad(5.0), 8.0, 50.0, 1e-9
# The paper's mechanism is that nodes MIGRATE TO THE MATERIAL BOUNDARY, so the bound on how far
# they may travel is not a safety knob -- it is the method's resolution. The 2-D tutorial uses
# 2.0*h; this was 0.30*h, i.e. 6.7x tighter, and measurably throttled: at 80,716 tets the median
# interior node had used 64 % of its allowance and the top decile was pinned against it, while
# nodes near the rho=0.5 interface still moved 1.37x further than nodes deep in void or solid --
# the mechanism working, against a wall.
MOVE_BOUND = float(os.environ.get("MOVE_BOUND_FRAC", "2.0")) * H_MESH
SPAN = float(os.environ.get("SPAN", "0.6"))   # loaded band; also the solid pmask depth
# Barrier schedule, eq. (39)-(41). BETA_MIN is a FLOOR on the barrier weight, and it decides whether
# the perimeter target still binds at convergence: with the paper's 2e-4 -> 1e-4 decay the 440-it
# pseudo-2D run ended at P = 1305.7 against PSTAR = 1300, i.e. the target was overrun because the
# barrier had decayed out of the way. Raising the floor keeps the length scale under control to the
# end; the trade is eq. (41)'s point, that a barrier still active at convergence distorts the optimum.
BETA = float(os.environ.get("BETA0", "2e-4"))
GAMMA = float(os.environ.get("GAMMA", "0.997"))
BETA_MIN = float(os.environ.get("BETA_MIN", "1e-4"))

d = jno.Shape.box(0, 0, 0, L, H, W, size=H_MESH).domain()
cells = np.asarray(d._cells_p1())
pts0 = np.asarray(d.mesh.points)[:, :3]
NT = cells.shape[0]
# Echo the RESOLVED configuration, loudly. Every knob is an env var, so a missing shell space --
# `CUDA_VISIBLE_DEVICES=1HMESH=0.17` sets neither -- silently runs a different problem than intended.
# Observed: that exact typo ran 64,816 tets instead of 14,063 and nobody noticed for 14 minutes.
_cfg = dict(volfrac=VOLFRAC, span=SPAN, box=f"{L}x{H}x{W}", load=f"-{'xyz'[LOAD_AXIS]}", HMESH=H_MESH, tets=NT, dofs=3 * pts0.shape[0], SOLVER=(os.environ.get("SOLVER", "custom") or "<jNO default>"),
            PSTAR=PSTAR, BUDGET=BUDGET, MOVING=os.environ.get("MOVING", "1"),
            PENALP=os.environ.get("PENALP", "1"), BETA_H=os.environ.get("BETA_H", "0"), TAG=TAG)
print("=" * 78, flush=True)
print("  CONFIG  " + "  ".join(f"{k}={v}" for k, v in _cfg.items()), flush=True)
for _k in ("HMESH", "PSTAR", "BUDGET", "SOLVER"):
    if _k not in os.environ:
        print(f"  NOTE    {_k} not set in the environment -- using the default above.", flush=True)
print("=" * 78, flush=True)

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
pmask = jnp.asarray((cen0[:, 0] > L - SPAN) & (cen0[:, 1 if LOAD_AXIS == 1 else 2] < SPAN))
patch = d.patch_filter()
# The physical density is FLOORED at RHO_MIN. With `penal` a differentiable parameter the SIMP
# exponent contributes d/dp (rho^p) = rho^p * ln(rho), which is 0 * (-inf) = NaN wherever the
# physical density is exactly zero -- and the patch filter reaches exactly zero (a fan whose
# product vanishes, or an element with no fan of three). Measured: without this floor the very
# first solve assembles a stiffness containing NaN, at every mesh size, whenever `penal` is a
# parameter; with a literal exponent the same design is fine, because there is no derivative in p.
RHO_MIN = 1e-3
_phys = (lambda r: jnp.where(pmask, 1.0, patch(r))) if os.environ.get("PMASK", "1") == "1" else patch

# HEAVISIDE PROJECTION -- Wang, Lazarov & Sigmund, Struct. Multidisc. Optim. 43 (2011) 767-784,
# after Guest, Prevost & Belytschko (2004). A manufactured part is one material, so intermediate
# density is a relaxation, not a physical state: it exists because a hard 0/1 has zero gradient
# everywhere and MMA would never move. This is the smoothed step that recovers the intent --
#
#     rho_bar = [tanh(beta*eta) + tanh(beta*(rho - eta))] / [tanh(beta*eta) + tanh(beta*(1 - eta))]
#
# differentiable for any finite beta, and -> a step as beta grows. beta is RAMPED (BETA_H is the
# final value, reached geometrically) because projecting hard from iteration 0 freezes the topology
# before it forms: the gradient away from the threshold vanishes and no element can cross back.
BETA_H = float(os.environ.get("BETA_H", "0"))
ETA_H = float(os.environ.get("ETA_H", "0.5"))
_beta_now = [1.0]   # ramped by the callback below; a list so the closure sees the update


def _project(x, b, eta=ETA_H):
    tb = jnp.tanh(b * eta)
    return (tb + jnp.tanh(b * (x - eta))) / (tb + jnp.tanh(b * (1.0 - eta)))


if BETA_H > 0:
    rho.constrain(lambda r: jnp.clip(_project(_phys(r), _beta_now[0]), RHO_MIN, 1.0))
else:
    rho.constrain(lambda r: jnp.clip(_phys(r), RHO_MIN, 1.0))
print(f"pmask: {int(np.asarray(pmask).sum())} solid non-designable tets", flush=True)

xi, yi, zi, _ = d.variable("interior", split=True)
root = d.variable("root", where=lambda x, y, z: x < TOL, split=True)[:3]
tip = d.variable("tip", where=lambda x, y, z: (x > L - TOL) & ((y if LOAD_AXIS == 1 else z) < SPAN), split=True)[:3]
_LOADVEC = jnp.array([-1.0 if k == LOAD_AXIS else 0.0 for k in range(3)])
eps = lambda w: symgrad(w, [xi, yi, zi])  # noqa: E731
a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731
USE_PM = os.environ.get("PMASK", "1") == "1"
E = (lambda r: EMIN + r**penal_p * (E0 - EMIN)) if USE_PP else (lambda r: EMIN + r**PENAL * (E0 - EMIN))  # noqa: E731
fem = jno.fem([E(rho) * a(eps(u), eps(phi)), u(*root) - (0.0, 0.0, 0.0),
               -1.0 * inner(_LOADVEC, phi.bind(**dict(zip("xyz", tip))), 1)],
              quad_degree=2)
_A, _b = fem.operator.evaluate({"rho": jnp.full(NT, VOLFRAC), **({"penal": jnp.array([PENAL])} if USE_PP else {})})
_f = np.asarray(jnp.asarray(_b)).reshape(-1, 3)
_load = float(np.abs(_f).sum())
assert _load > 1e-9, f"traction band assembled |f| = {_load:.2e}; widen SPAN past the mesh spacing"
# Check the ASSEMBLED resultant against LOAD_AXIS rather than trusting the banner. `_LOADVEC` used to
# be computed here and then ignored, the traction vector below it being a hard-coded -z: every
# LOAD_AXIS=1 run was pulled through the slab thickness while the banner reported `load=-y`.
_res = _f.sum(axis=0)
assert abs(_res[LOAD_AXIS]) > 0.9 * np.abs(_res).sum(), (
    f"assembled resultant {_res} is not along -{'xyz'[LOAD_AXIS]}")
print(f"load |f| = {_load:.4f}   resultant = ({_res[0]:+.3f}, {_res[1]:+.3f}, {_res[2]:+.3f})", flush=True)

cellv = d.cell_volume()
# SOLVER="custom" is the recommended setting: MKL PARDISO through scripts/pardiso_solver.py, which
# bypasses jNO's own pardiso backend (that one aborts the process whenever the patch filter is on --
# see the README). Measured against the default backend, per design iteration, with the filter ON:
# 2.3x at 8,802 tets, 3.2x at 14,063, 6.4x at 24,787, and 4.26 s/it at 64,816 where the default
# needs ~74 s. Compliance agrees to 4-5 digits at every size.
_BK = os.environ.get("SOLVER", "custom")
if _BK == "custom":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pardiso_solver import pardiso_lu as _spec
else:
    _spec = jno.solve.lu(backend=_BK) if _BK else None
compliance = (E(rho) * a(eps(u), eps(u))).integrate(fem, solver=_spec)
volume = (rho.reshape(-1) * cellv).sum / (VOLFRAC * cellv.sum)
g_ang = ((2 * np.pi - d.cell_angles()) / (2 * np.pi - THETA_MIN)).reshape(-1).pnorm(PN, normalize=True)
g_asp = (d.cell_aspect() / ASPECT_MAX).pnorm(PN, normalize=True)

# eq. (26) and (28): the element-volume bounds, and the DENSITY COUPLING in `(2 - rho)` is the
# point of them. A void element may grow up to twice V0MAX; a solid one may not. That is what makes
# the mesh coarsen where there is no material and stay fine where there is -- i.e. it is the half of
# the deformable-mesh method that concentrates resolution on the structure. A plain aspect-ratio
# constraint has no density coupling and cannot do this, which is why leaving these out produced a
# solid wedge with no internal structure.
#
# The bounds are RESCALED for tetrahedra. The paper's [0.05, 0.75]*h^2 are [0.116, 1.73]x the
# equilateral triangle's area (sqrt(3)/4); the regular tet's volume is 1/(6*sqrt(2)) = 0.1179*h^3,
# so the same relative window is [0.0137, 0.204]*h^3. Copying 0.05/0.75 across would give
# [0.42, 6.4]x nominal -- a tight floor and a ceiling that never binds.
# Calibrated from THIS mesh, not from the regular tetrahedron. The paper's 2-D bounds are quoted
# as multiples of the equilateral triangle, but a Delaunay TET mesh is far less uniform than a
# triangulation: measured here the median cell is 1.26x the regular tet and the largest is 5.80x.
# Normalising by the ideal element therefore made 27.5 %% of cells violate V0MAX at rho=1 on the
# INITIAL mesh -- an infeasible constraint from iteration 0, which MMA answers by abandoning the
# volume constraint (observed: V fell to 0.90 while vmx sat at 1.36). Anchoring on the mesh's own
# extremes starts feasible and still does the job, which is to stop elements GROWING.
_v0 = np.abs(np.linalg.det(np.stack([pts0[cells][:, i + 1] - pts0[cells][:, 0] for i in range(3)], -1))) / 6.0
V0MAX = float(os.environ.get("V0MAX_FRAC", "1.30")) * float(np.percentile(_v0, 99))
V0MIN = float(os.environ.get("V0MIN_FRAC", "0.50")) * float(np.percentile(_v0, 1))
rho_e = rho.reshape(-1)
g_vmx = (cellv / ((2.0 - rho_e) * V0MAX)).pnorm(PN, normalize=True)
g_vmn = ((2 * V0MAX - cellv) / (2 * V0MAX - V0MIN)).pnorm(PN, normalize=True)
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
terms += [jno.le(volume, 1.0)] + ([jno.le(g_ang, 1.0), jno.le(g_asp, 1.0),
                                   jno.le(g_vmx, 1.0), jno.le(g_vmn, 1.0)] if MOVING else [])
crux = jno.core(terms, domain=jno.domain.from_array({"_": np.zeros((1, 1))}))

coord_specs = list(getattr(d, "_trainable_coords", []) or [])
coord_nodes = [sp["expr"] for sp in coord_specs]
hist = {k: [] for k in ("it", "C", "V", "P", "mnd", "ang", "asp", "vmx", "vmn", "penal")}
frames_rho, frames_pts = [], []
ev = lambda n: float(np.asarray(crux.eval([n])).reshape(-1)[0])  # noqa: E731

DST = os.path.join(OUT, f"topo3d_{TAG}.npz")


def checkpoint():
    """Write the run to disk. Called EVERY chunk, not just at the end.

    A budgeted run holds everything in memory and used to save only after the loop, so anything
    that stopped it early -- a kill, a queue timeout, a full disk -- threw the whole run away.
    Two 29-minute runs were lost that way. The write is ~5 MB and a fraction of a second against
    a chunk that costs tens of seconds, so it is not worth striding. Save to a temporary file and
    rename, so a kill DURING the write cannot corrupt the previous checkpoint.
    """
    tmp = DST + ".tmp.npz"
    np.savez(tmp, cells=cells, pts0=pts0, pts_f=frames_pts[-1], rho_f=frames_rho[-1],
             frames_rho=np.array(frames_rho), frames_pts=np.array(frames_pts),
             **{k: np.asarray(v) for k, v in hist.items()})
    os.replace(tmp, DST)


it, t0 = 0, time.time()
while time.time() - t0 < BUDGET and (ITERS == 0 or it < ITERS):
    crux.solve(CHUNK, callbacks=callbacks)
    it += CHUNK
    if BETA_H > 0:  # geometric ramp 1 -> BETA_H over the run, so the topology forms before it sets
        # Tied to the ITERATION count, not to how the run is chunked. Ramping once per
        # `crux.solve()` call made CHUNK a physics knob: at CHUNK=10 beta reached 10.8 by
        # iteration 250, at CHUNK=50 only 1.6, and the two runs converged to different
        # designs (M_nd 0.008 vs 0.078) while looking like a pure performance comparison.
        _beta_now[0] = min(BETA_H, 1.10 ** (it / 10.0))
    r = np.asarray(crux.eval([rho])).reshape(-1)
    p = pts0.copy()
    for sp, node in zip(coord_specs, coord_nodes):
        p[np.asarray(sp["ids"], dtype=int), int(sp["axis"])] = np.asarray(crux.eval([node])).reshape(-1)
    for k, v in (("it", it), ("C", ev(compliance)), ("V", ev(volume)), ("P", ev(perim)),
                 ("mnd", float(4.0 * np.mean(r * (1.0 - r)))), ("ang", ev(g_ang)),
                 ("asp", ev(g_asp)), ("vmx", ev(g_vmx) if MOVING else 0.0), ("vmn", ev(g_vmn) if MOVING else 0.0), ("penal", float(np.asarray(crux.eval([penal_p])).reshape(-1)[0]) if USE_PP else PENAL)):
        hist[k].append(v)
    frames_rho.append(r.copy())
    frames_pts.append(p.copy())
    print(f"  it {it:4d}  C={hist['C'][-1]:9.3f}  V={hist['V'][-1]:.4f}  P={hist['P'][-1]:9.2f}  "
          f"M_nd={hist['mnd'][-1]:.4f}  penal={hist['penal'][-1]:.2f}  beta_H={_beta_now[0]:.1f}  "
          f"g_ang={hist['ang'][-1]:.3f} vmx={hist['vmx'][-1]:.3f} vmn={hist['vmn'][-1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    checkpoint()

print(f"saved topo3d_{TAG}.npz  ({len(frames_rho)} frames, {it} iterations, "
      f"{(time.time() - t0) / 60:.1f} min)", flush=True)

# How many cores did this actually use? CPU-seconds over wall-seconds is the honest answer, and it
# is the number to look at before asking for a bigger allocation: requesting 20 cores for a job that
# sustains 3 of them wastes 17. Counts the children too, since MKL and XLA both thread underneath.
_r, _rc = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
_cpu = _r.ru_utime + _r.ru_stime + _rc.ru_utime + _rc.ru_stime
_wall = time.time() - t0
print(f"cpu {_cpu:.0f} s over {_wall:.0f} s wall  ->  {_cpu / _wall:.1f} cores busy on average "
      f"(machine has {os.cpu_count()})", flush=True)
if _BK == "custom":
    from pardiso_solver import STATS
    print(f"pardiso: {STATS['n']} solves, {STATS['t']:.1f} s total = {100 * STATS['t'] / _wall:.2f} % "
          f"of wall ({1e3 * STATS['t'] / max(STATS['n'], 1):.1f} ms each); the rest is JAX-side "
          f"assembly, the patch filter, the geometry constraints, MMA and their gradients.", flush=True)
