"""Volume-to-point heat conduction: the case whose optimum is a TREE, not a truss.

Every case so far has been elasticity, and elasticity makes trusses -- members in tension and
compression, meeting at nodes. Conduction makes something structurally different: heat is generated
everywhere and must reach a single small sink, so the optimum is a **branching tree**, a trunk near
the sink splitting again and again into ever-finer limbs as it reaches out for the volume it drains.
That is the constructal problem of Bejan, *Int. J. Heat Mass Transfer* **40** (1997) 799-816, and its
density-based form is Gersborg-Hansen, Bendsoe & Sigmund, *Struct. Multidisc. Optim.* **31** (2006)
251-259.

It is worth having for two reasons beyond novelty.

* **The surfaces are different in kind.** A truss has cylindrical members and flat junctions; a tree
  has smoothly tapering trunks and saddle-shaped bifurcations. If a smoothness formulation only
  works on one of those, that is worth knowing before claiming it works at all.
* **Scalar physics buys resolution.** Temperature is one DOF per node against elasticity's three, so
  the same solve cost affords ~3x the elements -- and mesh resolution is the binding constraint on
  surface quality (measured on the bracket: a perfect sphere scores 3.6 deg at h=0.15 but 2.4 deg at
  h=0.10). This is the cheapest way to see what the method does when the mesh is not the limit.

    HMESH=0.05 python scripts/heattree.py          # ~25k tets, a first look
    HMESH=0.03 SMOOTH=2 python scripts/heattree.py # ~116k tets, the real one

Method: Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403 -- eq. (17)-(19) patch
filter (3-D: edge fans), eq. (21)/(24) dihedral bounds, eq. (26)/(28) density-coupled element-volume
bounds, eq. (38)-(41) perimeter barrier, plus the bending penalty of `rho.curvature()`.
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

inner = jno.np.inner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pardiso_solver import pardiso_lu  # noqa: E402

OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
os.makedirs(OUT, exist_ok=True)

L = 1.0  # the cube of heat-generating material
H_MESH = float(os.environ.get("HMESH", "0.05"))
# The conductivity CONTRAST is this problem's SIMP contrast, and it sets how tree-like the answer is:
# too little and the "tree" is a diffuse cloud, too much and the solve conditions badly. 1e-3 is the
# ratio the density-based conduction literature uses.
K0, KMIN, PENAL = 1.0, float(os.environ.get("KMIN", "1e-3")), 3.0
QGEN = 1.0  # uniform volumetric heat generation, INDEPENDENT of the design (constructal convention)
SINK = float(os.environ.get("SINK", "0.08"))  # half-width of the cold patch at the centre of z = 0
VOLFRAC = float(os.environ.get("VOLFRAC", "0.25"))  # trees want to be sparse
THETA_MIN, PN, TOL = np.deg2rad(5.0), 50.0, 1e-9
ASPECT_MAX = float(os.environ.get("ASPECT_MAX", "8.0"))
MOVE_BOUND = float(os.environ.get("MOVE_FRAC", "0.5")) * H_MESH
PSTAR = float(os.environ.get("PSTAR", "0"))
BETA, GAMMA = 1e-3, 0.997
BETA_MIN = float(os.environ.get("BETA_MIN", "5e-4"))
# Defaults to OFF: the projection drops volume as beta rises (V tracks beta exactly; measured 1.000
# -> 0.718 on the bracket at beta 1 -> 16, and V holds at 1.000 for 250 iterations with BETA_H=0).
BETA_H = float(os.environ.get("BETA_H", "0"))
ETA_H, RHO_MIN = 0.5, 1e-3
BETA_AT = float(os.environ.get("BETA_AT", "0.8"))
SMOOTH = float(os.environ.get("SMOOTH", "0"))
SMOOTH_P = float(os.environ.get("SMOOTH_P", "0"))
ITERS = int(os.environ.get("ITERS", "250"))
CHUNK = int(os.environ.get("CHUNK", "25"))
BUDGET = float(os.environ.get("BUDGET", "7200"))
TAG = os.environ.get("TAG", "heattree")

d = jno.Shape.box(0, 0, 0, L, L, L, size=H_MESH).domain()
cells = np.asarray(d._cells_p1())
pts0 = np.asarray(d.mesh.points)[:, :3]
NT = cells.shape[0]
print("=" * 92, flush=True)
print(f"  HEAT TREE  cube {L}^3  h={H_MESH}  tets={NT}  dofs={pts0.shape[0]} (scalar)", flush=True)
print(f"             volfrac={VOLFRAC}  k0/kmin={K0 / KMIN:.0f}  sink half-width={SINK}  TAG={TAG}", flush=True)
print("=" * 92, flush=True)

# --- movable nodes: everything not on the cube's own surface ------------------------------------
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
        jno.optimizers.mma(move=0.02, lower=c0 - MOVE_BOUND, upper=c0 + MOVE_BOUND, move_gamma=0.9985, move_min=0.001)
    )

# --- the design density -------------------------------------------------------------------------
T, phi = d.fem_symbols()  # SCALAR temperature: one dof per node, not three
_r, s = d.fem_symbols(space="P0", names=("r", "s"))
rho = jno.np.parameter(s, name="rho")
rho.dtype(jnp.float64)
rho.initialize(jax.nn.initializers.constant(VOLFRAC))
rho.optimizer(jno.optimizers.mma(move=0.15, lower=1e-3, upper=1.0, move_gamma=0.9985, move_min=0.0075))

# SIMP continuation, which the 2-D tutorial has and the 3-D scripts did not. Conduction discretises
# far more reluctantly than elasticity -- thermal compliance is much less sensitive to intermediate
# density than strain energy is -- so without it the design stays grey and thresholding it at 0.5
# gives a blob. Measured on the first attempt here: M_nd = 0.128 against the bracket's 0.03-0.05.
# `penal` is a runtime PARAMETER so raising it does not trigger a recompile.
penal_p = jno.np.parameter((1,), name="penal")
penal_p.dtype(jnp.float64)
penal_p.initialize(lambda kk, sh, dtype=None: jnp.full(sh, PENAL))
penal_p.optimizer(optax.sgd(1.0))

# A solid seed under the sink, for the same reason the bracket keeps solid under its bolts: without
# it the optimiser empties the elements carrying the boundary condition and the objective then
# measures that artefact instead of the tree.
cen0 = pts0[cells].mean(axis=1)
solid = (
    (np.abs(cen0[:, 0] - 0.5 * L) < SINK + 1.5 * H_MESH)
    & (np.abs(cen0[:, 1] - 0.5 * L) < SINK + 1.5 * H_MESH)
    & (cen0[:, 2] < 1.5 * H_MESH)
)
pmask = jnp.asarray(solid)
print(f"non-design solid (sink seed): {int(solid.sum())} of {NT} tets", flush=True)

patch = d.patch_filter()
_beta_now = [1.0]


def physical(r):
    x = jnp.where(pmask, 1.0, patch(r))
    if BETA_H <= 0:
        return jnp.clip(x, RHO_MIN, 1.0)
    tb = jnp.tanh(_beta_now[0] * ETA_H)
    proj = (tb + jnp.tanh(_beta_now[0] * (x - ETA_H))) / (tb + jnp.tanh(_beta_now[0] * (1.0 - ETA_H)))
    return jnp.clip(proj, RHO_MIN, 1.0)


rho.constrain(physical)

# --- the conduction problem ---------------------------------------------------------------------
#   k(rho) grad T . grad phi  =  Q phi        in the cube
#   T = 0                                     on the sink patch
# The source is a VOLUME term (unbound phi), not a surface one: heat is generated throughout the
# material, which is what makes this a volume-to-point problem rather than a boundary one.
xi, yi, zi, _ = d.variable("interior", split=True)
sink = d.variable(
    "coldpatch",
    where=lambda x, y, z: (z < TOL) & (jnp.abs(x - 0.5 * L) < SINK) & (jnp.abs(y - 0.5 * L) < SINK),
    split=True,
)[:3]
k = lambda r: KMIN + r**penal_p * (K0 - KMIN)  # noqa: E731  -- SIMP for conductivity
# Bound symbols: the coordinates say which region the term integrates over, exactly as the shipped
# Poisson tutorial writes it. `grad` keeps the weak form reading like the equation above.
Ti, vi = T.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
grad = lambda w: jno.np.grad(w, [xi, yi, zi])  # noqa: E731

fem = jno.fem([k(rho) * inner(grad(Ti), grad(vi), 1) - QGEN * vi, T(*sink) - 0.0], quad_degree=2)

# The sink must actually be constrained, or the problem is singular and the "tree" is noise.
_n_sink = int(np.sum((pts0[:, 2] < TOL) & (np.abs(pts0[:, 0] - 0.5 * L) < SINK) & (np.abs(pts0[:, 1] - 0.5 * L) < SINK)))
assert _n_sink > 0, f"the cold patch caught no nodes; SINK={SINK} is smaller than the mesh"
print(f"sink nodes: {_n_sink}", flush=True)

# --- objective and constraints --------------------------------------------------------------------
# Thermal compliance, the conduction analogue of structural compliance: int k |grad T|^2 = int Q T,
# i.e. the volume-averaged temperature. Minimising it is minimising how hot the block runs.
compliance = (k(rho) * inner(grad(Ti), grad(Ti), 1)).integrate(fem, solver=pardiso_lu)
cellv = d.cell_volume()
volume = (rho.reshape(-1) * cellv).sum / (VOLFRAC * cellv.sum)
g_ang = ((2 * np.pi - d.cell_angles()) / (2 * np.pi - THETA_MIN)).reshape(-1).pnorm(PN, normalize=True)
g_asp = (d.cell_aspect() / ASPECT_MAX).pnorm(PN, normalize=True)
_v0 = np.abs(np.linalg.det(np.stack([pts0[cells][:, i + 1] - pts0[cells][:, 0] for i in range(3)], -1))) / 6.0
V0MAX = float(os.environ.get("V0MAX_FRAC", "1.30")) * float(np.percentile(_v0, 99))
V0MIN = float(os.environ.get("V0MIN_FRAC", "0.50")) * float(np.percentile(_v0, 1))
rho_e = rho.reshape(-1)
g_vmx = (cellv / ((2.0 - rho_e) * V0MAX)).pnorm(PN, normalize=True)
g_vmn = ((2 * V0MAX - cellv) / (2 * V0MAX - V0MIN)).pnorm(PN, normalize=True)
perim = rho.perimeter(zeta=0.1)

curv_r = rho.curvature(zeta=0.1)
curv = curv_r.pnorm(SMOOTH_P) if SMOOTH_P > 0 else curv_r.sum
terms, callbacks = [compliance], [jno.optimizers.simp_continuation(penal_p, rho, physical=patch, every=25, watch=0)]
if SMOOTH > 0:
    _coord0 = [
        jnp.asarray(pts0[np.asarray(sp["ids"], dtype=int), int(sp["axis"])]) for sp in (d._trainable_coords or [])
    ]
    _cut = (pts0[cells].mean(axis=1)[:, 2] > 0.5 * L).astype(float)
    _flat = np.asarray(curv_r.fn(jnp.asarray(_cut), *_coord0))
    assert _flat.ndim == 1 and _flat.size > NT, (
        f"curvature() returned {_flat.shape}; expected one value per ridge. vendor/jNO is behind -- "
        "`.pnorm()` on a scalar is the identity and the run would be a lie."
    )
    S_REF = float((_flat**SMOOTH_P).sum() ** (1.0 / SMOOTH_P)) if SMOOTH_P > 0 else float(_flat.sum())
    terms.append(((SMOOTH / S_REF) * curv).name("S"))
    print(f"smoothness: SMOOTH={SMOOTH}  S_ref={S_REF:.4g}  weight={SMOOTH / S_REF:.4e}", flush=True)
if PSTAR > 0:
    beta_p = jno.np.parameter((1,), name="beta")
    beta_p.dtype(jnp.float64)
    beta_p.initialize(lambda kk, sh, dtype=None: jnp.full(sh, BETA))
    beta_p.optimizer(optax.sgd(1.0))
    terms.append((beta_p[0] * perim.log_barrier(PSTAR)).name("R"))
    callbacks.append(jno.optimizers.geometric_decay(beta_p, GAMMA, start=BETA, minimum=BETA_MIN))
terms += [jno.le(volume, 1.0), jno.le(g_ang, 1.0), jno.le(g_asp, 1.0), jno.le(g_vmx, 1.0), jno.le(g_vmn, 1.0)]
crux = jno.core(terms, domain=jno.domain.from_array({"_": np.zeros((1, 1))}))

coord_specs = list(getattr(d, "_trainable_coords", []) or [])
coord_nodes = [sp["expr"] for sp in coord_specs]
hist = {kk: [] for kk in ("it", "C", "V", "P", "S", "mnd", "ang", "asp", "vmx", "vmn")}
frames_rho, frames_pts = [], []
ev = lambda n: float(np.asarray(crux.eval([n])).reshape(-1)[0])  # noqa: E731
DST = os.path.join(OUT, f"topo3d_{TAG}.npz")
_fixed = np.asarray((pts0[:, 2] < TOL) & (np.abs(pts0[:, 0] - 0.5 * L) < SINK) & (np.abs(pts0[:, 1] - 0.5 * L) < SINK))


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
        fixed=_fixed,
        **{kk: np.asarray(v) for kk, v in hist.items()},
    )
    os.replace(tmp, DST)


def record(it_now):
    global it
    it = it_now
    r = np.asarray(crux.eval([rho])).reshape(-1)
    p = pts0.copy()
    for sp, node in zip(coord_specs, coord_nodes):
        p[np.asarray(sp["ids"], dtype=int), int(sp["axis"])] = np.asarray(crux.eval([node])).reshape(-1)
    for kk, v in (
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
        hist[kk].append(v)
    frames_rho.append(r.copy())
    frames_pts.append(p.copy())
    print(
        f"  it {it:4d}  C={hist['C'][-1]:10.4f}  V={hist['V'][-1]:.4f}  P={hist['P'][-1]:8.2f}  "
        f"S={hist['S'][-1]:8.2f}  M_nd={hist['mnd'][-1]:.4f}  beta_H={_beta_now[0]:.1f}  "
        f"g_ang={hist['ang'][-1]:.3f} asp={hist['asp'][-1]:.3f} vmx={hist['vmx'][-1]:.3f}  ({time.time() - t0:.0f}s)",
        flush=True,
    )
    checkpoint()


it, t0 = 0, time.time()
record(0)
while time.time() - t0 < BUDGET and it < ITERS:
    if BETA_H > 0:
        _beta_now[0] = min(BETA_H, float(BETA_H) ** min(1.0, it / max(BETA_AT * ITERS, 1.0)))
    crux.solve(CHUNK, callbacks=callbacks)
    record(it + CHUNK)

print(f"saved topo3d_{TAG}.npz  ({len(frames_rho)} frames, {it} iterations, {(time.time() - t0) / 60:.1f} min)", flush=True)
_ru, _rc = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"cpu {_ru.ru_utime + _ru.ru_stime + _rc.ru_utime + _rc.ru_stime:.0f} s over {time.time() - t0:.0f} s wall", flush=True)
