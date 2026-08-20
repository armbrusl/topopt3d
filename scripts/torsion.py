"""Torsion: the case whose optimum CANNOT be an extrusion of a 2-D one.

The pseudo-2D slab validated the machinery by reducing to a known 2-D answer -- which is exactly why
it cannot demonstrate a 3-D method. A reviewer can always say "sweep the 2-D result". Torsion closes
that: a torque about the long axis produces diagonal / helical members that no 2-D reduction
generates, and the answer is known independently -- torsional rigidity is maximised by a CLOSED
THIN-WALLED section (Bredt), so material should migrate to a tube far from the axis.

Setup: a 3:1:1 bar clamped at one end, a force COUPLE at the other. The couple is two opposite bands
carrying constant tractions rather than a tangential field, because a constant vector is what the
traction term takes; the pair still integrates to a pure torque, which the assertions below check on
the ASSEMBLED load rather than on the intent (see the LOAD_AXIS bug in the README).

    PSTAR=0 python scripts/torsion.py     # probe: no barrier, measure the natural perimeter AREA
    PSTAR=<measured> python scripts/torsion.py

Method: Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403 -- eq. (17)-(19) patch
filter (3-D: edge fans), eq. (21)/(24) dihedral bounds, eq. (26)/(28) density-coupled element-volume
bounds, eq. (38)-(41) perimeter barrier (in 3-D an AREA). Heaviside projection after Wang, Lazarov &
Sigmund, *Struct. Multidisc. Optim.* **43** (2011) 767-784.
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
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
os.makedirs(OUT, exist_ok=True)

L, H, W = 6.0, 2.0, 2.0
H_MESH = float(os.environ.get("HMESH", "0.20"))
E0, NU, PENAL = 1.0, 0.3, 3.0
EMIN = float(os.environ.get("EMIN", "1e-4"))
LAM, MU = E0 * NU / ((1 + NU) * (1 - 2 * NU)), E0 / (2 * (1 + NU))
VOLFRAC = float(os.environ.get("VOLFRAC", "0.25"))
THETA_MIN, ASPECT_MAX, PN, TOL = np.deg2rad(5.0), 8.0, 50.0, 1e-9
MOVE_BOUND = float(os.environ.get("MOVE_FRAC", "2.0")) * H_MESH
BAND = float(os.environ.get("BAND", "0.4"))      # thickness of each couple band, and of the end caps
PSTAR = float(os.environ.get("PSTAR", "0"))
BETA, GAMMA = 1e-3, 0.997
BETA_MIN = float(os.environ.get("BETA_MIN", "5e-4"))
BETA_H = float(os.environ.get("BETA_H", "16"))
ETA_H, RHO_MIN = 0.5, 1e-3
ITERS = int(os.environ.get("ITERS", "250"))
CHUNK = int(os.environ.get("CHUNK", "10"))
BUDGET = float(os.environ.get("BUDGET", "5400"))
TAG = os.environ.get("TAG", "torsion")

d = jno.Shape.box(0, 0, 0, L, H, W, size=H_MESH).domain()
cells = np.asarray(d._cells_p1())
pts0 = np.asarray(d.mesh.points)[:, :3]
NT = cells.shape[0]
print("=" * 88, flush=True)
print(f"  TORSION  box {L}x{H}x{W}  h={H_MESH}  tets={NT}  dofs={3 * pts0.shape[0]}  "
      f"volfrac={VOLFRAC}  PSTAR={PSTAR}  BETA_H={BETA_H}  TAG={TAG}", flush=True)
print("=" * 88, flush=True)

# --- design variables 2-4: interior nodal coordinates -------------------------------------------
xm, ym, zm, _ = d.variable("mv", where=lambda x, y, z: (
    (x > TOL) & (x < L - TOL) & (y > TOL) & (y < H - TOL) & (z > TOL) & (z < W - TOL)), split=True)
xm.trainable(name="mesh_x"), ym.trainable(name="mesh_y"), zm.trainable(name="mesh_z")
ids = np.asarray(d._trainable_coords[0]["ids"], dtype=int)
for spec in d._trainable_coords:
    c0 = pts0[ids, int(spec["axis"])]
    spec["expr"].optimizer(jno.optimizers.mma(
        move=0.02, lower=c0 - MOVE_BOUND, upper=c0 + MOVE_BOUND, move_gamma=0.9985, move_min=0.001))

# --- design variable 1: one density per tetrahedron ---------------------------------------------
u, phi = d.fem_symbols(value_shape=(3,))
_r, s = d.fem_symbols(space="P0", names=("r", "s"))
rho = jno.np.parameter(s, name="rho")
rho.dtype(jnp.float64)
rho.initialize(jax.nn.initializers.constant(VOLFRAC))
rho.optimizer(jno.optimizers.mma(move=0.15, lower=1e-3, upper=1.0, move_gamma=0.9985, move_min=0.0075))

# NON-DESIGN regions, frozen SOLID. Both ends: the clamped face has to be able to react the torque,
# and the two loaded bands would otherwise be emptied, leaving the compliance to measure that
# artefact rather than the structure. A point or band load on a designable cell is the classic way
# to get a meaningless number that still looks like a converged design.
cen0 = pts0[cells].mean(axis=1)
solid = ((cen0[:, 0] < BAND)                                        # the clamped end cap
         | ((cen0[:, 0] > L - BAND) & (cen0[:, 2] > W - BAND))      # +z band of the couple
         | ((cen0[:, 0] > L - BAND) & (cen0[:, 2] < BAND)))         # -z band of the couple
pmask = jnp.asarray(solid)
print(f"non-design solid: {int(solid.sum())} of {NT} tets", flush=True)

patch = d.patch_filter()  # eq. (17)-(19); on tets this walks EDGE FANS
_beta_now = [1.0]


def physical(r):
    x = jnp.where(pmask, 1.0, patch(r))
    if BETA_H <= 0:
        return jnp.clip(x, RHO_MIN, 1.0)
    tb = jnp.tanh(_beta_now[0] * ETA_H)
    proj = (tb + jnp.tanh(_beta_now[0] * (x - ETA_H))) / (tb + jnp.tanh(_beta_now[0] * (1.0 - ETA_H)))
    return jnp.clip(proj, RHO_MIN, 1.0)


rho.constrain(physical)

# --- the couple ----------------------------------------------------------------------------------
xi, yi, zi, _ = d.variable("interior", split=True)
root = d.variable("root", where=lambda x, y, z: x < TOL, split=True)[:3]
# `+ TOL` on every band edge: a node landing exactly on the bound is otherwise dropped and the band
# is one facet short -- half the force, a quarter of the compliance, silently. That is not
# hypothetical; it cost a full investigation on the 2-D cantilever.
# NOT "top"/"bottom"/"left": those are jNO's own face tags, and a reserved name silently OVERRIDES
# the `where=` predicate. Measured: `d.variable("top", where=(x > L-TOL) & (z > W-BAND-TOL))` tagged
# the entire z = W face over the whole 6-unit length -- 392 nodes spanning x in [0.15, 6.00] instead
# of a 35-node end strip -- and the run would simply have solved a different problem.
pz = d.variable("couple_pz", where=lambda x, y, z: (x > L - TOL) & (z > W - BAND - TOL), split=True)[:3]
mz = d.variable("couple_mz", where=lambda x, y, z: (x > L - TOL) & (z < BAND + TOL), split=True)[:3]
eps = lambda w: symgrad(w, [xi, yi, zi])  # noqa: E731
a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731
E = lambda r: EMIN + r**PENAL * (E0 - EMIN)  # noqa: E731


def _band_area(keep):
    """Area of the boundary facets whose three nodes all satisfy ``keep``.

    The two strips do NOT mesh to equal areas -- measured ~2 % apart at h = 0.2 -- and the leftover
    y-resultant times the 6-unit arm is a spurious moment about z that reached 15 % of the torque.
    Scaling each traction by its own band area makes the couple exactly balanced.

    Computed geometrically rather than by assembling once and rescaling, because **a second
    `jno.fem(...)` on the same domain returns the FIRST operator**: measured, tractions of 1.0 and
    10.0 both assemble |f| = 0.9692, so a calibrate-then-rebuild round trip is silently a no-op.
    """
    lf = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    seen: dict = {}
    for t in range(NT):
        c = cells[t]
        for f in lf:
            k = tuple(sorted(int(c[i]) for i in f))
            seen.pop(k, None) if k in seen else seen.setdefault(k, t)
    tri = np.array(list(seen.keys()), dtype=int)          # boundary facets: used by one cell only
    tri = tri[keep(pts0[tri]).all(axis=1)]
    v = pts0[tri]
    return float(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2.0)


_A_pz = _band_area(lambda P: (P[..., 0] > L - TOL) & (P[..., 2] > W - BAND - TOL))
_A_mz = _band_area(lambda P: (P[..., 0] > L - TOL) & (P[..., 2] < BAND + TOL))
assert min(_A_pz, _A_mz) > 1e-12, f"a couple band has no boundary facets (+z {_A_pz:.2e}, -z {_A_mz:.2e})"
print(f"couple band areas: +z {_A_pz:.4f}   -z {_A_mz:.4f}   (nominal {H * BAND:.4f})", flush=True)

fem = jno.fem([
    E(rho) * a(eps(u), eps(phi)),
    u(*root) - (0.0, 0.0, 0.0),
    -1.0 * inner(jnp.array([0.0, +1.0 / _A_pz, 0.0]), phi.bind(**dict(zip("xyz", pz))), 1),
    -1.0 * inner(jnp.array([0.0, -1.0 / _A_mz, 0.0]), phi.bind(**dict(zip("xyz", mz))), 1),
], quad_degree=2)

_A_op, _b = fem.operator.evaluate({"rho": jnp.full(NT, VOLFRAC)})
_f = np.asarray(jnp.asarray(_b)).reshape(-1, 3)
_res = _f.sum(axis=0)
_moment = np.cross(pts0 - np.array([0.0, H / 2, W / 2]), _f).sum(axis=0)
print(f"load |f| = {np.abs(_f).sum():.4f}   resultant = ({_res[0]:+.2e}, {_res[1]:+.2e}, {_res[2]:+.2e})", flush=True)
print(f"moment about (x, y, z) = ({_moment[0]:+.4f}, {_moment[1]:+.4f}, {_moment[2]:+.4f})", flush=True)
assert np.abs(_res).max() < 1e-8 * np.abs(_f).sum(), f"a couple must have zero resultant, got {_res}"
assert abs(_moment[0]) > 0.99 * np.abs(_moment).sum(), f"the moment must be about x, got {_moment}"

# --- objective and constraints --------------------------------------------------------------------
compliance = (E(rho) * a(eps(u), eps(u))).integrate(fem, solver=pardiso_lu)
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

terms = [compliance]
callbacks = []
if PSTAR > 0:
    beta_p = jno.np.parameter((1,), name="beta")
    beta_p.dtype(jnp.float64)
    beta_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, BETA))
    beta_p.optimizer(optax.sgd(1.0))
    terms.append((beta_p[0] * perim.log_barrier(PSTAR)).name("R"))
    callbacks.append(jno.optimizers.geometric_decay(beta_p, GAMMA, start=BETA, minimum=BETA_MIN))
terms += [jno.le(volume, 1.0), jno.le(g_ang, 1.0), jno.le(g_asp, 1.0), jno.le(g_vmx, 1.0), jno.le(g_vmn, 1.0)]
crux = jno.core(terms, domain=jno.domain.from_array({"_": np.zeros((1, 1))}))

coord_specs = list(getattr(d, "_trainable_coords", []) or [])
coord_nodes = [sp["expr"] for sp in coord_specs]
hist = {k: [] for k in ("it", "C", "V", "P", "mnd", "ang", "asp", "vmx", "vmn")}
frames_rho, frames_pts = [], []
ev = lambda n: float(np.asarray(crux.eval([n])).reshape(-1)[0])  # noqa: E731
DST = os.path.join(OUT, f"topo3d_{TAG}.npz")


def checkpoint():
    tmp = DST + ".tmp.npz"
    np.savez(tmp, cells=cells, pts0=pts0, pts_f=frames_pts[-1], rho_f=frames_rho[-1],
             frames_rho=np.array(frames_rho), frames_pts=np.array(frames_pts),
             solid=solid, **{k: np.asarray(v) for k, v in hist.items()})
    os.replace(tmp, DST)


it, t0 = 0, time.time()
while time.time() - t0 < BUDGET and it < ITERS:
    crux.solve(CHUNK, callbacks=callbacks)
    it += CHUNK
    if BETA_H > 0:
        _beta_now[0] = min(BETA_H, _beta_now[0] * 1.10)
    r = np.asarray(crux.eval([rho])).reshape(-1)
    p = pts0.copy()
    for sp, node in zip(coord_specs, coord_nodes):
        p[np.asarray(sp["ids"], dtype=int), int(sp["axis"])] = np.asarray(crux.eval([node])).reshape(-1)
    for k, v in (("it", it), ("C", ev(compliance)), ("V", ev(volume)), ("P", ev(perim)),
                 ("mnd", float(4.0 * np.mean(r * (1.0 - r)))), ("ang", ev(g_ang)), ("asp", ev(g_asp)),
                 ("vmx", ev(g_vmx)), ("vmn", ev(g_vmn))):
        hist[k].append(v)
    frames_rho.append(r.copy())
    frames_pts.append(p.copy())
    print(f"  it {it:4d}  C={hist['C'][-1]:10.3f}  V={hist['V'][-1]:.4f}  P={hist['P'][-1]:9.2f}  "
          f"M_nd={hist['mnd'][-1]:.4f}  beta_H={_beta_now[0]:.1f}  g_ang={hist['ang'][-1]:.3f} "
          f"asp={hist['asp'][-1]:.3f} vmx={hist['vmx'][-1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    checkpoint()

print(f"saved topo3d_{TAG}.npz  ({len(frames_rho)} frames, {it} iterations, {(time.time() - t0) / 60:.1f} min)",
      flush=True)
_ru, _rc = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
_cpu = _ru.ru_utime + _ru.ru_stime + _rc.ru_utime + _rc.ru_stime
print(f"cpu {_cpu:.0f} s over {time.time() - t0:.0f} s wall -> {_cpu / (time.time() - t0):.1f} cores "
      f"(machine has {os.cpu_count()})", flush=True)
