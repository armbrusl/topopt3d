"""The paper's own 2-D cantilever, reproduced -- the validation this repo did not have.

Everything else here is a 3-D EXTENSION of Jung, Yun & Kim, *Computers & Structures* **331**
(2026) 108403. The obvious question about an extension is whether the implementation reproduces
the method it extends, and until this script that question had no answer: `runs/p2d*` is a
pseudo-2-D SLAB run of the 3-D code (a box 2 elements thick) whose element-volume constraint sits
violated at vmx 1.2-1.4 throughout. That is a smoke test, not a reproduction.

This is a genuine 2-D plane-stress run on triangles, with the paper's own constants, against the
numbers printed in their Fig. 5 and Fig. 7:

                                  theirs      what to compare
    elements                       4,108      the mesh at h = 1 on 60 x 30
    dofs (E-FEM)                  12,870      2,145 nodes x 2 components x 3 slots
    compliance, E-FEM             80.809      SPACE=cover
    compliance, conventional      75.965      SPACE=Lagrange
    volume fraction                0.400      both
    iterations                       143      both, to their convergence rule

**The dof count is the sharpest of these and the cheapest to check.** First-order interpolation
cover carries 1 + dim slots per node per component: 3 in 2-D, 4 in 3-D (which is what the 3-D runs
here use). 2,145 x 2 x 3 = 12,870 exactly. A mismatch means the enrichment is not the one the
paper describes, and no compliance agreement would be meaningful on top of it.

Their setup, from Section 3: target edge length 1 independent of domain size; V0max = 0.75 and
V0min = 0.05; theta_min = 20 deg; movement variables bounded +/-2 except at boundary conditions
and on the outer boundary; p-norm p = 50; penalisation starts at 3 and rises by one whenever the
grey indicator M_nd is still above 1e-4 at convergence; convergence when the relative change in
the objective drops below 1e-4.

Note they use PENALISATION CONTINUATION to reach a binary design, not Heaviside projection. The
3-D runs here use projection, which is a deliberate difference and not one to import into a
validation -- so this script follows their rule, via `jno.optimizers.simp_continuation`.

    SPACE=cover     python scripts/cantilever2d.py     # their Fig. 5c / 7a -> 80.809
    SPACE=Lagrange  python scripts/cantilever2d.py     # their Fig. 7b      -> 75.965

Structure follows jNO's own deformable-mesh cantilever tutorial
(docs/tutorial_examples/08_fem_and_varpinns/topology_optimisation_cantilever.py), which already
carries the conventional formulation and the paper's constants; what this adds is the E-FEM arm,
the paper's convergence rule, and the comparison against their printed numbers.
"""

import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)  # p=50 aggregation needs the exponent range

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

SPACE = os.environ.get("SPACE", "cover")  # "cover" = E-FEM (their proposal) | "Lagrange"
TAG = os.environ.get("TAG", f"c2d_{SPACE}")
L, H = 60.0, 30.0  # their Fig. 5a: design domain L x 0.5L, L = 60
HMESH = float(os.environ.get("HMESH", "1.0"))  # "target edge length set to unity"
E0, EMIN, NU = 1.0, 1e-9, 0.3
LAM, MU = E0 * NU / (1 - NU**2), E0 / (2 * (1 + NU))  # PLANE STRESS -- their ABAQUS CPS4
VOLFRAC = float(os.environ.get("VOLFRAC", "0.40"))
PENAL = 3.0  # "the initial penalization factor is set to 3"
THETA_MIN, PNORM = np.deg2rad(20.0), 50.0  # "theta_min = 20 deg", "p = 50"
V0MAX, V0MIN = 0.75 * HMESH**2, 0.05 * HMESH**2  # "V0,max = 0.75 and V0,min = 0.05"
MOVE_BOUND = 2.0 * HMESH  # "movement design variables are bounded between -2 and 2"
SPAN, TOL = 2.0 * HMESH, 1e-6 * L
ITERS = int(os.environ.get("ITERS", "400"))  # a cap; their rule below is what stops it
CTOL = float(os.environ.get("CTOL", "1e-4"))  # "relative change in the objective ... 1e-4"
MND_TOL = float(os.environ.get("MND_TOL", "1e-4"))  # "if M_nd remains larger than 1e-4"
# NOT optional, and the paper does not print its value. Their Section 3 says complexity control
# via the perimeter is "included"; Fig. 5/7 give no P*. It cannot simply be left off: without a
# length scale the continuum problem is not well posed, the design fragments at any resolution,
# and here that is not a soft degradation -- the SIMP operator goes singular and the adjoint solve
# fails outright (SuperLU "failed to factorize matrix" on the first gradient). 650 is the value in
# jNO's own deformable-mesh cantilever tutorial for this exact domain, and it is an ASSUMPTION
# this reproduction makes on the paper's behalf, not a number taken from it.
PSTAR = float(os.environ.get("PSTAR", "650"))
BETA, GAMMA, BETA_MIN = 2e-4, 0.997, 1e-4

d = jno.Shape.rect(0, 0, L, H, size=HMESH).domain()
pts0 = np.asarray(d.mesh.points)[:, :2]
cells = np.asarray(d._cells_p1())
NT, NN = cells.shape[0], pts0.shape[0]

# --- design variables 2 and 3: the interior nodal coordinates ---------------------------------
# "except for nodes restricted by the boundary conditions and nodes located on the outer boundary
# of the mesh" -- so only strictly interior nodes move. In 3-D this repo also slides boundary
# nodes tangentially; that is an extension, and a validation must not use it.
RELOCATE = int(os.environ.get("RELOCATE", "1"))  # 0 pins the mesh: the fixed-mesh SIMP arm
interior = lambda x, y: (x > TOL) & (x < L - TOL) & (y > TOL) & (y < H - TOL)  # noqa: E731
xm, ym, _ = d.variable("mv", where=interior, split=True)
if RELOCATE:
    xm.trainable(name="mesh_x"), ym.trainable(name="mesh_y")
for spec in getattr(d, "_trainable_coords", []) or []:   # empty when RELOCATE=0
    _ids = np.asarray(spec["ids"], dtype=int)  # per-axis ids: never reuse axis 0's for the others
    c0 = pts0[_ids, int(spec["axis"])]
    spec["expr"].optimizer(
        jno.optimizers.mma(
            move=0.02, lower=c0 - MOVE_BOUND, upper=c0 + MOVE_BOUND,
            move_gamma=0.9985, move_min=0.001,
        )
    )

# --- design variable 1: one density per element -----------------------------------------------
u, phi = d.fem_symbols(value_shape=(2,), **({"space": "cover"} if SPACE == "cover" else {}))
_r, s = d.fem_symbols(space="P0", names=("r", "s"))
xi, yi, _ = d.variable("interior", split=True)
xl, yl, _ = d.variable("left", split=True)  # clamped root, their Fig. 5a hatching
xt, yt, _ = d.variable("tip", where=lambda x, y: (x > L - TOL) & (y < SPAN + TOL), split=True)

rho = jno.np.parameter(s, name="rho")
rho.dtype(jnp.float64)
rho.initialize(jax.nn.initializers.constant(VOLFRAC))
rho.optimizer(
    jno.optimizers.mma(move=0.15, lower=1e-3, upper=1.0, move_gamma=0.9985, move_min=0.0075)
)

penal_p = jno.np.parameter((1,), name="penal")
penal_p.dtype(jnp.float64)
penal_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, PENAL))
penal_p.optimizer(optax.sgd(1.0))

cen0 = pts0[cells].mean(axis=1)
pmask = jnp.asarray((cen0[:, 0] > L - SPAN) & (cen0[:, 1] < SPAN))  # solid under the load
patch = d.patch_filter()  # eq. (17)-(19)
rho.constrain(lambda r: jnp.where(pmask, 1.0, patch(r)))

eps = lambda w: symgrad(w, [xi, yi])  # noqa: E731
a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731
E = lambda r: EMIN + r**penal_p * (E0 - EMIN)  # noqa: E731

fem = jno.fem(
    [
        E(rho) * a(eps(u), eps(phi)),
        u(xl, yl) - (0.0, 0.0),
        -1.0 * inner(jnp.array([0.0, -1.0 / SPAN]), phi.bind(x=xt, y=yt), n_contract=1),
    ],
    quad_degree=2,
)

cellv, angles = d.cell_volume(), d.cell_angles()
rho_e = rho.reshape(-1)
# Through MKL PARDISO, not jNO's default sparse LU. As the design sharpens the void carries
# EMIN = 1e-9, so the operator's condition number approaches 1e9 and SuperLU stops being able to
# factorise it: measured here, the run descends cleanly to iteration 200 (C = 78.744) and then
# dies with "failed to factorize matrix" inside the gradient. `bracket.py` routes every 3-D solve
# the same way for the same reason.
compliance = (E(rho) * a(eps(u), eps(u))).integrate(fem, solver=pardiso_lu).name("C")
volume = ((rho_e * cellv).sum / (VOLFRAC * cellv.sum)).name("V")
g_ang = (((2 * jnp.pi - angles) / (2 * jnp.pi - THETA_MIN)).reshape(-1)).pnorm(PNORM, normalize=True)
g_vmx = (cellv / ((2.0 - rho_e) * V0MAX)).pnorm(PNORM, normalize=True)  # eq. (26)
g_vmn = ((2 * V0MAX - cellv) / (2 * V0MAX - V0MIN)).pnorm(PNORM, normalize=True)  # eq. (28)
perim = rho.perimeter(zeta=0.1)

terms = [compliance]
callbacks = [jno.optimizers.simp_continuation(penal_p, rho, physical=patch, every=25, watch=0)]
if PSTAR > 0:
    beta_p = jno.np.parameter((1,), name="beta")
    beta_p.dtype(jnp.float64)
    beta_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, BETA))
    beta_p.optimizer(optax.sgd(1.0))
    terms.append((beta_p[0] * perim.log_barrier(PSTAR)).name("R"))
    callbacks.append(jno.optimizers.geometric_decay(beta_p, GAMMA, start=BETA, minimum=BETA_MIN))
# On a pinned mesh the element-quality rows are constants (and g_vmx is density-coupled but
# capped by whatever the mesher chose) -- the same trap documented at length in bracket.py.
terms.append(jno.le(volume, 1.0))
if RELOCATE:
    terms += [jno.le(g_ang, 1.0), jno.le(g_vmx, 1.0), jno.le(g_vmn, 1.0)]

crux = jno.core(terms, domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
ev = lambda n: float(np.asarray(crux.eval([n])).reshape(-1)[0])  # noqa: E731

print("=" * 78, flush=True)
print(f"  CANTILEVER 2-D  {L:.0f}x{H:.0f}  h={HMESH}  tris={NT}  nodes={NN}  dofs={fem.dofs}", flush=True)
print(f"          SPACE={SPACE}  volfrac={VOLFRAC}  V0=[{V0MIN},{V0MAX}]  move=+/-{MOVE_BOUND}", flush=True)
print(f"          paper: 4108 tris, 12870 dofs (E-FEM), C=80.809 (E-FEM) / 75.965 (conventional)", flush=True)
print("=" * 78, flush=True)

# Their stopping rule, not a fixed count: converge the objective, then raise the penalisation if
# the design is still grey, and stop only when BOTH hold. `simp_continuation` owns the raise; this
# loop owns the stop, so the iteration count is comparable to their 143.
# Solve in CHUNKS, not one iteration at a time. MMA carries moving asymptotes between steps and
# a fresh `solve(1)` does not continue them: driven per-iteration this problem oscillates instead
# of descending (measured: C 645 -> 360 -> 242 -> 380 over four steps, then the operator goes
# singular and the adjoint fails to factorise around step 5). `bracket.py` runs CHUNK=50 for the
# same reason. It also matches the paper's own wording better -- their convergence test is over
# "three consecutive iteration intervals", not three iterations.
CHUNK = int(os.environ.get("CHUNK", "25"))
t0, prev, it, hist = time.time(), None, 0, []
while it < ITERS:
    crux.solve(CHUNK, callbacks=callbacks)
    it += CHUNK
    C = ev(compliance)
    rf = np.asarray(crux.eval([rho])).reshape(-1)
    mnd = float(np.mean(4 * rf * (1 - rf)))
    hist.append((it, C, mnd, ev(volume)))
    conv = prev is not None and abs(C - prev) <= CTOL * max(abs(prev), 1e-30)
    prev = C
    if True:
        print(f"  it {it:4d}  C={C:10.4f}  V={ev(volume):.4f}  M_nd={mnd:.5f}  "
              f"penal={ev(penal_p):.2f}  g_ang={ev(g_ang):.3f} vmx={ev(g_vmx):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    if conv and mnd <= MND_TOL:
        break

C = ev(compliance)
rho_f = np.asarray(crux.eval([rho])).reshape(-1)
pts_f = pts0.copy()
for spec in getattr(d, "_trainable_coords", []) or []:
    pts_f[np.asarray(spec["ids"], dtype=int), int(spec["axis"])] = np.asarray(
        crux.eval([spec["expr"]])
    ).reshape(-1)
area = np.abs(np.linalg.det(np.stack(
    [pts_f[cells][:, 1] - pts_f[cells][:, 0], pts_f[cells][:, 2] - pts_f[cells][:, 0]], -1))) / 2

TARGET = {"cover": 80.809, "Lagrange": 75.965}[SPACE]
# Report a WINDOW, never the last iterate. The conventional arm does not settle on this problem --
# measured over the last 200 iterations it swings 23.9 % (77.36 to 98.06) -- so its final value is
# wherever the run happened to stop, and quoting that as "converged" turns a coin flip into an
# agreement. (Stopping at iteration 400 flatters it to +1.8 %; the window says +14.0 %.) The E-FEM
# arm on identical machinery swings 3.2 %, so this is a property of the formulation, not of the
# harness -- which is the paper's own claim, showing up as instability rather than as a bad number.
_w = np.array([h[1] for h in hist[-8:]])
_swing = 100 * (_w.max() - _w.min()) / _w.mean()
print(f"\n  C over the last {len(_w)} chunks: mean {_w.mean():.4f}  "
      f"range [{_w.min():.4f}, {_w.max():.4f}]  swing {_swing:.1f} %", flush=True)
print(f"  their Fig. 7 gives {TARGET}  ->  window mean {100 * (_w.mean() / TARGET - 1):+.2f} %"
      f"   (last iterate alone would say {100 * (C / TARGET - 1):+.2f} %)", flush=True)
if _swing > 5.0:
    print(f"  NOT CONVERGED: a {_swing:.1f} % swing means no single value here is the answer; "
          f"the window mean is a summary of an oscillation, not a converged compliance", flush=True)
print(f"  dofs {fem.dofs} (their E-FEM count is 12,870)   elements {NT} (theirs 4,108)   "
      f"iterations {it} (theirs 143)", flush=True)
print(f"  volume fraction {np.sum(rho_f * area) / np.sum(area):.4f} (theirs 0.400)   "
      f"M_nd {float(np.mean(4 * rho_f * (1 - rho_f))):.5f}   "
      f"inverted elements {int((area <= 0).sum())}", flush=True)

np.savez_compressed(
    os.path.join(OUT, f"topo2d_{TAG}.npz"),
    cells=cells, pts0=pts0, pts_f=pts_f, rho_f=rho_f,
    hist=np.array(hist), C=C, dofs=fem.dofs, space=SPACE, target=TARGET,
)
print(f"  saved topo2d_{TAG}.npz", flush=True)
