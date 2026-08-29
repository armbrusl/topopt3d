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
validation -- so this script follows their rule.

It cannot follow it the way jNO usually would. `jno.optimizers.simp_continuation` drives `penal`
as a runtime parameter, and README blocker 1 records why that path is closed: the exponent's own
derivative, d/dp(rho^p) = rho^p ln(rho), blows up where the filtered density reaches zero, so the
gradient NaNs on the first solve. `penal` is not a design variable -- a callback owns it and the
derivative is computed only to be thrown away -- but jNO has no way to say so, and inventing one
would be a public API addition.

So the continuation is STAGED instead, which is what their Fig. 4 flowchart actually draws: hold
the exponent at a literal value, converge the compliance, and if M_nd is still above 1e-4 raise
the exponent by one and rebuild. A rebuild per stage costs a recompile and there are only a few
stages. Warm-starting across it needs the RAW density, not the physical one `crux.eval` returns --
re-filtering an already-filtered field every stage would smooth the design away -- and that is
read from `crux.models`, which holds parameters before their `constrain` is applied.

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

import equinox as eqx  # noqa: E402
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
PENAL0 = float(os.environ.get("PENAL0", "3"))  # "the initial penalization factor is set to 3"
PENAL_MAX = float(os.environ.get("PENAL_MAX", "8"))  # a cap; their rule below is what stops it
THETA_MIN, PNORM = np.deg2rad(20.0), 50.0  # "theta_min = 20 deg", "p = 50"
V0MAX, V0MIN = 0.75 * HMESH**2, 0.05 * HMESH**2  # "V0,max = 0.75 and V0,min = 0.05"
MOVE_BOUND = 2.0 * HMESH  # "movement design variables are bounded between -2 and 2"
SPAN, TOL = 2.0 * HMESH, 1e-6 * L
STAGE_ITERS = int(os.environ.get("STAGE_ITERS", "200"))  # cap per penalisation stage
CHUNK = int(os.environ.get("CHUNK", "25"))
CTOL = float(os.environ.get("CTOL", "1e-4"))  # "relative change in the objective ... 1e-4"
MND_TOL = float(os.environ.get("MND_TOL", "1e-4"))  # "if M_nd remains larger than 1e-4"
RELOCATE = int(os.environ.get("RELOCATE", "1"))  # 0 pins the mesh: the fixed-mesh SIMP arm
# NOT optional, and the paper does not print its value. Their Section 3 says complexity control
# via the perimeter is "included"; Fig. 5/7 give no P*. It cannot simply be left off: without a
# length scale the continuum problem is not well posed, the design fragments at any resolution,
# and here that is not a soft degradation -- the SIMP operator goes singular and the adjoint solve
# fails outright. 650 is the value in jNO's own deformable-mesh cantilever tutorial for this exact
# domain, and it is an ASSUMPTION this reproduction makes on the paper's behalf, not their number.
PSTAR = float(os.environ.get("PSTAR", "650"))
BETA, GAMMA, BETA_MIN = 2e-4, 0.997, 1e-4

_probe = jno.Shape.rect(0, 0, L, H, size=HMESH).domain()
PTS0 = np.asarray(_probe.mesh.points)[:, :2]
CELLS = np.asarray(_probe._cells_p1())
NT, NN = CELLS.shape[0], PTS0.shape[0]
del _probe


def raw_density(crux, n):
    """The density parameter BEFORE its `constrain` -- what a warm start has to carry.

    `crux.eval([rho])` applies the patch filter, so feeding that back as the next stage's initial
    value filters an already-filtered field and the design smooths away over a few stages.
    `crux.models` holds parameters unconstrained; the density is the leaf with one entry per cell.
    """
    hits = [
        np.asarray(lf).reshape(-1)
        for v in crux.models.values()
        for lf in jax.tree_util.tree_leaves(v)
        if eqx.is_inexact_array(lf) and lf.size == n
    ]
    if len(hits) != 1:  # never guess which leaf is the density
        raise RuntimeError(f"expected exactly one parameter leaf of size {n}, found {len(hits)}")
    return hits[0]


def build(penal, warm_rho=None, warm_coords=None):
    """One penalisation stage. `penal` is a LITERAL, so its derivative is never formed."""
    d = jno.Shape.rect(0, 0, L, H, size=HMESH).domain()
    interior = lambda x, y: (x > TOL) & (x < L - TOL) & (y > TOL) & (y < H - TOL)  # noqa: E731
    xm, ym, _ = d.variable("mv", where=interior, split=True)
    if RELOCATE:
        xm.trainable(name="mesh_x"), ym.trainable(name="mesh_y")
    for spec in getattr(d, "_trainable_coords", []) or []:
        ids = np.asarray(spec["ids"], dtype=int)  # per-axis: never reuse axis 0's for the others
        ax = int(spec["axis"])
        c0 = PTS0[ids, ax]  # bounds stay relative to the ORIGINAL mesh, as the paper specifies
        if warm_coords is not None:
            v = jnp.asarray(warm_coords[ax])
            spec["expr"].initialize(lambda k, sh, dtype=None, _v=v: _v.reshape(sh))
        spec["expr"].optimizer(
            jno.optimizers.mma(
                move=0.02, lower=c0 - MOVE_BOUND, upper=c0 + MOVE_BOUND,
                move_gamma=0.9985, move_min=0.001,
            )
        )

    u, phi = d.fem_symbols(value_shape=(2,), **({"space": "cover"} if SPACE == "cover" else {}))
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)  # clamped root, their Fig. 5a hatching
    xt, yt, _ = d.variable("tip", where=lambda x, y: (x > L - TOL) & (y < SPAN + TOL), split=True)

    rho = jno.np.parameter(s, name="rho")
    rho.dtype(jnp.float64)
    if warm_rho is None:
        rho.initialize(jax.nn.initializers.constant(VOLFRAC))
    else:
        _w = jnp.asarray(warm_rho)
        rho.initialize(lambda k, sh, dtype=None, _v=_w: _v.reshape(sh))
    rho.optimizer(
        jno.optimizers.mma(move=0.15, lower=1e-3, upper=1.0, move_gamma=0.9985, move_min=0.0075)
    )

    cen0 = PTS0[CELLS].mean(axis=1)
    pmask = jnp.asarray((cen0[:, 0] > L - SPAN) & (cen0[:, 1] < SPAN))  # solid under the load
    patch = d.patch_filter()  # eq. (17)-(19)
    rho.constrain(lambda r: jnp.where(pmask, 1.0, patch(r)))

    eps = lambda w: symgrad(w, [xi, yi])  # noqa: E731
    a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731
    E = lambda r: EMIN + r**penal * (E0 - EMIN)  # noqa: E731  -- LITERAL exponent

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
    # EMIN = 1e-9, so the operator's condition number approaches 1e9 and SuperLU stops being able
    # to factorise it: measured, the run descends cleanly to iteration 200 (C = 78.744) and then
    # dies with "failed to factorize matrix" inside the gradient. `bracket.py` does the same.
    compliance = (E(rho) * a(eps(u), eps(u))).integrate(fem, solver=pardiso_lu).name("C")
    volume = ((rho_e * cellv).sum / (VOLFRAC * cellv.sum)).name("V")
    g_ang = (((2 * jnp.pi - angles) / (2 * jnp.pi - THETA_MIN)).reshape(-1)).pnorm(
        PNORM, normalize=True
    )
    g_vmx = (cellv / ((2.0 - rho_e) * V0MAX)).pnorm(PNORM, normalize=True)  # eq. (26)
    g_vmn = ((2 * V0MAX - cellv) / (2 * V0MAX - V0MIN)).pnorm(PNORM, normalize=True)  # eq. (28)
    perim = rho.perimeter(zeta=0.1)

    terms, callbacks = [compliance], []
    if PSTAR > 0:
        beta_p = jno.np.parameter((1,), name="beta")
        beta_p.dtype(jnp.float64)
        beta_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, BETA))
        beta_p.optimizer(optax.sgd(1.0))
        terms.append((beta_p[0] * perim.log_barrier(PSTAR)).name("R"))
        callbacks.append(
            jno.optimizers.geometric_decay(beta_p, GAMMA, start=BETA, minimum=BETA_MIN)
        )
    # On a pinned mesh the element-quality rows are constants of the mesh -- the trap documented
    # at length in bracket.py -- so only the volume budget survives RELOCATE=0.
    terms.append(jno.le(volume, 1.0))
    if RELOCATE:
        terms += [jno.le(g_ang, 1.0), jno.le(g_vmx, 1.0), jno.le(g_vmn, 1.0)]

    crux = jno.core(terms, domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    return dict(d=d, crux=crux, rho=rho, fem=fem, callbacks=callbacks,
                compliance=compliance, volume=volume, g_ang=g_ang, g_vmx=g_vmx, perim=perim)


print("=" * 82, flush=True)
print(f"  CANTILEVER 2-D  {L:.0f}x{H:.0f}  h={HMESH}  tris={NT}  nodes={NN}  SPACE={SPACE}", flush=True)
print(f"          volfrac={VOLFRAC}  V0=[{V0MIN},{V0MAX}]  move=+/-{MOVE_BOUND}  P*={PSTAR}", flush=True)
print(f"          paper: 4108 tris, 12870 dofs (E-FEM), C=80.809 (E-FEM) / 75.965 (conventional)", flush=True)
print("=" * 82, flush=True)

# Their Fig. 4 loop: converge C at this exponent, and if the design is still grey, raise the
# exponent by one and carry the design across. Each stage is a fresh build because the exponent is
# a literal; the design is carried by warm-starting the RAW density and the node coordinates.
t0, penal, warm_rho, warm_coords, hist, total_it = time.time(), PENAL0, None, None, [], 0
stages = []
while penal <= PENAL_MAX:
    B = build(penal, warm_rho, warm_coords)
    crux, ev = B["crux"], (lambda n: float(np.asarray(B["crux"].eval([n])).reshape(-1)[0]))
    if not stages:
        print(f"  dofs {B['fem'].dofs}   (their E-FEM count is 12,870)", flush=True)
    it, prev, stage_hist = 0, None, []
    while it < STAGE_ITERS:
        crux.solve(CHUNK, callbacks=B["callbacks"])
        it += CHUNK
        total_it += CHUNK
        C = ev(B["compliance"])
        rphys = np.asarray(crux.eval([B["rho"]])).reshape(-1)
        mnd = float(np.mean(4 * rphys * (1 - rphys)))
        stage_hist.append(C)
        hist.append((total_it, penal, C, mnd, ev(B["volume"])))
        print(f"  p={penal:.0f}  it {total_it:4d}  C={C:10.4f}  V={ev(B['volume']):.4f}  "
              f"M_nd={mnd:.5f}  g_ang={ev(B['g_ang']):.3f} vmx={ev(B['g_vmx']):.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if prev is not None and abs(C - prev) <= CTOL * max(abs(prev), 1e-30):
            break
        prev = C
    stages.append((penal, float(np.mean(stage_hist[-4:])), mnd))
    warm_rho = raw_density(crux, NT)
    warm_coords = {
        int(sp["axis"]): np.asarray(crux.eval([sp["expr"]])).reshape(-1)
        for sp in (getattr(B["d"], "_trainable_coords", []) or [])
    } or None
    if mnd <= MND_TOL:
        print(f"  M_nd {mnd:.2e} <= {MND_TOL:g} at penal {penal:.0f}: their stopping rule is met",
              flush=True)
        break
    penal += 1.0

rho_f = np.asarray(crux.eval([B["rho"]])).reshape(-1)
pts_f = PTS0.copy()
for spec in getattr(B["d"], "_trainable_coords", []) or []:
    pts_f[np.asarray(spec["ids"], dtype=int), int(spec["axis"])] = np.asarray(
        crux.eval([spec["expr"]])
    ).reshape(-1)
area = np.abs(np.linalg.det(np.stack(
    [pts_f[CELLS][:, 1] - pts_f[CELLS][:, 0], pts_f[CELLS][:, 2] - pts_f[CELLS][:, 0]], -1))) / 2

TARGET = {"cover": 80.809, "Lagrange": 75.965}[SPACE]
# Report a WINDOW, never the last iterate. Before staged continuation the conventional arm swung
# 23.9 % over its last 200 iterations, so its final value was wherever the run happened to stop --
# quoting that as "converged" turns a coin flip into an agreement (it flattered +14.0 % to +1.8 %).
_w = np.array([h[2] for h in hist[-8:]])
_swing = 100 * (_w.max() - _w.min()) / _w.mean()
print(f"\n  stages (penal, mean C, M_nd): " +
      "  ".join(f"[{p:.0f}: {c:.3f}, {m:.4f}]" for p, c, m in stages), flush=True)
print(f"  C over the last {len(_w)} chunks: mean {_w.mean():.4f}  "
      f"range [{_w.min():.4f}, {_w.max():.4f}]  swing {_swing:.1f} %", flush=True)
print(f"  their Fig. 7 gives {TARGET}  ->  window mean {100 * (_w.mean() / TARGET - 1):+.2f} %",
      flush=True)
if _swing > 5.0:
    print(f"  NOT CONVERGED: a {_swing:.1f} % swing means no single value here is the answer; "
          f"the window mean summarises an oscillation, not a converged compliance", flush=True)
print(f"  dofs {B['fem'].dofs} (theirs 12,870)   elements {NT} (theirs 4,108)   "
      f"iterations {total_it} (theirs 143)", flush=True)
_mnd = float(np.mean(4 * rho_f * (1 - rho_f)))
print(f"  volume fraction {np.sum(rho_f * area) / np.sum(area):.4f} (theirs 0.400)   "
      f"M_nd {_mnd:.5f} (their rule stops at {MND_TOL:g})   "
      f"inverted elements {int((area <= 0).sum())}", flush=True)
if _mnd > MND_TOL:
    print(f"  STILL GREY: the exponent reached {penal:.0f} and M_nd is {_mnd:.4f}, so this design "
          f"is NOT the binary one their {TARGET} describes -- read the comparison with that in mind",
          flush=True)

np.savez_compressed(
    os.path.join(OUT, f"topo2d_{TAG}.npz"),
    cells=CELLS, pts0=PTS0, pts_f=pts_f, rho_f=rho_f,
    hist=np.array(hist), C=_w.mean(), dofs=B["fem"].dofs, space=SPACE, target=TARGET,
)
print(f"  saved topo2d_{TAG}.npz", flush=True)
