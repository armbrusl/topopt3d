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

**THE COMPLIANCE COLUMN IS NOT DIRECTLY COMPARABLE, and that is not a detail.** Compliance scales
as the square of the applied load, and the paper never states its load magnitude. Measured here on
the E-FEM cantilever, changing only the declared load:

    |F| = 0.9153   C = 80.667   -0.18 % against their 80.809
    |F| = 1.0      C = 95.643  +18.4 %  against their 80.809

Both are the same design at the same settings; the ratio 1.186 is the load ratio squared (1.194).
An earlier version of this script left the coefficient at 1/SPAN, which happened to assemble
|F| = 0.9153, and reported the resulting -0.18 % as agreement. It was not: it was an unmeasured
free parameter landing near their (unknown) value. Absolute compliance against this paper is
INDETERMINATE, and no number here should be quoted as matching theirs.

What survives is what does not depend on the load: the dof count above, the ratio between the two
formulations, the reanalysis gap (reported vs an independent re-solve of the same design, their
Fig. 7c/d), and the convergence behaviour. Those are the comparisons to make.

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
PROBLEM = os.environ.get("PROBLEM", "cantilever")  # "cantilever" (Fig. 5/7) | "lbracket" (Fig. 9)
TAG = os.environ.get("TAG", f"p2d_{PROBLEM}_{SPACE}")
HMESH = float(os.environ.get("HMESH", "1.0"))  # "target edge length set to unity"
E0, EMIN, NU = 1.0, 1e-9, 0.3
LAM, MU = E0 * NU / (1 - NU**2), E0 / (2 * (1 + NU))  # PLANE STRESS -- their ABAQUS CPS4

# Their two 2-D benchmarks, each with the numbers printed in its own figure. `elements` is their
# count and is a check on the mesh, not an input: at a unit target edge the L-bracket's area
# 40^2 - 24^2 = 1024 gives ~2,365 triangles against their 2,320.
_TOL = 1e-6
PROBLEMS = {
    # Fig. 5a: L x 0.5L, clamped on the whole left edge, point load down at the bottom-right.
    "cantilever": dict(
        L=60.0, H=30.0, volfrac=0.40, elements=4108, iters=143,
        target={"cover": 80.809, "Lagrange": 75.965},
        clamp=lambda x, y, L=60.0, H=30.0: x < _TOL * L,
        tip=lambda x, y, L=60.0, H=30.0, s=2.0: (x > L - _TOL * L) & (y < s + _TOL * L),
    ),
    # Fig. 9a: the full L x L square minus its upper-right 0.6L x 0.6L block, clamped along the
    # TOP of the vertical arm, loaded downward at the free end of the horizontal arm.
    "lbracket": dict(
        L=40.0, H=40.0, volfrac=0.35, elements=2320, iters=196,
        target={"cover": 220.578, "Lagrange": None},  # they publish only the proposed method here
        clamp=lambda x, y, L=40.0, H=40.0: y > H - _TOL * H,
        # The free END of the horizontal arm -- an EDGE segment, exactly like the cantilever's.
        # Written as an area patch first, it caught 8 nodes against the cantilever's 3, so the two
        # benchmarks were carrying |F| = 4.0 against 1.5. Compliance goes as F^2, so that is a 7x
        # distortion between problems that are supposed to differ only in geometry.
        tip=lambda x, y, L=40.0, H=40.0, s=2.0: (x > L - _TOL * L) & (y > 0.4 * H - s),
    ),
}
if PROBLEM not in PROBLEMS:
    raise SystemExit(f"PROBLEM must be one of {sorted(PROBLEMS)}, not {PROBLEM!r}")
P = PROBLEMS[PROBLEM]
L, H = P["L"], P["H"]
VOLFRAC = float(os.environ.get("VOLFRAC", str(P["volfrac"])))
PENAL0 = float(os.environ.get("PENAL0", "3"))  # "the initial penalization factor is set to 3"
PENAL_MAX = float(os.environ.get("PENAL_MAX", "8"))  # a cap; their rule below is what stops it
THETA_MIN, PNORM = np.deg2rad(20.0), 50.0  # "theta_min = 20 deg", "p = 50"
V0MAX, V0MIN = 0.75 * HMESH**2, 0.05 * HMESH**2  # "V0,max = 0.75 and V0,min = 0.05"
MOVE_BOUND = 2.0 * HMESH  # "movement design variables are bounded between -2 and 2"
SPAN, TOL = 2.0 * HMESH, 1e-6 * L
# Total applied force. The paper never states it, and compliance scales as its square, so it is a
# free multiplier on every number here -- fixed at 1.0 so the two benchmarks are at least
# comparable to each other, and so a changed tip predicate cannot silently rescale the objective.
LOAD = float(os.environ.get("LOAD", "1.0"))
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
# The cantilever's 650 is jNO's tutorial value for that domain. The L-bracket has neither a
# published nor an inherited one, so it is scaled by the blank's area (perimeter is a length, so
# it scales as sqrt(area)) purely to have SOMETHING active, and is a calibration target, not a
# measurement: see "Calibrating PSTAR" in the README for the probe-then-set procedure.
_PSTAR_DEFAULT = {"cantilever": 650.0, "lbracket": 650.0 * (1024.0 / 1800.0) ** 0.5}[PROBLEM]
PSTAR = float(os.environ.get("PSTAR", str(_PSTAR_DEFAULT)))
BETA, GAMMA, BETA_MIN = 2e-4, 0.997, 1e-4

def geometry():
    """The blank. The L-bracket is the square minus its upper-right block, via Shape CSG."""
    sq = jno.Shape.rect(0, 0, L, H, size=HMESH)
    if PROBLEM == "lbracket":
        sq = sq - jno.Shape.rect(0.4 * L, 0.4 * H, L, H, size=HMESH)
    return sq.domain()


_probe = geometry()
PTS0 = np.asarray(_probe.mesh.points)[:, :2]
CELLS = np.asarray(_probe._cells_p1())
NT, NN = CELLS.shape[0], PTS0.shape[0]
del _probe
BND_XY = None  # filled once `_boundary_xy` is defined, below


def _boundary_xy():
    """The blank's boundary nodes: edges of the P1 triangulation used exactly once.

    Computed from connectivity rather than from plane tests, because the L-bracket's re-entrant
    corner is not a box and hand-written planes would silently free the nodes along its two inner
    edges -- which is the same class of mistake as the 3-D support drift: nothing fails, the
    optimiser just quietly gets to move a boundary it was never meant to move.
    """
    e = np.sort(np.stack([CELLS[:, [0, 1]], CELLS[:, [1, 2]], CELLS[:, [0, 2]]], 1).reshape(-1, 2), axis=1)
    _u, cnt = np.unique(e, axis=0, return_counts=True)
    return PTS0[np.unique(_u[cnt == 1])]


def _on_boundary(x, y):
    """Exact coordinate match against the blank's boundary nodes."""
    Q = np.stack([np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)], -1)
    out = np.zeros(Q.shape[0], dtype=bool)
    for i in range(0, Q.shape[0], 1024):  # chunked: the pairwise block is O(chunk * n_boundary)
        blk = Q[i : i + 1024]
        out[i : i + 1024] = ((blk[:, None, :] - BND_XY[None, :, :]) ** 2).sum(-1).min(axis=1) < 1e-18
    return out


BND_XY = _boundary_xy()
_TIPN = int(np.asarray(P["tip"](PTS0[:, 0], PTS0[:, 1])).sum())
_CLAMPN = int(np.asarray(P["clamp"](PTS0[:, 0], PTS0[:, 1])).sum())
if _TIPN == 0:
    raise SystemExit(f"{PROBLEM}: the tip predicate caught no nodes -- the load would be zero")
if _CLAMPN == 0:
    raise SystemExit(f"{PROBLEM}: the clamp predicate caught no nodes -- the problem is unsupported")
LOAD_COEFF = 1.0  # provisional; replaced by the probe below once a fem exists to measure


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
    d = geometry()
    # Interior = not on the outer boundary, whatever that boundary is. Written as a distance to
    # the blank's own boundary nodes rather than as plane tests, because the L-bracket's re-entrant
    # corner is not expressible as a box and hand-written planes would silently free its edges.
    interior = lambda x, y: ~_on_boundary(x, y)  # noqa: E731
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
    xl, yl, _ = d.variable("clamp", where=P["clamp"], split=True)
    xt, yt, _ = d.variable("tip", where=P["tip"], split=True)

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

    # Solid, non-designable material under the load. Without it the optimiser empties the loaded
    # elements and the compliance measures that artefact rather than the structure. Built from the
    # problem's OWN tip predicate so it follows the load wherever the problem puts it.
    cen0 = PTS0[CELLS].mean(axis=1)
    pmask = jnp.asarray(np.asarray(P["tip"](cen0[:, 0], cen0[:, 1])).reshape(-1))
    patch = d.patch_filter()  # eq. (17)-(19)
    rho.constrain(lambda r: jnp.where(pmask, 1.0, patch(r)))

    eps = lambda w: symgrad(w, [xi, yi])  # noqa: E731
    a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731
    E = lambda r: EMIN + r**penal * (E0 - EMIN)  # noqa: E731  -- LITERAL exponent

    # The load coefficient is scaled so the ASSEMBLED resultant is exactly LOAD. Counting tip
    # nodes and dividing is wrong -- the term is a boundary integral, not a sum of nodal forces,
    # so the node count is not its magnitude. Getting that backwards put |F| = 4.0 on the
    # L-bracket against 1.5 on the cantilever, and then over-corrected the other way; compliance
    # goes as F^2, so both errors rescale every number the run reports. Measure the resultant
    # from the assembled right-hand side, exactly as `bracket.py` does in 3-D.
    fem = jno.fem(
        [
            E(rho) * a(eps(u), eps(phi)),
            u(xl, yl) - (0.0, 0.0),
            -1.0 * inner(jnp.array([0.0, -LOAD_COEFF]), phi.bind(x=xt, y=yt), n_contract=1),
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


# --- calibrate the load coefficient so the assembled resultant is exactly LOAD ----------------
_probe_B = build(PENAL0)
_A, _b = _probe_B["fem"].operator.evaluate({"rho": jnp.full(NT, VOLFRAC)})
_f = np.asarray(jnp.asarray(_b)).reshape(-1, 2)
_res = _f.sum(axis=0)
_mag = float(np.linalg.norm(_res))
if not np.isfinite(_mag) or _mag < 1e-12:
    raise SystemExit(f"{PROBLEM}: the tip band assembled a zero load resultant ({_res}) -- a band "
                     f"narrower than the mesh spacing contains no whole facet and integrates to "
                     f"nothing, silently")
LOAD_COEFF = LOAD / _mag
print(f"  load probe: coefficient 1.0 assembles |F| = {_mag:.4f} "
      f"(direction {np.round(_res / _mag, 3)}); scaling by {LOAD_COEFF:.4f} for |F| = {LOAD:.3f}",
      flush=True)
del _probe_B, _A, _b, _f

_tgt = P["target"]
_tgt_s = "  ".join(f"{k}={v}" for k, v in _tgt.items() if v is not None) or "(none published)"
print("=" * 82, flush=True)
print(f"  {PROBLEM.upper()} 2-D  {L:.0f}x{H:.0f}  h={HMESH}  tris={NT}  nodes={NN}  "
      f"SPACE={SPACE}", flush=True)
print(f"          volfrac={VOLFRAC}  V0=[{V0MIN},{V0MAX}]  move=+/-{MOVE_BOUND}  "
      f"P*={PSTAR:.1f}{' (calibration target, not theirs)' if PROBLEM == 'lbracket' else ''}",
      flush=True)
print(f"          paper: {P['elements']} elements, {P['iters']} iterations, C: {_tgt_s}", flush=True)
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
        print(f"  dofs {B['fem'].dofs}   tip nodes {_TIPN}   |F| = {LOAD:.3f}   "
              f"clamp nodes {_CLAMPN}", flush=True)
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

TARGET = P["target"][SPACE]  # None where the paper publishes no number for that arm
# Report a WINDOW, never the last iterate. Before staged continuation the conventional arm swung
# 23.9 % over its last 200 iterations, so its final value was wherever the run happened to stop --
# quoting that as "converged" turns a coin flip into an agreement (it flattered +14.0 % to +1.8 %).
_w = np.array([h[2] for h in hist[-8:]])
_swing = 100 * (_w.max() - _w.min()) / _w.mean()
print(f"\n  stages (penal, mean C, M_nd): " +
      "  ".join(f"[{p:.0f}: {c:.3f}, {m:.4f}]" for p, c, m in stages), flush=True)
print(f"  C over the last {len(_w)} chunks: mean {_w.mean():.4f}  "
      f"range [{_w.min():.4f}, {_w.max():.4f}]  swing {_swing:.1f} %", flush=True)
if TARGET is None:
    print(f"  the paper publishes no {SPACE} number for the {PROBLEM}, so there is nothing to "
          f"compare this against -- it stands on its own", flush=True)
else:
    print(f"  the paper gives {TARGET}  ->  window mean {100 * (_w.mean() / TARGET - 1):+.2f} %",
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
          f"is NOT the binary one the paper's numbers describe -- read any comparison with that "
          f"in mind", flush=True)

np.savez_compressed(
    os.path.join(OUT, f"topo2d_{TAG}.npz"),
    cells=CELLS, pts0=PTS0, pts_f=pts_f, rho_f=rho_f,
    hist=np.array(hist), C=_w.mean(), dofs=B["fem"].dofs, space=SPACE, target=TARGET,
)
print(f"  saved topo2d_{TAG}.npz", flush=True)
