"""The whole 3-D deformable-mesh topology optimisation, as jNO API and nothing else.

This is `run.py` with every environment knob, diagnostic and figure hook stripped out, pinned to the
configuration that produced `topo3d_p2d_best.npz` (C = 244, P = 1058, M_nd = 0.0030). It is meant to
be read: the point is that the 3-D case needs NO new public API over the published 2-D cantilever
tutorial -- the same calls, with a box instead of a rectangle and three coordinate blocks instead
of two.

Method: Jung, Yun & Kim, "Density-based topology optimization on a deformable unstructured mesh
with enriched finite elements", Computers & Structures 331 (2026) 108403 -- their eq. (17)-(19)
patch filter, eq. (26)/(28) element-volume bounds, eq. (38)-(41) perimeter barrier. The perimeter
functional is Haber, Jog & Bendsoe, Struct. Optim. 11 (1996) 1-12. The Heaviside projection is
Wang, Lazarov & Sigmund, Struct. Multidisc. Optim. 43 (2011) 767-784, after Guest (2004).

    python scripts/minimal.py
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pardiso_solver import pardiso_lu  # noqa: E402

inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace

# A 60 x 30 x 2 slab: pseudo-2D, two elements thick, z-faces traction-free. Its answer is known --
# it must reproduce the classic 2-D cantilever truss -- which is what makes it the validation case.
L, H, W, h = 60.0, 30.0, 2.0, 1.0
E0, NU, PENAL, EMIN, VOLFRAC = 1.0, 0.3, 3.0, 1e-4, 0.40
LAM, MU = E0 * NU / ((1 + NU) * (1 - 2 * NU)), E0 / (2 * (1 + NU))
SPAN, TOL, PN = 2.0, 1e-9, 50.0
THETA_MIN, ASPECT_MAX, MOVE_BOUND = np.deg2rad(5.0), 8.0, 2.0 * h
PSTAR, BETA, GAMMA, BETA_MIN = 1100.0, 1e-3, 0.997, 5e-4   # perimeter target + barrier schedule
BETA_H, ETA_H, RHO_MIN = 16.0, 0.5, 1e-3                   # Heaviside projection
ITERS, CHUNK = 540, 10

d = jno.Shape.box(0, 0, 0, L, H, W, size=h).domain()
cells, pts0 = np.asarray(d._cells_p1()), np.asarray(d.mesh.points)[:, :3]

# --- design variables 2-4: the interior nodal coordinates -------------------------------------
# This is the deformable mesh. Each interior node's x, y, z becomes a design variable with its own
# box bounds, so nodes MIGRATE TO THE MATERIAL BOUNDARY -- the paper's mechanism, and the reason
# the bound is a resolution setting rather than a safety knob.
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

# `penal` is a runtime parameter so the SIMP continuation can raise it without a recompile.
penal_p = jno.np.parameter((1,), name="penal")
penal_p.dtype(jnp.float64)
penal_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, PENAL))
penal_p.optimizer(optax.sgd(1.0))

# Solid, non-designable material under the load: without it the optimiser empties the loaded
# elements and the compliance measures that artefact instead of the structure.
cen0 = pts0[cells].mean(axis=1)
pmask = jnp.asarray((cen0[:, 0] > L - SPAN) & (cen0[:, 1] < SPAN))

# THE PHYSICAL DENSITY. `d.patch_filter()` is eq. (17)-(19) -- in 3-D it runs over EDGE FANS, and
# that dispatch is internal: the call is identical to the 2-D tutorial's. It REPLACES a density
# filter rather than supplementing one. The Heaviside projection is composed on top of it, its beta
# ramped so the topology forms before it sets. Floored at RHO_MIN because d/dp (rho^p) carries a
# ln(rho) that is NaN at exactly zero, and the patch filter does reach zero.
patch = d.patch_filter()
beta_h = [1.0]   # a list so the closure below sees the ramp


def physical(r):
    x = jnp.where(pmask, 1.0, patch(r))
    tb = jnp.tanh(beta_h[0] * ETA_H)
    proj = (tb + jnp.tanh(beta_h[0] * (x - ETA_H))) / (tb + jnp.tanh(beta_h[0] * (1.0 - ETA_H)))
    return jnp.clip(proj, RHO_MIN, 1.0)


rho.constrain(physical)

# --- the physics: linear elasticity, clamped root, traction band at the tip --------------------
xi, yi, zi, _ = d.variable("interior", split=True)
root = d.variable("root", where=lambda x, y, z: x < TOL, split=True)[:3]
tip = d.variable("tip", where=lambda x, y, z: (x > L - TOL) & (y < SPAN), split=True)[:3]
eps = lambda w: symgrad(w, [xi, yi, zi])  # noqa: E731
a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731
E = lambda r: EMIN + r**penal_p * (E0 - EMIN)  # noqa: E731  SIMP
fem = jno.fem([E(rho) * a(eps(u), eps(phi)),
               u(*root) - (0.0, 0.0, 0.0),
               -1.0 * inner(jnp.array([0.0, -1.0, 0.0]), phi.bind(x=tip[0], y=tip[1], z=tip[2]), 1)],
              quad_degree=2)

# --- objective and constraints, every one an expression over the SAME mesh nodes ---------------
# `.integrate(fem, solver=)` evaluates a test-free expression on the FE solution, differentiable in
# the solution, the density AND the mesh coordinates -- which is what lets one objective drive all
# four design blocks. `solver=pardiso_lu` is a plain callable; see pardiso_solver.py for why.
compliance = (E(rho) * a(eps(u), eps(u))).integrate(fem, solver=pardiso_lu)
cellv = d.cell_volume()
volume = (rho.reshape(-1) * cellv).sum / (VOLFRAC * cellv.sum)

# Mesh-quality constraints. p-norm aggregates with normalize=True, otherwise an unnormalised
# aggregate over this many elements reads as violated from iteration 0 and freezes MMA.
g_ang = ((2 * np.pi - d.cell_angles()) / (2 * np.pi - THETA_MIN)).reshape(-1).pnorm(PN, normalize=True)
g_asp = (d.cell_aspect() / ASPECT_MAX).pnorm(PN, normalize=True)

# eq. (26) and (28): element-volume bounds, DENSITY-COUPLED through `(2 - rho)`. A void element may
# grow toward twice V0MAX, a solid one may not -- that coupling is what coarsens the mesh in void
# and keeps it fine in material. Calibrated from this mesh's own percentiles, not from the regular
# tetrahedron: a Delaunay tet mesh has a median cell 1.26x and a largest 5.80x the ideal, so
# normalising by the ideal element makes 27.5 % of cells infeasible at rho = 1 on the initial mesh.
_v0 = np.abs(np.linalg.det(np.stack([pts0[cells][:, i + 1] - pts0[cells][:, 0] for i in range(3)], -1))) / 6.0
V0MAX, V0MIN = 1.30 * float(np.percentile(_v0, 99)), 0.50 * float(np.percentile(_v0, 1))
g_vmx = (cellv / ((2.0 - rho.reshape(-1)) * V0MAX)).pnorm(PN, normalize=True)
g_vmn = ((2 * V0MAX - cellv) / (2 * V0MAX - V0MIN)).pnorm(PN, normalize=True)

# eq. (38)-(41): the feature-scale lever. An INTERIOR PENALTY R = -beta*log(P* - P), not a jno.le
# constraint, with beta decaying geometrically so the barrier stops distorting the converged
# optimum. In 3-D `perimeter()` measures the material boundary's AREA, so P* carries area units.
# The BETA_MIN floor decides whether the target still binds at the end: with the paper's 1e-4 the
# perimeter overran P* and the design spent the slack on fragmenting the interior.
perim = rho.perimeter(zeta=0.1)
beta_p = jno.np.parameter((1,), name="beta")
beta_p.dtype(jnp.float64)
beta_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, BETA))
beta_p.optimizer(optax.sgd(1.0))

terms = [compliance,
         (beta_p[0] * perim.log_barrier(PSTAR)).name("R"),
         jno.le(volume, 1.0),
         jno.le(g_ang, 1.0), jno.le(g_asp, 1.0), jno.le(g_vmx, 1.0), jno.le(g_vmn, 1.0)]
callbacks = [jno.optimizers.simp_continuation(penal_p, rho, physical=patch, every=25, watch=0, patience=6),
             jno.optimizers.geometric_decay(beta_p, GAMMA, start=BETA, minimum=BETA_MIN)]

crux = jno.core(terms, domain=jno.domain.from_array({"_": np.zeros((1, 1))}))

for it in range(0, ITERS, CHUNK):
    crux.solve(CHUNK, callbacks=callbacks)
    beta_h[0] = min(BETA_H, beta_h[0] * 1.10)
    C, V, P = (float(np.asarray(crux.eval([n])).reshape(-1)[0]) for n in (compliance, volume, perim))
    r = np.asarray(crux.eval([rho])).reshape(-1)
    print(f"  it {it + CHUNK:4d}  C={C:9.3f}  V={V:.4f}  P={P:8.2f}  "
          f"M_nd={4.0 * np.mean(r * (1.0 - r)):.4f}  beta_H={beta_h[0]:.1f}", flush=True)
