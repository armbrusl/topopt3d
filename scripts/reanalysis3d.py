"""The paper's own fidelity check, in 3-D: does the design survive on a clean mesh?

An optimiser that moves nodes can lower compliance two ways -- by improving the structure, or by
distorting elements until they under-integrate strain energy -- and it cannot tell those apart from
the inside. Jung, Yun & Kim's answer (Fig. 7) is to take the converged design, transfer it to an
undistorted mesh, and re-solve. With E-FEM they measure 80.809 -> 80.411, a gap of -0.5%. With
conventional elements, 75.965 -> 89.339: +17.6%, the optimiser having bought its compliance with
mesh distortion rather than structure.

We have never run this in 3-D, and every 3-D result so far uses the conventional formulation, so
this measures how much of our 3-D compliance is real.

**The uniform-density control is not optional.** Transferring between two different meshes changes
the answer by itself -- different discretisation, different quadrature, a piecewise-constant density
resampled across cells. Running the same transfer on a design-free uniform field measures that, so a
pure meshing effect cannot be reported as a method result. The number that means something is the
design's gap MINUS the control's.

    RUN=topo3d_align.npz python scripts/reanalysis3d.py
    RUN=topo3d_align.npz HCLEAN=0.12 python scripts/reanalysis3d.py    # finer clean mesh
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jno  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pardiso_solver import pardiso_lu  # noqa: E402

inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

LX, LY, LZ, SHOULDER, TOPL, TOPR = 4.0, 3.0, 1.5, 1.0, 1.4, 2.6
BX1, BX2, BY, RB = 0.8, 3.2, 0.6, 0.35
E0, NU, PENAL = 1.0, 0.3, 3.0
EMIN = float(os.environ.get("EMIN", "1e-4"))
LAM, MU = E0 * NU / ((1 + NU) * (1 - 2 * NU)), E0 / (2 * (1 + NU))
VOLFRAC = float(os.environ.get("VOLFRAC", "0.30"))
TOL = 1e-9
PULL = np.array([0.30, -0.85, 0.43])
PULL = PULL / np.linalg.norm(PULL)
EYE = float(os.environ.get("EYE", "0.20"))

RUN = os.environ.get("RUN", "topo3d_align.npz")
r = np.load(os.path.join(OUT, RUN))
cells_d, pts_d, rho_d = r["cells"], r["pts_f"], r["rho_f"]
H_DESIGN = float(os.environ.get("HMESH", "0.15"))
H_CLEAN = float(os.environ.get("HCLEAN", str(H_DESIGN)))
print(f"design: {RUN}  {cells_d.shape[0]} tets, volfrac {float((rho_d * 1.0).mean()):.3f}")


def build(h):
    blank = jno.Shape.polygon(
        [(0.0, 0.0), (LX, 0.0), (LX, SHOULDER), (TOPR, LY), (TOPL, LY), (0.0, SHOULDER)], size=h
    ).extrude(LZ)
    _bolt = lambda bx: jno.Shape.cylinder(bx, BY, -0.1, 0.0, 0.0, LZ + 0.2, RB, size=h)  # noqa: E731
    return (blank - _bolt(BX1) - _bolt(BX2)).domain()


def compliance_of(d, rho_vals, pts_override=None, tag=""):
    """Solve the bracket on domain ``d`` with a FIXED density and return the compliance.

    ``pts_override`` moves the mesh nodes to the design's converged coordinates, so the same
    routine measures both 'on the mesh the optimiser left' and 'on a clean one'.
    """
    cells = np.asarray(d._cells_p1())
    pts0 = np.asarray(d.mesh.points)[:, :3]
    if pts_override is not None:
        d.mesh.points[:, :3] = pts_override  # the deformed configuration
        pts0 = np.asarray(d.mesh.points)[:, :3]

    u, phi = d.fem_symbols(value_shape=(3,))
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    rho = jno.np.parameter(s, name=f"rho_re_{tag}")
    rho.dtype(jnp.float64)
    vals = jnp.asarray(np.asarray(rho_vals, dtype=float).reshape(-1))
    rho.initialize(lambda k, sh, dtype=None, _v=vals: _v.reshape(sh))

    xi, yi, zi, _ = d.variable("interior", split=True)  # built-in pool tag; each domain is fresh
    bolts = d.variable(
        "boltwalls",
        where=lambda x, y, z: (jnp.hypot(x - BX1, y - BY) < RB + 0.10) | (jnp.hypot(x - BX2, y - BY) < RB + 0.10),
        split=True,
    )[:3]
    eye = d.variable(
        "ropepatch",
        where=lambda x, y, z: (y > LY - TOL) & (x > TOPL - TOL) & (x < TOPR + TOL),
        split=True,
    )[:3]
    eps = lambda w: symgrad(w, [xi, yi, zi])  # noqa: E731
    a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731
    E = lambda rr: EMIN + rr**PENAL * (E0 - EMIN)  # noqa: E731

    # The load patch's area differs between meshes, so normalise by the ASSEMBLED resultant rather
    # than by a nominal area -- otherwise the two meshes are pulled by different total forces and
    # the comparison measures that instead of the design.
    LF = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    seen: dict = {}
    for t in range(cells.shape[0]):
        c = cells[t]
        for f in LF:
            k = tuple(sorted(int(c[i]) for i in f))
            seen.pop(k, None) if k in seen else seen.setdefault(k, t)
    tri = np.array(list(seen.keys()), dtype=int)
    keep = ((pts0[tri][..., 1] > LY - 1e-6) & (pts0[tri][..., 0] > TOPL - 1e-6) & (pts0[tri][..., 0] < TOPR + 1e-6)).all(axis=1)
    v = pts0[tri[keep]]
    area = float(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2.0)
    assert area > 1e-12, "the rope patch has no facets on this mesh"

    fem = jno.fem(
        [
            E(rho) * a(eps(u), eps(phi)),
            u(*bolts) - (0.0, 0.0, 0.0),
            -1.0 * inner(jnp.asarray(PULL / area), phi.bind(**dict(zip("xyz", eye))), 1),
        ],
        quad_degree=2,
    )
    C = (E(rho) * a(eps(u), eps(u))).integrate(fem, solver=pardiso_lu)
    crux = jno.core([C], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    return float(np.asarray(crux.eval([C])).reshape(-1)[0]), area


# --- the four solves ------------------------------------------------------------------------------
d_def = build(H_DESIGN)
n_def = np.asarray(d_def._cells_p1()).shape[0]
assert n_def == cells_d.shape[0], f"rebuilt mesh has {n_def} tets, the design has {cells_d.shape[0]}"

d_clean = build(H_CLEAN)
n_clean = np.asarray(d_clean._cells_p1()).shape[0]
print(f"clean mesh at h={H_CLEAN}: {n_clean} tets  (design mesh {n_def})")

def _vol(d, pts=None):
    c = np.asarray(d._cells_p1())
    p = np.asarray(d.mesh.points)[:, :3] if pts is None else pts
    return np.abs(np.linalg.det(np.stack([p[c][:, i + 1] - p[c][:, 0] for i in range(3)], -1))) / 6.0


# Preserve the MATERIAL VOLUME across the transfer. A piecewise-constant density resampled onto a
# different mesh does not conserve it -- measured 0.362 -> 0.331, an 8.5% loss -- and a design with
# 8.5% less material is stiffer for a reason that has nothing to do with mesh distortion. The paper
# does the same: the reanalysis geometry is "reconstructed from an iso-density contour while
# preserving the target volume fraction".
_v_def, _v_cln = _vol(d_def, pts_d), _vol(d_clean)
rho_raw = np.asarray(d_def.transfer_cell_field(rho_d, d_clean, points=pts_d))
_mat_want = float((rho_d * _v_def).sum())
_scale = _mat_want / max(float((rho_raw * _v_cln).sum()), 1e-300)
rho_clean = np.clip(rho_raw * _scale, 1e-3, 1.0)
print(f"transferred density: material volume {_mat_want:.4f} -> {float((rho_raw * _v_cln).sum()):.4f} "
      f"raw, rescaled by {_scale:.4f} -> {float((rho_clean * _v_cln).sum()):.4f}")

uni = np.full(cells_d.shape[0], VOLFRAC)
uni_raw = np.asarray(d_def.transfer_cell_field(uni, d_clean, points=pts_d))
_um = float((uni * _v_def).sum())
uni_clean = np.clip(uni_raw * (_um / max(float((uni_raw * _v_cln).sum()), 1e-300)), 1e-3, 1.0)

rows = []
for tag, dom, vals, pts_ov in (
    ("design, deformed mesh", build(H_DESIGN), rho_d, pts_d),
    ("design, CLEAN mesh", build(H_CLEAN), rho_clean, None),
    ("uniform, deformed mesh", build(H_DESIGN), uni, pts_d),
    ("uniform, CLEAN mesh", build(H_CLEAN), uni_clean, None),
):
    C, area = compliance_of(dom, vals, pts_ov, tag=tag.replace(" ", "_").replace(",", ""))
    rows.append((tag, C, area))
    print(f"  {tag:26s} C = {C:10.4f}   (load patch area {area:.4f})")

C_dd, C_dc, C_ud, C_uc = (x[1] for x in rows)
g_design = C_dc / C_dd - 1.0
g_uniform = C_uc / C_ud - 1.0
print()
print(f"design  gap on transfer : {100 * g_design:+7.2f} %")
print(f"control gap on transfer : {100 * g_uniform:+7.2f} %   <- pure meshing/transfer effect")
print(f"ATTRIBUTABLE to the design (design - control): {100 * (g_design - g_uniform):+7.2f} %")
print()
print("Reference, their Fig. 7 in 2-D:  E-FEM -0.5 %,  conventional elements +17.6 %.")
print("Our 3-D runs use the conventional formulation, so a large positive number here is expected")
print("and is the case for turning E-FEM on -- it is not yet a statement about the 3-D method.")
