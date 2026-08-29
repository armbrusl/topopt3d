"""Reanalysis on the EXTRACTED BOUNDARY -- the paper's Fig. 7 check, without a density transfer.

`reanalysis3d.py` answers this by resampling the density field onto a clean mesh. That works, but a
piecewise-constant field does not survive a mesh change intact: the control there moves a
design-free field by 4.5-7.4 %, and that noise sits underneath every number it reports.

The paper does not resample. For its own method it "directly utilizes the optimized mesh boundary
(Fig. 7c) ... without additional reconstruction or post-processing", and the reanalysis geometry is
"reconstructed from an iso-density contour while preserving the target volume fraction". So the
object being re-solved is a SOLID BODY whose surface is the optimiser's own element boundary -- not
a density field at all.

That is reproducible exactly here, because a tetrahedral element boundary is piecewise PLANAR:

  1. threshold rho. On a GREY design the level is chosen so the retained tets carry the design's
     own material volume. On a BINARY one that criterion is degenerate and flatters the design --
     every level in (0,1) picks the same tets, so the search can only meet the volume target by
     dropping toward zero and admitting void, which is then re-solved as solid and bridges gaps the
     design does not close. Binary designs are therefore extracted at rho > 0.5 and the material the
     repair drops is REPORTED, as the fragmentation measure it is. `BINARY_MND` sets the cutoff;
     `LEVEL=` overrides both.
  2. keep the largest face-connected piece, then drop tets until the boundary is a closed manifold;
  3. subdivide each boundary triangle 1 -> 4 at edge midpoints. Planar facets, so this changes the
     geometry by NOTHING -- it only buys resolution;
  4. let gmsh fill that closed surface with well-shaped tets, the input triangles preserved;
  5. re-solve the solid.

Because the surface is bit-identical between steps 2 and 5, the comparison splits into steps that
each move exactly one thing:

    C_opt    -> C_design  the optimiser's own solve -> a standard P1 read of it  (formulation)
    C_design -> C_kept    same mesh, same distortion; SIMP field -> solid        (material model)
    C_kept   -> C_match   same geometry, same ELEMENT COUNT, undistorted tets    (DISTORTION)
    C_kept   -> C_fine    same geometry, UNIFORMLY refined                        (total mesh error)

**C_match is the one that isolates distortion, and it is the reason this script exists.** Refining
a P1 tet mesh softens it whatever its shape, so `C_kept -> C_fine` charges plain coarseness to
distortion and reports a number several times too large. Holding the element count fixed removes
that: same geometry, same resolution, the only difference is element quality.

Every solve here uses the standard P1 formulation -- the stand-in for the paper's "FEA using
ABAQUS". C_opt is whatever the optimiser reported, E-FEM or not, so C_opt -> C_design measures the
enrichment on a frozen configuration.

    RUN=topo3d_h25_cover.npz python scripts/reanalysis_boundary.py
    RUN=topo3d_h25_cover.npz REFINE=3 python scripts/reanalysis_boundary.py   # deeper series
    RUN=topo3d_h25_cover.npz LEVEL=0.5 python scripts/reanalysis_boundary.py   # fixed level instead
    RUN=topo3d_hv_cover.npz BINARY_MND=0 python scripts/reanalysis_boundary.py  # force the old path
    RUN=topo3d_fxfine.npz REFINE=2 FEMCHUNK=64 python scripts/reanalysis_boundary.py  # low memory
    RUN=topo3d_fxfine.npz REFINE=2 LINSOLVE=iterative python scripts/reanalysis_boundary.py  # big
    RUN=topo3d_fxfine.npz REFINE=2 LEVEL_FROM=2 ... reanalysis_boundary.py  # that level, alone
"""

import collections
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402

import jno  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pardiso_solver import pardiso_lu  # noqa: E402

inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")
SCRATCH = os.environ.get("SCRATCH", OUT)

LX, LY, LZ, SHOULDER, TOPL, TOPR = 4.0, 3.0, 1.5, 1.0, 1.4, 2.6
BX1, BX2, BY, RB = 0.8, 3.2, 0.6, 0.35
E0, NU, PENAL = 1.0, 0.3, 3.0
EMIN = float(os.environ.get("EMIN", "1e-4"))
LAM, MU = E0 * NU / ((1 + NU) * (1 - 2 * NU)), E0 / (2 * (1 + NU))
TOL = 1e-9
PULL = np.array([0.30, -0.85, 0.43])
PULL = PULL / np.linalg.norm(PULL)

RUN = os.environ.get("RUN", "topo3d_h25_cover.npz")
TAG = RUN.replace("topo3d_", "").replace(".npz", "")
REFINE = int(os.environ.get("REFINE", "2"))  # levels of UNIFORM refinement of the extracted solid
# Bounds the element-assembly batch, nothing else -- a pure MEMORY knob, so a chunked run must
# return the identical compliance to an unchunked one. Needed because the refinement series grows
# 8x per level: on a 62 GB workstation a level-2 refill of a 16k-tet body (~700k tets) is killed
# during assembly while levels 0 and 1 fit easily. Verified against the unchunked reference on
# this design -- level 0 = 20.7456 and level 1 = 29.3477 either way.
FEMCHUNK = int(os.environ.get("FEMCHUNK", "0"))  # 0 = let jNO's own policy size the batch
# Direct vs iterative for the re-solve. PARDISO is the right default -- it is exact, and at these
# sizes it is fast. It does not survive the top of the refinement series: at level 2 of a 16k-tet
# body the mesh is 777,750 tets / 472,980 dofs, and a DIRECT sparse LU on 3-D elasticity fills in
# far past what a 62 GB workstation holds, so the process dies in the factorisation (FEMCHUNK only
# bounds ASSEMBLY, which is a different ceiling and already cleared by then).
#
# "iterative" hands the solve back to jNO's default: Jacobi-preconditioned matrix-free BiCGStab,
# which never forms a factor and RAISES on non-convergence rather than returning a plausible wrong
# number. It costs accuracy at the 1e-8 tolerance rather than machine precision, which is four
# orders below any difference this script reports. Verified against the direct reference on the
# fine fixed design: level 0 and level 1 agree to the digits printed either way.
LINSOLVE = os.environ.get("LINSOLVE", "direct")  # "direct" (PARDISO) | "iterative" (BiCGStab)
# First refinement level to SOLVE. Each level is an independent computation -- subdivide, refill,
# solve -- so the only thing carried from one to the next is accumulated JAX state: compiled
# programs and device buffers for every level already done. That accumulation is what puts the top
# of a long series over the edge, and it is avoidable by running the expensive level in a fresh
# process. LEVEL_FROM=2 REFINE=2 solves level 2 ALONE, which is the same number as it would get in
# a 0..2 sweep -- levels do not feed each other -- but without arriving at it holding levels 0 and
# 1. The cheap steps before the series (C_design, C_kept, C_match) always run; they are small.
LEVEL_FROM = int(os.environ.get("LEVEL_FROM", "0"))
LEVEL = os.environ.get("LEVEL")  # fixed iso-level instead of the volume-preserving one

r = np.load(os.path.join(OUT, RUN))
cells, pts, rho = r["cells"], r["pts_f"].astype(float), r["rho_f"].astype(float)
LF = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def vols(c, p):
    v = p[c]
    return np.abs(np.linalg.det(np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], -1))) / 6.0


V = vols(cells, pts)
M_TARGET = float((rho * V).sum())
print(f"design {RUN}: {len(cells)} tets, blank {V.sum():.4f}, material {M_TARGET:.4f} "
      f"(fraction {M_TARGET / V.sum():.4f}), M_nd {float((4 * rho * (1 - rho)).mean()):.4f}")


# --- 1-2. threshold, largest piece, manifold repair -------------------------------------------
def _face_adj(idx):
    f = collections.defaultdict(list)
    for j, t in enumerate(idx):
        for tri in LF:
            f[tuple(sorted(int(cells[t][i]) for i in tri))].append(j)
    adj = collections.defaultdict(list)
    for ts in f.values():
        if len(ts) == 2:
            adj[ts[0]].append(ts[1])
            adj[ts[1]].append(ts[0])
    return adj


def _groups(local, adj):
    s = set(local)
    out = []
    while s:
        st = [s.pop()]
        g = [st[0]]
        while st:
            u = st.pop()
            for w in adj[u]:
                if w in s:
                    s.discard(w)
                    st.append(w)
                    g.append(w)
        out.append(g)
    return out


def _manifold(mask, log=False):
    """Drop tets until every vertex star and every edge fan of the kept set is face-connected.

    A tet subcomplex has a manifold boundary exactly when that holds; two lobes meeting only along
    an edge or at a point are what gmsh cannot fill, and thresholding a tet mesh produces them.
    """
    mask = mask.copy()
    for sweep in range(40):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return mask
        adj, w, drop = _face_adj(idx), V[idx], set()
        vt, ed = collections.defaultdict(list), collections.defaultdict(list)
        for j, t in enumerate(idx):
            c = cells[t]
            for n in c:
                vt[int(n)].append(j)
            for a in range(4):
                for b in range(a + 1, 4):
                    ed[tuple(sorted((int(c[a]), int(c[b]))))].append(j)
        for star in list(vt.values()) + list(ed.values()):
            g = _groups(star, adj)
            if len(g) > 1:
                g.sort(key=lambda gg: -w[gg].sum())
                for gg in g[1:]:
                    drop.update(gg)
        if not drop:
            return mask
        if log:
            print(f"    repair sweep {sweep}: -{len(drop)} tets, -{w[list(drop)].sum():.5f} vol")
        mask[idx[list(drop)]] = False
    raise RuntimeError("manifold repair did not converge in 40 sweeps")


def extract(tau, log=False):
    mask = rho > tau
    if not mask.any():
        return mask
    idx = np.where(mask)[0]
    comps = _groups(range(len(idx)), _face_adj(idx))
    comps.sort(key=lambda g: -V[idx[g]].sum())
    if log and len(comps) > 1:
        print(f"    {len(comps)} components; keeping the largest "
              f"({V[idx[comps[0]]].sum():.4f} of {V[idx].sum():.4f})")
    mask[:] = False
    mask[idx[comps[0]]] = True
    return _manifold(mask, log)


M_ND = float((4 * rho * (1 - rho)).mean())
BINARY_MND = float(os.environ.get("BINARY_MND", "0.01"))
IS_BINARY = M_ND < BINARY_MND

if LEVEL is not None:
    TAU, WHY = float(LEVEL), "requested"
elif IS_BINARY:
    # A BINARY design makes the volume-preserving bisection degenerate, and degenerate in a
    # direction that FLATTERS the design. Every level in (0,1) selects the same tets, so the only
    # way the search can raise the retained volume to meet M_TARGET is to drive the level toward
    # zero -- which admits VOID tets. Those are then re-solved as fully solid material, and because
    # the void wisps a SIMP design leaves behind sit exactly where its members almost touch, they
    # act as load-bearing bridges that the design does not actually contain.
    #
    # Measured on topo3d_h10b_Lagrange (M_nd 0.0004), 24 Aug 2026: the bisection settled at
    # rho > 0.000016 and produced a body that was 9.6 % void by volume, whose compliance was
    # 5.8716. Extracting the same design at rho > 0.5 gives 18.7890 -- 3.2x stiffer on paper for
    # want of bridges that are not there. The reported understatement moved from >=94.4 % to
    # >=648.6 %. The h=0.25 pair moved >=567 % -> >=1914 % (Lagrange) and >=14.4 % -> >=20.0 %
    # (cover); the correction is ~2.6x heavier on the Lagrange side, so it is NOT a common factor
    # that cancels in the comparison.
    #
    # So: on a binary design the honest level is 0.5, and the material the repair then drops is a
    # PROPERTY OF THE DESIGN -- fragmentation and non-manifold thinness -- to be reported, not a
    # deficit to be made up by lowering the threshold until the arithmetic balances.
    TAU, WHY = 0.5, f"binary design, M_nd {M_ND:.4f} < {BINARY_MND}"
else:
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if V[extract(mid)].sum() > M_TARGET:
            lo = mid
        else:
            hi = mid
    TAU, WHY = lo, "volume-preserving"
mask = extract(TAU, log=True)
_kept = V[mask].sum()
print(f"  level rho > {TAU:.4f} [{WHY}]: {int(mask.sum())} tets, "
      f"vol {_kept:.4f} ({100 * (_kept / M_TARGET - 1):+.2f}% vs the design's material)")
if IS_BINARY and LEVEL is None:
    # The shortfall IS the result here, so state it rather than leaving it as a percentage to be
    # read past. It is material the design holds that no connected, manifold body can carry.
    _solid = V[rho > 0.5].sum()
    print(f"  shortfall: the design holds {_solid:.4f} of solid, of which "
          f"{_solid - _kept:.4f} ({100 * (_solid - _kept) / _solid:.1f} %) is disconnected debris or "
          f"non-manifold junction that the extraction cannot keep")
# The level only means something if the answer does not swing with it. On a binary design it does not.
probe = [(t, extract(t)) for t in (0.3, 0.5, 0.7)]
print("  sensitivity to the level: " + ", ".join(
    f"rho>{t}: {int(m.sum())} tets / {V[m].sum():.4f}" for t, m in probe))

# Save what the extraction decided, so the reanalysed BODY can be inspected without re-deriving it
# (and without re-running the solves). EXTRACT_ONLY=1 stops here, which makes that free.
np.savez(
    os.path.join(OUT, f"extract_{TAG}.npz"),
    mask=mask, tau=TAU, cells=cells, pts=pts, rho=rho,
    material_target=M_TARGET, material_kept=V[mask].sum(),
)
if os.environ.get("EXTRACT_ONLY") == "1":
    print(f"EXTRACT_ONLY: wrote extract_{TAG}.npz ({int(mask.sum())} tets), stopping before the solves")
    raise SystemExit(0)

kept = cells[mask]


def boundary_tris(kc):
    seen = {}
    for t in range(len(kc)):
        for f in LF:
            k = tuple(sorted(int(kc[t][i]) for i in f))
            seen[k] = None if k in seen else (t, f)
    tris = []
    for k, v in seen.items():
        if v is None:
            continue
        t, f = v
        c = kc[t]
        tri = [int(c[i]) for i in f]
        opp = next(int(c[i]) for i in range(4) if i not in f)
        a, b, cc = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        if np.dot(np.cross(b - a, cc - a), pts[opp] - a) > 0:  # point the normal AWAY from the solid
            tri = [tri[0], tri[2], tri[1]]
        tris.append(tri)
    return np.asarray(tris, dtype=int)


btri = boundary_tris(kept)
ec = collections.Counter()
for tri in btri:
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ec[tuple(sorted((tri[a], tri[b])))] += 1
assert set(ec.values()) == {2}, f"boundary is not a closed manifold: edge counts {sorted(set(ec.values()))}"
print(f"  extracted surface: {len(btri)} triangles, closed and manifold")
# How much of the solid mesh is interior at all. The equal-count control below can only see
# distortion that lives in interior nodes, so this bounds what that control is able to detect.
_nn, _ns = len(np.unique(kept)), len(np.unique(btri))
print(f"  solid mesh: {_nn} nodes, {_ns} on the surface, {_nn - _ns} interior "
      f"({100 * (_nn - _ns) / _nn:.1f}%) -- all the room a distorted interior has")


# --- 3. midpoint subdivision: exact on planar facets -------------------------------------------
def subdivide(P, T, times):
    P = list(map(tuple, P))
    for _ in range(times):
        mid, new = {}, []
        for a, b, c in T:
            m = []
            for u, v in ((a, b), (b, c), (c, a)):
                k = (min(u, v), max(u, v))
                if k not in mid:
                    mid[k] = len(P)
                    P.append(tuple((np.asarray(P[u]) + np.asarray(P[v])) / 2.0))
                m.append(mid[k])
            new += [[a, m[0], m[2]], [m[0], b, m[1]], [m[2], m[1], c], [m[0], m[1], m[2]]]
        T = np.asarray(new, dtype=int)
    return np.asarray(P, dtype=float), T


def surf_area(P, T):
    v = P[T]
    return float(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2.0)


used = np.unique(btri)
remap = np.full(pts.shape[0], -1, dtype=int)
remap[used] = np.arange(len(used))
P0, T0 = pts[used], remap[btri]
A0 = surf_area(P0, T0)
_e = P0[T0]
H0 = float(np.median(np.linalg.norm(np.stack([_e[:, 1] - _e[:, 0], _e[:, 2] - _e[:, 1], _e[:, 0] - _e[:, 2]]), axis=-1)))
print(f"  surface area {A0:.6f}, median facet edge {H0:.4f} -- the base size of the refinement series")


# --- 4. fill the closed surface with gmsh -------------------------------------------------------
def refill(P, T, h):
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("solid")
        s = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(2, s, list(range(1, len(P) + 1)), P.ravel().tolist())
        gmsh.model.mesh.addElementsByType(s, 2, [], (T + 1).ravel().tolist())
        gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([s])])
        gmsh.model.geo.synchronize()
        # The interior size must NOT be inherited from the boundary triangles, or asking for a
        # finer fill silently returns the same mesh back.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(3)
        _t, _tags, nodes = gmsh.model.mesh.getElements(3)
        tet = np.asarray(nodes[0], dtype=np.int64).reshape(-1, 4)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(np.asarray(ntags))
        idx = np.asarray(ntags)[order]
        xyz = np.asarray(coords).reshape(-1, 3)[order]
        lut = np.full(int(idx.max()) + 1, -1, dtype=np.int64)
        lut[idx] = np.arange(len(idx))
        return xyz, lut[tet]
    finally:
        gmsh.finalize()


# --- the solve ----------------------------------------------------------------------------------
def compliance(P, C, rho_vals, tag):
    path = os.path.join(SCRATCH, f"_re_{TAG}_{tag}.msh")
    meshio.write(path, meshio.Mesh(points=P, cells=[("tetra", C)]), file_format="gmsh")
    d = jno.domain(path)
    cc = np.asarray(d._cells_p1())
    assert cc.shape == C.shape, f"{tag}: jNO read {cc.shape} cells, wrote {C.shape}"
    p0 = np.asarray(d.mesh.points)[:, :3]
    # ELEMENTWISE, not sorted: a sorted comparison passes under a permutation, which would silently
    # scramble rho against the cells it belongs to. (Measured: the round trip preserves order exactly.)
    assert np.allclose(vols(cc, p0), vols(C, P)), f"{tag}: cells were reordered by the round trip"
    assert np.allclose(p0[cc].mean(axis=1), P[C].mean(axis=1)), f"{tag}: the mesh moved on the round trip"

    u, phi = d.fem_symbols(value_shape=(3,))
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    rr = jno.np.parameter(s, name=f"rho_{tag}")
    rr.dtype(jnp.float64)
    vals = jnp.asarray(np.asarray(rho_vals, dtype=float).reshape(-1))
    rr.initialize(lambda k, sh, dtype=None, _v=vals: _v.reshape(sh))

    xi, yi, zi, _ = d.variable("interior", split=True)
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
    E = lambda x: EMIN + x**PENAL * (E0 - EMIN)  # noqa: E731

    # Normalise by the ASSEMBLED patch area. Subdivision and refilling leave it invariant, so this
    # doubles as a check that the load really is the same load on every mesh.
    seen: dict = {}
    for t in range(cc.shape[0]):
        c = cc[t]
        for f in LF:
            k = tuple(sorted(int(c[i]) for i in f))
            seen.pop(k, None) if k in seen else seen.setdefault(k, t)
    tri = np.array(list(seen.keys()), dtype=int)
    keep = ((p0[tri][..., 1] > LY - 1e-6) & (p0[tri][..., 0] > TOPL - 1e-6) & (p0[tri][..., 0] < TOPR + 1e-6)).all(axis=1)
    v = p0[tri[keep]]
    area = float(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2.0)
    assert area > 1e-12, f"{tag}: the rope patch has no facets on this mesh"

    fem = jno.fem(
        [
            E(rr) * a(eps(u), eps(phi)),
            u(*bolts) - (0.0, 0.0, 0.0),
            -1.0 * inner(jnp.asarray(PULL / area), phi.bind(**dict(zip("xyz", eye))), 1),
        ],
        quad_degree=2,
        **({"chunk": FEMCHUNK} if FEMCHUNK > 0 else {}),
    )
    _integrand = E(rr) * a(eps(u), eps(u))
    Cc = (_integrand.integrate(fem) if LINSOLVE == "iterative"
          else _integrand.integrate(fem, solver=pardiso_lu))
    crux = jno.core([Cc], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    return float(np.asarray(crux.eval([Cc])).reshape(-1)[0]), area


C_OPT = float(np.asarray(r["C"]).reshape(-1)[-1]) if "C" in r.files else float("nan")

print()
C_design, a_d = compliance(pts, cells, rho, "design")
print(f"  {'design, SIMP field, design mesh':40s} C = {C_design:10.4f}  ({len(cells):6d} tets, patch {a_d:.5f})")
C_kept, a_k = compliance(pts, kept, np.ones(len(kept)), "kept")
print(f"  {'extracted solid, the SAME tets':40s} C = {C_kept:10.4f}  ({len(kept):6d} tets, patch {a_k:.5f})")

# The equal-count fill. It must use the RAW surface, not the subdivided one: 5,648 boundary
# triangles put a floor on the tet count far above what the design spent, so a matched fill is
# only reachable on the original 1,412. That is the stricter control anyway -- identical geometry
# AND identical surface triangulation, leaving element quality as the only difference.
n_want, lo_h, hi_h, best = len(kept), 0.01, 2.0, None
for _ in range(20):
    mh = 0.5 * (lo_h + hi_h)
    P, Cx = refill(pts[used], remap[btri], mh)
    if best is None or abs(len(Cx) - n_want) < abs(len(best[1]) - n_want):
        best = (P, Cx, mh)
    if len(Cx) > n_want:
        lo_h = mh
    else:
        hi_h = mh
    if abs(len(Cx) - n_want) <= max(8, n_want // 100):
        break
P_m, C_m, h_m = best
C_match, a_m = compliance(P_m, C_m, np.ones(len(C_m)), "match")
print(f"  {'same geometry, EQUAL COUNT, clean tets':40s} C = {C_match:10.4f}  ({len(C_m):6d} tets, patch {a_m:.5f}, h={h_m:.4f})")

# UNIFORM refinement. The surface must halve with the interior: leaving it fixed pins the element
# size near the boundary, and the series then measures the interior alone -- which is what made an
# earlier version of this look divergent. Midpoint subdivision is exact on planar facets, so every
# level below solves the IDENTICAL body.
refills = []
for k in range(LEVEL_FROM, REFINE + 1):
    Pk, Tk = subdivide(P0, T0, k)
    ak = surf_area(Pk, Tk)
    assert abs(ak - A0) < 1e-9 * max(A0, 1.0), f"subdivision moved the surface: {A0} -> {ak}"
    hk = H0 / 2**k
    P, Cx = refill(Pk, Tk, hk)
    Cr, a_r = compliance(P, Cx, np.ones(len(Cx)), f"lvl{k}")
    refills.append((k, hk, len(Cx), Cr, a_r))
    print(f"  {'uniform refinement level ' + str(k) + f' (h={hk:.4f})':40s} C = {Cr:10.4f}  "
          f"({len(Cx):6d} tets, {len(Tk):6d} facets, patch {a_r:.5f})")

print()
print(f"formulation   C_opt {C_OPT:.4f} -> P1 read {C_design:.4f}       : {100 * (C_design / C_OPT - 1):+7.2f} %")
print(f"material model  SIMP field -> extracted solid, same mesh: {100 * (C_kept / C_design - 1):+7.2f} %")
print()
_frac = 100 * (_nn - _ns) / _nn
print(f"DISTORTION      same geometry, {len(C_m)} vs {len(kept)} tets   : {100 * (C_match / C_kept - 1):+7.2f} %")
print("                (equal resolution, so this is element QUALITY alone)")
print(f"                the extracted solid is {_frac:.1f} % interior nodes, and a refill can only")
if _frac < 10.0:
    print("                move those -- with the surface pinned this tightly there is barely an")
    print("                alternative tetrahedralisation to find, so read this as 'distortion has no")
    print("                room to operate at this resolution', NOT as 'distortion is harmless'.")
else:
    print("                move those, which bounds what this control is able to detect.")
print()
print("uniform refinement of the IDENTICAL body:")
for k, hk, n, Cr, _ in refills:
    print(f"  level {k} (h={hk:.4f}, {n:7d} tets): C = {Cr:8.4f}   {100 * (Cr / C_kept - 1):+7.2f} % vs the design's own tets")

# P1 tets are stiff and refinement can only soften them, so a monotone series is a one-sided bound
# on the truth -- which is all that is needed, and is honest where an extrapolation would not be.
Cs = [x[3] for x in refills]
mono = all(Cs[i + 1] > Cs[i] for i in range(len(Cs) - 1))
LB = max(Cs)
if len(Cs) == 1:
    # LEVEL_FROM skipped the rest of the series. One level is still a one-sided bound -- coarse P1
    # tets are too stiff, so any level understates -- but "monotone" would be a claim about a
    # comparison that was never made, and `all()` over an empty range says True.
    print(f"  ONE level only (LEVEL_FROM={LEVEL_FROM}), so there is no series to check for "
          f"monotonicity. A single level is still a lower bound: AT LEAST {LB:.4f}")
    print(f"    the optimiser reported {C_OPT:.4f}  -> it understates the truth by at least "
          f"{100 * (LB / C_OPT - 1):+.1f} %")
elif mono:
    print(f"  monotone increasing, so the true compliance of this body is AT LEAST {LB:.4f}")
    print(f"    the optimiser reported {C_OPT:.4f}  -> it understates the truth by at least "
          f"{100 * (LB / C_OPT - 1):+.1f} %")
    print(f"    a plain P1 read gives {C_design:.4f} -> understates by at least "
          f"{100 * (LB / C_design - 1):+.1f} %")
    if len(Cs) > 2:
        print(f"    still moving {100 * (Cs[-1] / Cs[-2] - 1):+.2f} % at the last level, so these "
              f"bounds are loose, not converged values")
else:
    print("  NOT monotone -- the series is unreliable, do not read a bound off it")
