"""Which nodes the optimiser may move, and the guard that says so if it was wrong.

This module exists because getting it wrong is silent. A Dirichlet tag resolves to node IDS once, at
setup, and clamps those ids for the whole run. Let a clamped node move and it keeps being clamped
while sliding off the geometry its predicate described: every solve stays well posed, the objective
keeps falling, and the reported number simply stops being the compliance of the mesh being saved.
Measured on the h=0.25 bracket -- 146 clamped nodes at setup, 106 still on the hole wall at
convergence, and a reported 4.6412 against 5.2876 from an independent solve of the same mesh and
density. A 13.9 % error, growing every iteration, invisible from inside the run.

Two rules follow, and both are enforced here rather than left to each problem script:

**Sliding, not moving.** A boundary node may slide WITHIN its own surface and never off it. Then the
boundary is the same point set at every iteration and the design domain is exactly invariant --
which is what makes the material budget mean one thing for the whole run.

**Freezing, with a margin.** A node carrying a boundary condition is frozen, and so is any node close
enough to drift into one. Both directions matter: a clamped node must not leave, and a free node
must not arrive.

The classification is derived from the mesh, not from the geometry. Axis ``e_i`` is free at boundary
node ``v`` iff every boundary facet meeting ``v`` has ``|n_i| < tol`` -- i.e. every one of them is
parallel to that axis, so moving along it stays in all of them. Measured against the bracket's
earlier hand-rolled version, which spelled out its two taper planes as equations: composed with the
same BC exclusion, free per axis 2468 / 2383 / 2320 of 3868 nodes and 2987 movable -- **identical,
zero nodes classified differently**, with no plane equations anywhere. And the classification does
not move from ``tol = 1e-9`` to ``1e-4``: "planar and parallel to this axis" is a discrete property
of the mesh, not a threshold to tune.

That equality is composed, and the uncomposed comparison is the more interesting one: on its own
this rule is STRICTER by 386 nodes. The hand-rolled version tested membership of the surfaces it
knew about -- six axis planes and two tapers -- so a node on a bolt-hole CYLINDER matched none of
them and came out free in all three axes. It is frozen in the bracket only because the clamp band
happens to cover the hole walls. Shrink that band and the hand-rolled rule walks nodes off the
cylinder; this one cannot, because it asks the facets rather than a list of known surfaces.

**Scope limit.** This is AXIS-ALIGNED sliding: exact for a planar face parallel to the freed axis,
and conservative -- it freezes -- for anything else. A discretised curved surface fails the test even
where the true surface would pass: the bracket's bolt-hole facets have ``max |n_z| = 0.105`` at
R = 0.35, h = 0.15, because three vertices at three angles give the x-y projection real area. So hole
walls freeze. Loosening ``tol`` to admit them would allow ``tol * move_bound`` of off-surface drift
per step -- ``tol`` is a direction cosine with a meaning, not a fudge factor -- and even an exact
tangent plane leaves ``d**2 / 2R = 9.2 %`` of the hole radius after one step at that size. General
tangent-plane sliding needs a reduced coordinate ``X = X0 + T q``, which is a different design
variable, not a looser tolerance.
"""

import numpy as np

from .mesh import boundary_facets, boundary_mask, facet_normals


def free_axes(pts, cells, *, tol=1e-7):
    """``(n_points, 3)`` bool: may this node move along this axis without leaving its own surface?

    Interior nodes are free in every axis -- they have no surface to leave. A boundary node is free
    along ``e_i`` only if every boundary facet meeting it is parallel to ``e_i``.
    """
    pts = np.asarray(pts)
    bf = boundary_facets(cells)
    free = np.ones((pts.shape[0], 3), dtype=bool)
    if bf.size == 0:
        return free
    n = np.abs(facet_normals(pts, bf))  # (n_facets, 3)
    # The largest normal component over the facets meeting each node. `maximum.at` rather than a
    # loop: at 4 M tets the loop is the whole runtime.
    worst = np.zeros((pts.shape[0], 3))
    for k in range(3):
        np.maximum.at(worst, bf[:, k], n)
    on_b = boundary_mask(cells, pts.shape[0])
    free[on_b] = worst[on_b] < tol
    return free


def frozen_nodes(pts, cells, zones, *, margin=0.0):
    """``(n_points,)`` bool: nodes a boundary condition is anchored to, grown by ``margin``.

    ``zones`` is a sequence of predicates ``f(X, Y, Z) -> bool`` over coordinate columns, written
    exactly as the domain tags are.

    Restricted to BOUNDARY nodes, and that restriction is not an optimisation. A Dirichlet or
    Neumann condition lives on boundary facets, so an interior node satisfying the predicate carries
    nothing and freezing it is pure over-constraint -- measured, freezing interior nodes near the
    bolts captured the elements that violate the element-volume bound, made ``g_vmx`` unsatisfiable
    by any nodal motion, and sent compliance to 1.65e3 against 4.07.

    ``margin`` should cover the furthest a free node can travel: a coordinate box of half-width ``b``
    per axis admits ``sqrt(2) * b`` of in-plane excursion, so that is the useful default for a zone
    described in the x-y plane.
    """
    pts = np.asarray(pts)
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    hit = np.zeros(pts.shape[0], dtype=bool)
    for zone in zones:
        hit |= np.asarray(zone(X, Y, Z), dtype=bool).reshape(-1)
    hit &= boundary_mask(cells, pts.shape[0])
    if margin > 0.0 and hit.any():
        # Distance to the resolved set, not a dilated predicate: it works for any predicate,
        # including a non-monotone one, and does not require the predicate to mean anything off
        # its own surface.
        anchors = pts[hit]
        for i in range(0, pts.shape[0], 4096):
            blk = pts[i : i + 4096]
            d2 = ((blk[:, None, :] - anchors[None, :, :]) ** 2).sum(-1).min(axis=1)
            hit[i : i + 4096] |= d2 < margin * margin
    return hit


def movable(pts, cells, *, zones=(), margin=0.0, tol=1e-7):
    """``(n_points, 3)`` bool: the per-axis design freedom, sliding and freezing combined.

    Returns the mask each axis's trainable coordinate set is drawn from. Per axis, because a
    coordinate design variable is a box-bounded scalar -- a node on a face normal to ``z`` simply
    never gets its ``z`` component promoted, which is what makes sliding expressible at all.
    """
    free = free_axes(pts, cells, tol=tol)
    free[frozen_nodes(pts, cells, zones, margin=margin)] = False
    return free


def register(d, pts0, cells, *, zones=(), margin=0.0, tol=1e-7, move_bound, mma, log=print):
    """Promote the movable coordinates and give each its own MMA box. Returns the per-axis masks.

    ``mma`` is a factory ``(lower, upper) -> optax chain``, so the calibration stays with the
    problem and only the geometry lives here.

    Each axis gets its OWN id set and its OWN box, built from that axis's own ids. Reusing axis 0's
    ids for all three -- which four scripts in this repo still do -- is correct only while one
    predicate feeds every axis, and silently anchors every bound to the wrong node's coordinate the
    moment it does not.
    """
    free = movable(pts0, cells, zones=zones, margin=margin, tol=tol)
    on_b = boundary_mask(cells, pts0.shape[0])
    names = ("mesh_x", "mesh_y", "mesh_z")
    for axis in range(3):
        ids = np.flatnonzero(free[:, axis])
        if ids.size == 0:
            continue
        sel = np.zeros(pts0.shape[0], dtype=bool)
        sel[ids] = True
        # A membership test on the node's own coordinates: `where=` is evaluated on coordinate
        # columns, so an id-derived set has to be expressed as one.
        v = d.variable(f"mv{axis}", where=_membership(pts0, sel), split=True)[axis]
        v.trainable(name=names[axis])
    for spec in getattr(d, "_trainable_coords", []) or []:
        sid = np.asarray(spec["ids"], dtype=int)
        c0 = pts0[sid, int(spec["axis"])]
        spec["coords0"] = c0
        spec["expr"].optimizer(mma(c0 - move_bound, c0 + move_bound))
    n_any = int(free.any(axis=1).sum())
    log(
        f"movable: {n_any} of {pts0.shape[0]} nodes "
        f"({int((free.any(axis=1) & ~on_b).sum())} interior, {int((free.any(axis=1) & on_b).sum())} sliding); "
        f"free dofs x/y/z = {int(free[:, 0].sum())}/{int(free[:, 1].sum())}/{int(free[:, 2].sum())} "
        f"of {3 * pts0.shape[0]}"
    )
    return free


def _membership(pts0, selected):
    """A coordinate predicate that reproduces an id set exactly, by matching the coordinates."""
    anchors = pts0[selected]

    def where(x, y, z):
        P = np.stack([np.asarray(v).reshape(-1) for v in (x, y, z)], axis=-1)
        out = np.zeros(P.shape[0], dtype=bool)
        if anchors.size == 0:
            return out
        for i in range(0, P.shape[0], 1024):
            blk = P[i : i + 1024]
            out[i : i + 1024] = ((blk[:, None, :] - anchors[None, :, :]) ** 2).sum(-1).min(axis=1) < 1e-18
        return out

    return where


class DriftGuard:
    """Checks that a boundary condition still describes the nodes it was resolved onto.

    Mandatory, not optional. This is the check whose absence invalidated a whole set of results, and
    the failure it catches produces no exception and no visibly wrong number -- only an independent
    re-solve of the saved mesh disagrees.

    Boundary nodes only: an interior node drifting into the zone joins no Dirichlet facet, and
    counting it reports drift that does not exist (measured: 6-8 phantom nodes per snapshot on a mesh
    whose supports were provably invariant).
    """

    def __init__(self, pts0, cells, zones):
        self._cells, self._zones = cells, tuple(zones)
        self._bnd = boundary_mask(cells, pts0.shape[0])
        self.initial = self._resolve(pts0)

    def _resolve(self, pts):
        pts = np.asarray(pts)
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        hit = np.zeros(pts.shape[0], dtype=bool)
        for zone in self._zones:
            hit |= np.asarray(zone(X, Y, Z), dtype=bool).reshape(-1)
        return hit & self._bnd

    def check(self, pts):
        """``(n_initial, n_now, n_changed)``; ``n_changed == 0`` is the only acceptable answer."""
        now = self._resolve(pts)
        return int(self.initial.sum()), int(now.sum()), int((now != self.initial).sum())
