"""The shared topology-optimisation machinery, on meshes small enough to check by hand.

Every module under test exists because a copy of it diverged and caused a bug, so these assert the
property the bug violated rather than the implementation that happens to be there now:

* the boundary walk finds the boundary (four scripts each grew their own);
* a node may only slide within its own surface (a bounding box let 175 boundary nodes move and the
  supports slid off the bolt holes -- 146 clamped nodes became 106, and the reported compliance
  stopped matching the saved mesh by 13.9 %);
* the material budget does not move when the mesh does (a budget written against the current mesh
  let the design domain inflate 42 % and take 42 % more material with it).

What is deliberately NOT asserted: any compliance value, and any claim about which design is good.
Those belong to the runs, not to the machinery.

    python -m pytest scripts/tests/test_topopt_layer.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topopt import coords, mesh  # noqa: E402


# The Kuhn/Freudenthal decomposition: six tets all sharing the main diagonal 0-7. Conforming when
# applied with the same vertex numbering to every hex, which the obvious-looking alternatives are
# not -- a non-conforming decomposition makes every internal face appear once, so the boundary walk
# calls the whole mesh boundary and every assertion below quietly tests nothing.
_KUHN = ((0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7))


def _unit_cube():
    """Six tets filling [0,1]^3. Every node is a corner, so every answer is knowable by hand."""
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=float,
    )
    return pts, np.asarray(_KUHN)


def _slab(n=3):
    """An n x n x 2 lattice of hexes, Kuhn-split. Has genuine interior nodes."""
    xs = np.linspace(0, 1, n + 1)
    zs = np.linspace(0, 1, 3)
    pts = np.array([[x, y, z] for z in zs for y in xs for x in xs], dtype=float)
    npx = n + 1

    def vid(i, j, k):
        return k * npx * npx + j * npx + i

    cells = []
    for k in range(2):
        for j in range(n):
            for i in range(n):
                c = [vid(i, j, k), vid(i + 1, j, k), vid(i, j + 1, k), vid(i + 1, j + 1, k),
                     vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i, j + 1, k + 1), vid(i + 1, j + 1, k + 1)]
                cells.extend([[c[m] for m in t] for t in _KUHN])
    return pts, np.asarray(cells)


class TestBoundaryWalk:
    """A facet bounds the mesh exactly when one cell owns it."""

    def test_the_cube_has_twelve_boundary_facets(self):
        pts, cells = _unit_cube()
        assert len(mesh.boundary_facets(cells)) == 12, "six square faces, two triangles each"

    def test_every_cube_node_is_on_the_boundary(self):
        pts, cells = _unit_cube()
        assert int(mesh.boundary_mask(cells, len(pts)).sum()) == 8, "a cube of corners has no interior"

    def test_a_slab_has_interior_nodes(self):
        pts, cells = _slab()
        m = mesh.boundary_mask(cells, len(pts))
        assert (~m).any(), "the slab must have at least one interior node or it tests nothing"

    def test_the_volume_is_the_volume(self):
        pts, cells = _unit_cube()
        assert mesh.cell_volumes(cells, pts).sum() == pytest.approx(1.0, abs=1e-12)

    def test_a_patch_area_counts_only_facets_wholly_inside_it(self):
        pts, cells = _unit_cube()
        bf = mesh.boundary_facets(cells)
        a = mesh.facet_area(pts, bf, keep=lambda v: v[..., 2] < 1e-9)  # the z = 0 face
        assert a == pytest.approx(1.0, abs=1e-12), f"one unit square, got {a}"


class TestSliding:
    """A boundary node may move along an axis only if every facet meeting it is parallel to it."""

    def test_a_corner_is_frozen_in_every_axis(self):
        pts, cells = _unit_cube()
        assert not coords.free_axes(pts, cells).any(), "a corner meets three orthogonal faces"

    def test_an_interior_node_is_free_in_every_axis(self):
        pts, cells = _slab()
        free = coords.free_axes(pts, cells)
        interior = ~mesh.boundary_mask(cells, len(pts))
        assert free[interior].all(), "an interior node has no surface to leave"

    def test_a_face_node_slides_in_plane_but_not_through(self):
        pts, cells = _slab()
        free = coords.free_axes(pts, cells)
        # a node strictly inside the z = 0 face
        on_z0 = (np.abs(pts[:, 2]) < 1e-12) & (pts[:, 0] > 1e-9) & (pts[:, 0] < 1 - 1e-9) \
            & (pts[:, 1] > 1e-9) & (pts[:, 1] < 1 - 1e-9)
        assert on_z0.any(), "the slab must have a face-interior node or this asserts nothing"
        assert not free[on_z0, 2].any(), "z is normal to that face and must be withheld"
        assert free[on_z0, 0].all() and free[on_z0, 1].all(), "x and y lie in the face and must be free"

    def test_the_classification_does_not_move_with_the_tolerance(self):
        """Planar-and-parallel is discrete, so five decades of tolerance must agree."""
        pts, cells = _slab()
        ref = coords.free_axes(pts, cells, tol=1e-9)
        for tol in (1e-7, 1e-5, 1e-4):
            assert (coords.free_axes(pts, cells, tol=tol) == ref).all(), (
                f"tol={tol} classified differently from tol=1e-9; the plateau claim is wrong"
            )


class TestFreezing:
    """A BC-carrying node is frozen, and so is anything close enough to drift into one."""

    def test_a_zone_freezes_only_boundary_nodes(self):
        pts, cells = _slab()
        everywhere = [lambda X, Y, Z: np.ones(X.shape, bool)]
        froz = coords.frozen_nodes(pts, cells, everywhere)
        assert froz.sum() == mesh.boundary_mask(cells, len(pts)).sum(), (
            "an interior node carries no boundary condition, so freezing it is over-constraint"
        )

    def test_the_margin_reaches_further_than_the_zone(self):
        pts, cells = _slab()
        zone = [lambda X, Y, Z: X < 1e-9]  # the x = 0 face
        bare = coords.frozen_nodes(pts, cells, zone).sum()
        grown = coords.frozen_nodes(pts, cells, zone, margin=0.5).sum()
        assert grown > bare, f"a 0.5 margin on a unit slab must catch more than {bare} nodes"


class TestDriftGuard:
    """The check whose absence let a whole set of results go unnoticed."""

    def test_an_unmoved_mesh_reports_no_drift(self):
        pts, cells = _slab()
        g = coords.DriftGuard(pts, cells, [lambda X, Y, Z: X < 1e-9])
        n0, n1, changed = g.check(pts)
        assert changed == 0 and n0 == n1 > 0, f"nothing moved, so nothing may have drifted: {g.check(pts)}"

    def test_a_node_pushed_into_the_zone_is_caught(self):
        pts, cells = _slab()
        zone = [lambda X, Y, Z: X < 0.2]
        g = coords.DriftGuard(pts, cells, zone)
        moved = pts.copy()
        # drag a boundary node that starts outside the band into it
        outside = np.flatnonzero(mesh.boundary_mask(cells, len(pts)) & (pts[:, 0] > 0.3))
        moved[outside[0], 0] = 0.05
        n0, n1, changed = g.check(moved)
        assert changed >= 1 and n1 > n0, (
            f"a node entering the band changes the support set: {n0} -> {n1}, changed {changed}"
        )
