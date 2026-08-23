"""Boundary walks on a tetrahedral mesh.

Every problem script needs the same three answers -- which facets bound the mesh, which nodes sit on
them, and what area a named patch of them covers -- and four scripts each grew their own copy of the
walk. They agree today. The reason to have one is that the *fourth* copy lives in
``reanalysis_boundary.py``, where it computes the area the applied traction is normalised by: a copy
that drifts there does not produce a wrong mesh, it produces a differently-loaded problem while
reporting the same numbers.

Numpy throughout, no jNO: these are properties of the connectivity, wanted before any node has
moved and often outside a solve entirely.
"""

import numpy as np

#: The four vertex triples of a tetrahedron's faces.
TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def boundary_facets(cells):
    """The ``(n_facets, 3)`` triangles that bound the mesh, as sorted vertex triples.

    A facet is on the boundary exactly when one tetrahedron owns it: the walk pops a face the second
    time it is seen, so what is left over is the boundary. Vectorised rather than the dict loop the
    scripts use, because at 4 M tets the loop is minutes and this is seconds.
    """
    c = np.asarray(cells)
    faces = np.sort(np.stack([c[:, list(f)] for f in TET_FACES], axis=1), axis=-1).reshape(-1, 3)
    uniq, counts = np.unique(faces, axis=0, return_counts=True)
    return uniq[counts == 1]


def boundary_mask(cells, n_points):
    """``(n_points,)`` bool: is this node on the boundary?

    "Interior" means *not on any boundary facet* -- including hole walls and sloped faces, which is
    the distinction a bounding-box test cannot make and which cost this project a whole set of
    results. See :mod:`topopt.coords`.
    """
    m = np.zeros(int(n_points), dtype=bool)
    bf = boundary_facets(cells)
    if bf.size:
        m[np.unique(bf)] = True
    return m


def facet_area(pts, facets, keep=None):
    """Total area of ``facets``, optionally restricted to those whose every node satisfies ``keep``.

    ``keep`` is a predicate on an ``(n, 3, 3)`` array of facet vertex coordinates returning an
    ``(n, 3)`` bool -- the same shape the scripts' inline versions use, so a load patch is written
    the same way here as in the weak form.

    The load must be normalised by the ASSEMBLED area rather than a nominal one: two meshes of the
    same patch do not carry the same facet area, and normalising by a nominal value pulls them with
    different total forces while the comparison claims to measure the design.
    """
    f = np.asarray(facets)
    if keep is not None and f.size:
        f = f[keep(np.asarray(pts)[f]).all(axis=1)]
    if f.size == 0:
        return 0.0
    v = np.asarray(pts)[f]
    return float(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2.0)


def cell_volumes(cells, pts):
    """``|det J| / 6`` per tetrahedron. Unsigned, so an inverted cell still reads positive."""
    v = np.asarray(pts)[np.asarray(cells)]
    return np.abs(np.linalg.det(np.stack([v[:, i + 1] - v[:, 0] for i in range(3)], -1))) / 6.0


def facet_normals(pts, facets):
    """Unit normals of ``facets``, ``(n_facets, 3)``. Orientation is not fixed; only ``|n|`` is used."""
    v = np.asarray(pts)[np.asarray(facets)]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
