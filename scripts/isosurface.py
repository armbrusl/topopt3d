"""Marching-tetrahedra iso-surface of the density field, vs the per-element extraction.

The per-element view (`render.py`) draws whole tet FACES of the elements above the level. Its
smallest drawable feature is therefore one tetrahedron, and it looks blocky at any resolution --
that is a property of the extraction, not of the design.

This one averages the P0 density to nodes (volume-weighted, the standard projection) and cuts each
tet on the rho = level contour. Nothing is smoothed or invented: the vertices sit at the exact
linear interpolation of the same densities, and the surface passes through the level set of the
node-projected field. It is what topology-optimisation papers show, and the honest comparison is
that the two are the SAME design read at two resolutions of the same data.

    python scripts/isosurface.py            # writes runs/topo3d_isosurface.png
"""

import os

import numpy as np
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")
run = np.load(os.path.join(OUT, os.environ.get("RUN", "topo3d_final.npz")))
cells, pts, rho = run["cells"], run["pts_f"], run["rho_f"]
L, H, W = 4.0, 2.0, 2.0
LEVEL = float(os.environ.get("LEVEL", "0.5"))
ACCENT = np.array([13, 148, 136]) / 255.0


def nodal_density(cells, pts, rho):
    """Volume-weighted P0 -> P1 projection. Each node takes the mean of its incident elements,
    weighted by element volume so a sliver does not count as much as a healthy tet."""
    v = pts[cells]
    vol = np.abs(np.linalg.det(np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], -1))) / 6.0
    num = np.zeros(len(pts))
    den = np.zeros(len(pts))
    for k in range(4):
        np.add.at(num, cells[:, k], rho * vol)
        np.add.at(den, cells[:, k], vol)
    return num / np.maximum(den, 1e-300)


# The six sign patterns of a tet reduce to two cases: one vertex on one side (a triangle) or two
# on each (a quad, split into two triangles). Edge list is fixed so the interpolation is exact.
TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def marching_tets(cells, pts, phi, level):
    """Triangles of the phi = level surface, cut through each tetrahedron."""
    tris = []
    inside = phi[cells] > level  # (n_cells, 4)
    n_in = inside.sum(axis=1)
    active = np.where((n_in > 0) & (n_in < 4))[0]
    for t in active:
        c = cells[t]
        f = phi[c]
        cut = {}
        for a, b in TET_EDGES:
            if (f[a] > level) != (f[b] > level):
                w = (level - f[a]) / (f[b] - f[a])  # exact linear crossing on this edge
                cut[(a, b)] = pts[c[a]] + w * (pts[c[b]] - pts[c[a]])
        p = list(cut.values())
        if len(p) == 3:
            tris.append(p)
        elif len(p) == 4:
            # order the quad by angle about its centroid so the two triangles do not cross
            q = np.array(p)
            cen = q.mean(axis=0)
            e1 = q[0] - cen
            e1 /= np.linalg.norm(e1) or 1.0
            nrm = np.cross(q[1] - cen, q[2] - cen)
            nl = np.linalg.norm(nrm)
            nrm = nrm / nl if nl else np.array([0.0, 0.0, 1.0])
            e2 = np.cross(nrm, e1)
            ang = np.arctan2([(x - cen) @ e2 for x in q], [(x - cen) @ e1 for x in q])
            o = q[np.argsort(ang)]
            tris.append([o[0], o[1], o[2]])
            tris.append([o[0], o[2], o[3]])
    return np.array(tris) if tris else np.zeros((0, 3, 3))


def weld(tri, tol=9):
    """Weld the triangle soup into a shared-vertex mesh. Returns (verts, faces).

    Marching tets emits each triangle independently, so a vertex shared by several triangles exists
    several times over and there is no connectivity to average a normal across. Rounding to `tol`
    decimals is safe here because the crossing points are computed from the same edge endpoints by
    the same interpolation, so shared vertices agree to far better than that.
    """
    flat = tri.reshape(-1, 3)
    key = np.round(flat, tol)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return flat[first], np.asarray(inv).reshape(-1, 3)


def vertex_normals(verts, faces):
    """Area-weighted vertex normals. The cross product is not normalised before accumulation, so a
    big triangle counts for more than a sliver -- which is what makes the result stable on the very
    uneven triangles marching tets produces."""
    v = verts[faces]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    out = np.zeros_like(verts)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-300)


def taubin(verts, faces, iters=8, lam=0.55, mu=-0.58):
    """Taubin lambda|mu smoothing of the SURFACE, not the data.

    A shrink pass followed by an inflate pass, which is why it does not collapse the shape the way
    repeated Laplacian smoothing does. It moves the extracted vertices only -- the density field,
    the compliance and every number reported elsewhere are untouched -- so it is a presentation
    step, and one worth naming in a caption.
    """
    nb = [[] for _ in range(len(verts))]
    for f in faces:
        for a in range(3):
            nb[f[a]].append(f[(a + 1) % 3])
            nb[f[a]].append(f[(a + 2) % 3])
    idx = [np.unique(x) for x in nb]
    V = verts.copy()
    for _ in range(iters):
        for step in (lam, mu):
            avg = np.array([V[i].mean(axis=0) if len(i) else np.zeros(3) for i in idx])
            V = V + step * (avg - V)
    return V


def draw(ax, tri, *, az=-0.72, el=0.42, lw=0.0, smooth=False, vn=None):
    """Same projection / painter sort / screen-space Lambert as the console.

    ``smooth`` shades each face from the mean of its three VERTEX normals instead of its own face
    normal. The geometry is identical -- this only changes the shading -- but it removes the
    faceting that makes an unstructured triangulation read as grainy at any resolution.
    """
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    w = tri - np.array([L / 2, H / 2, W / 2])
    X = w[..., 0] * ca + w[..., 2] * sa
    Z = -w[..., 0] * sa + w[..., 2] * ca
    Q = np.stack([X, w[..., 1] * ce - Z * se, w[..., 1] * se + Z * ce], axis=-1)
    order = np.argsort(Q[:, :, 2].mean(axis=1))
    Q = Q[order]
    A, B, C = Q[:, 0], Q[:, 1], Q[:, 2]
    n = np.cross(B - A, C - A)
    nl = np.linalg.norm(n, axis=1)
    nl[nl == 0] = 1.0
    if smooth and vn is not None:
        # rotate the vertex normals into the same camera frame the faces were projected into
        ca2, sa2, ce2, se2 = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
        NX = vn[..., 0] * ca2 + vn[..., 2] * sa2
        NZ = -vn[..., 0] * sa2 + vn[..., 2] * ca2
        nsm = np.stack([NX, vn[..., 1] * ce2 - NZ * se2, vn[..., 1] * se2 + NZ * ce2], axis=-1)
        n = nsm.mean(axis=1)[order]
        nl = np.linalg.norm(n, axis=1)
        nl[nl == 0] = 1.0
    lam = np.clip(np.abs(n @ np.array([0.34, 0.60, 0.72]) / nl) * 0.8 + 0.34, 0.3, 1.0)[:, None]
    rgb = np.repeat(ACCENT[None, :], len(Q), axis=0) * lam
    ax.add_collection(PolyCollection(Q[:, :, :2], facecolors=rgb, edgecolors=rgb, linewidths=lw))
    ax.set_xlim(Q[:, :, 0].min() - 0.05, Q[:, :, 0].max() + 0.05)
    ax.set_ylim(Q[:, :, 1].min() - 0.05, Q[:, :, 1].max() + 0.05)
    ax.set_aspect("equal")
    ax.set_axis_off()


LF = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def element_faces(cells, pts, rho, level):
    """The per-element extraction, for the comparison panel."""
    seen = {}
    for t in np.where(rho > level)[0]:
        c = cells[t]
        for f in LF:
            k = tuple(sorted((int(c[f[0]]), int(c[f[1]]), int(c[f[2]]))))
            seen.pop(k, None) if k in seen else seen.setdefault(k, t)
    return pts[np.array(list(seen.keys()), dtype=int)]


def boundary_caps(cells, pts, phi, level):
    """Close the iso-surface where the solid meets the domain boundary.

    Without these the level set is an open shell -- it only cuts INTERIOR material interfaces, so a
    structure that runs into the box face is drawn as a floating sheet. Each boundary triangle is
    clipped to its `phi > level` part by the 2-D analogue of the tet march, so the caps come from
    the same field and meet the shell exactly.
    """
    seen = {}
    for t in range(len(cells)):
        c = cells[t]
        for f in LF:
            k = tuple(sorted((int(c[f[0]]), int(c[f[1]]), int(c[f[2]]))))
            seen.pop(k, None) if k in seen else seen.setdefault(k, t)
    out = []
    for tri_nodes in seen:  # faces used once == the mesh boundary
        n = np.array(tri_nodes)
        f = phi[n]
        inside = f > level
        if inside.all():
            out.append(pts[n])
            continue
        if not inside.any():
            continue
        poly = []
        for a in range(3):
            b = (a + 1) % 3
            if inside[a]:
                poly.append(pts[n[a]])
            if inside[a] != inside[b]:
                w = (level - f[a]) / (f[b] - f[a])
                poly.append(pts[n[a]] + w * (pts[n[b]] - pts[n[a]]))
        for i in range(1, len(poly) - 1):  # fan-triangulate the clipped polygon
            out.append([poly[0], poly[i], poly[i + 1]])
    return np.array(out) if out else np.zeros((0, 3, 3))


if __name__ == "__main__":
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "FreeSans", "DejaVu Sans"],
        "font.weight": "light", "axes.titleweight": "light", "font.size": 9.5, "axes.titlesize": 9.5,
    })
    import matplotlib.pyplot as plt

    phi = nodal_density(cells, pts, rho)
    tri = marching_tets(cells, pts, phi, LEVEL)
    cap = boundary_caps(cells, pts, phi, LEVEL)
    tri = np.concatenate([tri, cap]) if len(cap) else tri
    ele = element_faces(cells, pts, rho, LEVEL)
    print(f"tets={len(cells)}  per-element faces={len(ele)}  "
          f"marching-tets triangles={len(tri)} (of which {len(cap)} boundary caps)")

    verts, faces = weld(tri)
    vn = vertex_normals(verts, faces)
    smoothV = taubin(verts, faces, iters=int(os.environ.get("TAUBIN", "8")))
    vn_s = vertex_normals(smoothV, faces)
    print(f"welded to {len(verts)} vertices / {len(faces)} faces")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    draw(axes[0], ele, lw=0.25)
    axes[0].set_title("per-element\nwhole tet faces above $\\bar\\rho=0.5$")
    draw(axes[1], verts[faces], smooth=True, vn=vn[faces])
    axes[1].set_title("marching tets + smooth normals\nsame geometry, shading only")
    draw(axes[2], smoothV[faces], smooth=True, vn=vn_s[faces])
    axes[2].set_title("+ Taubin surface smoothing\npresentation step, data untouched")
    fig.savefig(os.path.join(OUT, "topo3d_isosurface.png"))
    print("wrote topo3d_isosurface.png")
