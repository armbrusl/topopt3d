"""Mesh refinement preview: the actual tetrahedra, cut away so interior element size is visible.

Same projection / painter sort / screen-space Lambert as the console and the result figures. Every
edge drawn is a real mesh edge -- no wireframe stylisation, no subdivision.
"""

import os

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib as mpl  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

import jno  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
L, H, W = 4.0, 2.0, 2.0
ACCENT = np.array([13, 148, 136]) / 255.0

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "Frutiger 45 Light", "FreeSans", "DejaVu Sans"],
    "font.weight": "light", "axes.titleweight": "light", "font.size": 9.5, "axes.titlesize": 9.5,
})
import matplotlib.pyplot as plt  # noqa: E402

LF = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def cut_surface(cells, pts, keep):
    """Boundary faces of the KEPT sub-block: a face used by exactly one kept tet.

    Drawing every face of every kept tet would paint interior faces twice and hide the cut plane
    behind them; taking the boundary of the sub-block shows exactly the cut surface plus the outer
    skin, which is what makes element size legible.
    """
    seen = {}
    for t in np.where(keep)[0]:
        c = cells[t]
        for f in LF:
            k = tuple(sorted((int(c[f[0]]), int(c[f[1]]), int(c[f[2]]))))
            if k in seen:
                del seen[k]
            else:
                seen[k] = t
    return np.array(list(seen.keys()), dtype=int)


def draw(ax, pts, faces, *, az=-0.72, el=0.42, lw=0.25):
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    w = pts - np.array([L / 2, H / 2, W / 2])
    X = w[:, 0] * ca + w[:, 2] * sa
    Z = -w[:, 0] * sa + w[:, 2] * ca
    P = np.stack([X, w[:, 1] * ce - Z * se, w[:, 1] * se + Z * ce], axis=1)
    Q = P[faces]
    Q = Q[np.argsort(Q[:, :, 2].mean(axis=1))]
    A, B, C = Q[:, 0], Q[:, 1], Q[:, 2]
    n = np.cross(B - A, C - A)
    nl = np.linalg.norm(n, axis=1)
    nl[nl == 0] = 1.0
    lam = np.clip(np.abs(n @ np.array([0.34, 0.60, 0.72]) / nl) * 0.8 + 0.34, 0.3, 1.0)[:, None]
    face = np.repeat(ACCENT[None, :], len(Q), axis=0) * lam
    # A real mesh edge, drawn dark, is the whole point of this figure -- it is what shows the size.
    ax.add_collection(PolyCollection(Q[:, :, :2], facecolors=face, edgecolors="#12343B", linewidths=lw))
    ax.set_xlim(Q[:, :, 0].min() - 0.05, Q[:, :, 0].max() + 0.05)
    ax.set_ylim(Q[:, :, 1].min() - 0.05, Q[:, :, 1].max() + 0.05)
    ax.set_aspect("equal")
    ax.set_axis_off()


SIZES = [(0.30, "shown earlier"), (0.14, ""), (0.10, "the 30-min target")]
fig, axes = plt.subplots(1, len(SIZES), figsize=(13.5, 3.4))
for ax, (h, note) in zip(axes, SIZES):
    d = jno.Shape.box(0, 0, 0, L, H, W, size=h).domain()
    cells = np.asarray(d._cells_p1())
    pts = np.asarray(d.mesh.points)[:, :3]
    cen = pts[cells].mean(axis=1)
    keep = ~((cen[:, 0] > L / 2) & (cen[:, 2] > W / 2))          # quarter cut-away
    faces = cut_surface(cells, pts, keep)
    # mean edge length, measured on the mesh rather than assumed from `size`
    e = np.concatenate([np.linalg.norm(pts[cells[:, i]] - pts[cells[:, j]], axis=1)
                        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    draw(ax, pts, faces, lw=0.25 if len(cells) < 30000 else 0.12)
    ax.set_title(f"{len(cells):,} tets   ·   {3 * len(pts):,} dofs   ·   $\\bar h$ = {e.mean():.3f}"
                 + (f"\n{note}" if note else "\n"))
    print(f"h={h}: {len(cells)} tets, {len(pts)} nodes, {3 * len(pts)} dofs, "
          f"mean edge {e.mean():.4f}, {L / e.mean():.1f} across the span", flush=True)

fig.savefig(os.path.join(HERE, "mesh_refinement.png"))
print("wrote mesh_refinement.png")
