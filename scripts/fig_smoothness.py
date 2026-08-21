"""Side-by-side render of the same bracket with and without the bending penalty.

Everything drawn is computed from the saved design -- the material surface is the set of faces used
by exactly one dense tetrahedron, its normals come from the deformed node positions, and the colour
is the roughness metric itself (the angle between a face's normal and those of its edge-neighbours,
folded into [0, 90] exactly as scripts/rough.py measures it). The top row is the same geometry under
a fixed Lambertian light, which is what makes a faceted surface legible at all; no geometry, field
or feature is added.

    python scripts/fig_smoothness.py sm0 sm0.5
"""

import os
import sys

import numpy as np

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "figure.dpi": 220,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "FreeSans", "DejaVu Sans"],
        "font.size": 8.5,
        "text.color": "#1A202C",
        "axes.labelcolor": "#1A202C",
        "savefig.facecolor": "white",
    }
)
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
TAGS = sys.argv[1:3] or ["sm0", "sm0.5"]
LABELS = ["no bending penalty", "with bending penalty"]
LF = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
LIGHT = np.array([0.35, -0.55, 0.76])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def surface(tag):
    """The material surface of a saved design, with a roughness value per face."""
    r = np.load(os.path.join(RUNS, f"topo3d_{tag}.npz"))
    cells, pts, rho = r["cells"], r["pts_f"], r["rho_f"]
    dense = np.flatnonzero(rho > 0.5)

    seen: dict = {}
    for t in dense:
        c = cells[t]
        for f in LF:
            k = tuple(sorted(int(c[i]) for i in f))
            seen.pop(k, None) if k in seen else seen.setdefault(k, t)
    tri = np.array(list(seen.keys()), dtype=int)
    v = pts[tri]
    nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    area = np.linalg.norm(nrm, axis=1) / 2.0
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30)

    edge: dict = {}
    for i, t3 in enumerate(tri):
        for a, b in ((0, 1), (1, 2), (0, 2)):
            key = (int(t3[a]), int(t3[b])) if t3[a] < t3[b] else (int(t3[b]), int(t3[a]))
            edge.setdefault(key, []).append(i)

    per_face: list = [[] for _ in range(len(tri))]
    angles = []
    for fs in edge.values():
        if len(fs) != 2:
            continue
        c = abs(float(nrm[fs[0]] @ nrm[fs[1]]))
        ang = np.degrees(np.arccos(min(c, 1.0)))
        angles.append(ang)
        per_face[fs[0]].append(ang)
        per_face[fs[1]].append(ang)
    rough = np.array([np.median(a) if a else 0.0 for a in per_face])
    return v, nrm, rough, np.array(angles), float(area.sum())


data = [surface(t) for t in TAGS]
vmax = 60.0

# One frame for both panels, from the union of the two designs, so the comparison is like for like.
allv = np.concatenate([d[0].reshape(-1, 3) for d in data])
lo, hi = allv.min(axis=0), allv.max(axis=0)
mid, span = 0.5 * (lo + hi), (hi - lo)

fig = plt.figure(figsize=(7.4, 5.2))

for col, (tag, label, (verts, nrm, rough, angles, area)) in enumerate(zip(TAGS, LABELS, data)):
    order = np.argsort(-verts[:, :, 1].mean(axis=1))  # painter's algorithm, far faces first
    shade = 0.30 + 0.70 * np.abs(nrm @ LIGHT)

    for row, mode in enumerate(("shape", "rough")):
        ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
        if mode == "shape":
            base = np.tile(np.array([0.62, 0.66, 0.70]), (len(verts), 1))
        else:
            base = plt.cm.magma_r(np.clip(rough / vmax, 0, 1))[:, :3]
        rgb = np.clip(base * shade[:, None] + 0.10, 0, 1)
        ax.add_collection3d(Poly3DCollection(verts[order], facecolors=rgb[order], edgecolors="none"))

        for setter, c, s in ((ax.set_xlim, mid[0], span[0]), (ax.set_ylim, mid[1], span[1]), (ax.set_zlim, mid[2], span[2])):
            setter(c - 0.52 * s, c + 0.52 * s)
        ax.set_box_aspect(tuple(span), zoom=1.30)
        ax.view_init(elev=24, azim=-58)
        ax.set_axis_off()
        if row == 0:
            ax.text2D(
                0.5,
                1.02,
                f"{label}\nmedian {np.median(angles):.1f}°  ·  {100 * np.mean(angles > 60):.0f}% above 60°  ·  area {area:.1f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
                linespacing=1.5,
                color="#1A202C",
            )

cax = fig.add_axes([0.34, 0.045, 0.32, 0.018])
cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax), cmap="magma_r"),
    cax=cax,
    orientation="horizontal",
)
cb.set_label("angle to adjacent surface facets  (degrees)", color="#1A202C")
cb.outline.set_visible(False)
cb.ax.tick_params(colors="#1A202C", length=2)
fig.subplots_adjust(wspace=0.0, hspace=0.02, left=0.0, right=1.0, top=0.90, bottom=0.09)
dst = os.path.join(RUNS, "smoothness_compare.png")
fig.savefig(dst)
print(f"wrote {dst}")
for tag, (_v, _n, _r, angles, area) in zip(TAGS, data):
    print(f"  {tag:8s} median {np.median(angles):5.1f} deg   p90 {np.percentile(angles, 90):5.1f}   "
          f">60 deg {100 * np.mean(angles > 60):4.1f} %   area {area:.2f}")
