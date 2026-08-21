"""The converged bracket from several viewpoints, as the ELEMENT boundary.

What is drawn is the material surface of the mesh itself -- every face of every tetrahedron above
rho = 0.5, with the nodes where the optimiser left them. That is deliberate: the deformable-mesh
method's claim is that no post-processing is needed, so the mesh IS the deliverable and an
iso-surface would be the very step the method exists to avoid. Nothing is smoothed, and the only
rendering choice is a fixed Lambertian light, without which a faceted surface is unreadable.

    python scripts/fig_angles.py sw2            # one design, six views
    python scripts/fig_angles.py np0 sw2        # two designs, three views each
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
        "font.size": 8.0,
        "text.color": "#1A202C",
        "savefig.facecolor": "white",
    }
)
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
TAGS = sys.argv[1:] or ["sw2"]
LF = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
# A headlight: the light follows the camera, offset up and to the left. A world-fixed light leaves
# whichever views face away from it unreadably dark, which would hide exactly what is being judged.
def headlight(cam):
    up = np.array([0.0, 0.0, 1.0]) if abs(cam[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    left = np.cross(up, cam)
    left /= np.linalg.norm(left)
    lit = cam + 0.45 * left + 0.35 * np.cross(cam, left)
    return lit / np.linalg.norm(lit)

# Named by where the CAMERA is, in world axes, so the labels mean the same thing whatever case is
# being drawn. NOTE gives a case-specific hint (which end carries the load) without renaming a view.
NOTE = os.environ.get("NOTE", "")
VIEWS = [
    (22, -60, "three-quarter"),
    (22, -120, "three-quarter, opposite side"),
    (2, -90, "face on, from +z"),
    (2, 0, "edge on, from +x"),
    (78, -90, "from above (+z)"),
    (-70, -90, "from below (-z)"),
]


def surface(tag):
    r = np.load(os.path.join(RUNS, f"topo3d_{tag}.npz"))
    cells, pts, rho = r["cells"], r["pts_f"], r["rho_f"]
    seen: dict = {}
    for t in np.flatnonzero(rho > 0.5):
        c = cells[t]
        for fa in LF:
            k = tuple(sorted(int(c[i]) for i in fa))
            seen.pop(k, None) if k in seen else seen.setdefault(k, t)
    tri = np.array(list(seen.keys()), dtype=int)
    v = pts[tri]
    nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    v, nrm = v[ok], nrm[ok] / ln[ok][:, None]

    edge: dict = {}
    for i, t3 in enumerate(tri[ok]):
        for a, b in ((0, 1), (1, 2), (0, 2)):
            k = (int(t3[a]), int(t3[b])) if t3[a] < t3[b] else (int(t3[b]), int(t3[a]))
            edge.setdefault(k, []).append(i)
    pr = np.array([e for e in edge.values() if len(e) == 2])
    ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(nrm[pr[:, 0]] * nrm[pr[:, 1]], axis=1)), 0, 1)))
    return v, nrm, float(np.median(ang)), float(np.mean(ang > 60) * 100)


data = {t: surface(t) for t in TAGS}
allv = np.concatenate([d[0].reshape(-1, 3) for d in data.values()])
mid, span = 0.5 * (allv.min(0) + allv.max(0)), (allv.max(0) - allv.min(0))

views = VIEWS if len(TAGS) == 1 else VIEWS[:3]
ncol = len(views) if len(TAGS) == 1 else 3
nrow = 2 if len(TAGS) == 1 else len(TAGS)
if len(TAGS) == 1:
    ncol = 3
fig = plt.figure(figsize=(3.0 * ncol, 2.9 * nrow))

k = 0
for tag in TAGS:
    verts, nrm, med, hi = data[tag]
    for elev, azim, name in views:
        k += 1
        ax = fig.add_subplot(nrow, ncol, k, projection="3d")
        # Painter's algorithm in the CAMERA's depth direction, or faces sort wrongly off-axis.
        cam = np.array(
            [
                np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                np.sin(np.radians(elev)),
            ]
        )
        shade = 0.28 + 0.72 * np.abs(nrm @ headlight(cam))
        rgb = np.clip(np.array([0.62, 0.66, 0.70]) * shade[:, None] + 0.10, 0, 1)
        order = np.argsort(verts.mean(axis=1) @ cam)
        ax.add_collection3d(Poly3DCollection(verts[order], facecolors=rgb[order], edgecolors="none"))
        for setter, c, s in ((ax.set_xlim, mid[0], span[0]), (ax.set_ylim, mid[1], span[1]), (ax.set_zlim, mid[2], span[2])):
            setter(c - 0.55 * s, c + 0.55 * s)
        ax.set_box_aspect(tuple(span), zoom=1.25)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        label = name if len(TAGS) == 1 else f"{tag} · {name}"
        ax.text2D(0.5, 0.04, label, transform=ax.transAxes, ha="center", fontsize=8, color="#1A202C")

hdr = "   ·   ".join(f"{t}: median {data[t][2]:.1f}°, {data[t][3]:.0f}% above 60°" for t in TAGS)
if NOTE:
    hdr += f"\n{NOTE}"
fig.text(0.5, 0.985, hdr, ha="center", va="top", fontsize=8.5, color="#1A202C")
fig.subplots_adjust(wspace=0.0, hspace=0.02, left=0.0, right=1.0, top=0.95, bottom=0.01)
dst = os.path.join(RUNS, f"angles_{'_'.join(TAGS)}.png")
fig.savefig(dst)
print(f"wrote {dst}")
for t in TAGS:
    print(f"  {t}: {len(data[t][0])} boundary facets, median {data[t][2]:.1f}°, {data[t][3]:.1f}% above 60°")
