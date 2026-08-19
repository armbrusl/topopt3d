"""What problem is actually being solved: the domain, the clamped face, the loaded patch.

Drawn from the SAME predicates `run.py` uses, evaluated on the real mesh -- the highlighted nodes
are the ones the solver actually constrains and loads, not a sketch of them. The load arrow is the
assembled resultant direction, scaled for visibility (its length carries no magnitude).
"""

import os

import numpy as np
from matplotlib.collections import PolyCollection

import jno

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")
os.makedirs(OUT, exist_ok=True)

L, H, W = 4.0, 2.0, 2.0
TOL, SPAN = 1e-9, 0.6
h = float(os.environ.get("HMESH", "0.28"))
ACCENT = np.array([13, 148, 136]) / 255.0
CLAMP = np.array([26, 32, 44]) / 255.0
LOAD = np.array([203, 63, 63]) / 255.0

d = jno.Shape.box(0, 0, 0, L, H, W, size=h).domain()
cells = np.asarray(d._cells_p1())
pts = np.asarray(d.mesh.points)[:, :3]

LF = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
seen = {}
for t in range(len(cells)):
    c = cells[t]
    for f in LF:
        k = tuple(sorted((int(c[f[0]]), int(c[f[1]]), int(c[f[2]]))))
        seen.pop(k, None) if k in seen else seen.setdefault(k, t)
faces = np.array(list(seen.keys()), dtype=int)

# the run's own predicates
root_n = pts[:, 0] < TOL
tip_n = (pts[:, 0] > L - TOL) & (pts[:, 2] < SPAN)
print(f"clamped nodes: {int(root_n.sum())}   loaded nodes: {int(tip_n.sum())}")

# a face belongs to a region when ALL THREE of its nodes do -- the same rule the assembler uses to
# decide whether a facet carries the traction, which is why a band narrower than the spacing loads
# nothing at all.
f_root = root_n[faces].all(axis=1)
f_tip = tin = tip_n[faces].all(axis=1)
print(f"clamped facets: {int(f_root.sum())}   loaded facets: {int(f_tip.sum())}")


def project(P, az=-0.72, el=0.42):
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    w = P - np.array([L / 2, H / 2, W / 2])
    X = w[..., 0] * ca + w[..., 2] * sa
    Z = -w[..., 0] * sa + w[..., 2] * ca
    return np.stack([X, w[..., 1] * ce - Z * se, w[..., 1] * se + Z * ce], axis=-1)


def draw(ax, az, el, title):
    Q = project(pts[faces], az, el)
    col = np.where(f_root[:, None], CLAMP, np.where(f_tip[:, None], LOAD, ACCENT))
    o = np.argsort(Q[:, :, 2].mean(axis=1))
    Q, col = Q[o], col[o]
    A, B, C = Q[:, 0], Q[:, 1], Q[:, 2]
    n = np.cross(B - A, C - A)
    nl = np.linalg.norm(n, axis=1)
    nl[nl == 0] = 1.0
    lam = np.clip(np.abs(n @ np.array([0.34, 0.60, 0.72]) / nl) * 0.8 + 0.34, 0.3, 1.0)[:, None]
    ax.add_collection(PolyCollection(Q[:, :, :2], facecolors=col * lam,
                                     edgecolors="#12343B", linewidths=0.18))
    # the load arrow: from the loaded patch centroid, along the applied traction (0, 0, -1)
    c0 = pts[tip_n].mean(axis=0)
    tip = project(np.stack([c0, c0 + np.array([0.0, 0.0, -1.2])]), az, el)
    ax.annotate("", xy=tip[1, :2], xytext=tip[0, :2],
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#CB3F3F", shrinkA=0, shrinkB=0))
    lo = Q[:, :, :2].reshape(-1, 2)
    ax.set_xlim(min(lo[:, 0].min(), tip[:, 0].min()) - 0.15, max(lo[:, 0].max(), tip[:, 0].max()) + 0.15)
    ax.set_ylim(min(lo[:, 1].min(), tip[:, 1].min()) - 0.15, max(lo[:, 1].max(), tip[:, 1].max()) + 0.15)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)


if __name__ == "__main__":
    import matplotlib as mpl

    mpl.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
                         "font.family": "sans-serif",
                         "font.sans-serif": ["IBM Plex Sans", "FreeSans", "DejaVu Sans"],
                         "axes.titleweight": "light", "font.size": 9.5, "axes.titlesize": 9.5})
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    draw(axes[0], -0.72, 0.42, "oblique")
    draw(axes[1], 0.0, 0.0, "side  $x$–$y$   (the load points INTO the page)")
    draw(axes[2], 0.0, 1.5708, "plan  $x$–$z$   (the bending plane)")
    fig.savefig(os.path.join(OUT, "topo3d_problem.png"))
    print("wrote topo3d_problem.png")
