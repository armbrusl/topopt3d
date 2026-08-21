"""How much of our extra structural detail is just an unbound perimeter target?

The paper is explicit that its method produces fine-scale features and that perimeter control is the
remedy (Sec. 3.5): "such fine-scale features may not always be practical in engineering
applications ... To regulate the geometric complexity, the perimeter of the material-void interface
is introduced as a measure of structural detail." Their runs use P* = 650 / 500 / 600.

Our cantilever ran with P* = 650 against a NATURAL perimeter of 513.7, so the log barrier -- which
only bites as P approaches P* -- never engaged. This sweeps P* into the binding range and measures
what it costs.

Complexity is counted, not eyeballed: the number of enclosed voids (holes) in the solid element set,
found by flooding the void elements through shared edges and discarding the one that reaches the
domain boundary. That is a topological count, so it does not depend on any threshold beyond rho>0.5.

    python scripts/fig_perimeter_sweep.py
"""

import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "FreeSans", "DejaVu Sans"],
        "font.size": 9.0,
        "text.color": "#1A202C",
    }
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
INK, TEAL, GRAY = "#1A202C", "#0D9488", "#94A3B8"

CASES = [
    ("tut2d.npz", "P* = 650  (never binds)", 513.7, 78.97),
    ("ps450.npz", "P* = 450", 435.8, 78.61),
    ("ps350.npz", "P* = 350", 348.7, 78.05),
    ("ps250.npz", "P* = 250", 250.3, 81.60),
]


def holes(cells, rho):
    """Enclosed voids in the solid set: flood the void elements through shared EDGES, drop the
    component that touches the domain boundary, count what is left."""
    solid = rho > 0.5
    edge: dict = {}
    for t, tri in enumerate(cells):
        for a, b in ((0, 1), (1, 2), (0, 2)):
            k = (min(tri[a], tri[b]), max(tri[a], tri[b]))
            edge.setdefault(k, []).append(t)
    nbr = [[] for _ in range(len(cells))]
    outer = set()
    for k, ts in edge.items():
        if len(ts) == 2:
            nbr[ts[0]].append(ts[1])
            nbr[ts[1]].append(ts[0])
        else:
            outer.add(ts[0])  # a boundary facet: this element touches the domain edge
    seen = np.zeros(len(cells), dtype=bool)
    n_enclosed = 0
    for t in range(len(cells)):
        if solid[t] or seen[t]:
            continue
        stack, comp, touches = [t], [], False
        seen[t] = True
        while stack:
            c = stack.pop()
            comp.append(c)
            if c in outer:
                touches = True
            for m in nbr[c]:
                if not solid[m] and not seen[m]:
                    seen[m] = True
                    stack.append(m)
        if not touches and len(comp) > 2:  # >2 elements, so a single sliver is not a "hole"
            n_enclosed += 1
    return n_enclosed


data = []
for fn, label, P, C in CASES:
    r = np.load(os.path.join(RUNS, fn))
    h = holes(r["cells"], r["rho_f"])
    data.append((r, label, P, C, h))
    print(f"{label:26s}  P={P:7.1f}  C={C:7.2f}  enclosed voids={h:3d}  solid={int((r['rho_f'] > 0.5).sum())}")

L, H = 60.0, 30.0
fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.0))
for ax, (r, label, P, C, h) in zip(axes.ravel(), data):
    pts, rho, cells = r["pts_f"], r["rho_f"], r["cells"]
    ax.add_collection(PolyCollection(pts[cells], facecolors=plt.cm.gray_r(np.clip(rho, 0, 1)), edgecolors="none"))
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, H + 0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(label, fontsize=10, color=INK)
    ax.text(0.5, -0.04, f"P = {P:.1f}    C = {C:.2f}    enclosed voids = {h}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=GRAY)
fig.tight_layout()
dst = os.path.join(RUNS, "perimeter_sweep.png")
fig.savefig(dst, dpi=190, facecolor="white")
print(f"wrote {dst}")

# ---- the same four, as an animation, so the coarsening reads as one motion -----------------------
fig2, ax = plt.subplots(figsize=(7.4, 3.9))
ax.set_xlim(-0.5, L + 0.5)
ax.set_ylim(-0.5, H + 0.5)
ax.set_aspect("equal")
ax.set_axis_off()
pc = PolyCollection(data[0][0]["pts_f"][data[0][0]["cells"]], edgecolors="none")
ax.add_collection(pc)
cap = ax.text(0.5, 1.03, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=11, color=INK)
sub = ax.text(0.5, -0.03, "", transform=ax.transAxes, ha="center", va="top", fontsize=9, color=GRAY)

gif = os.path.join(RUNS, "perimeter_sweep.gif")
with PillowWriter(fps=1).saving(fig2, gif, dpi=150) as w:
    for r, label, P, C, h in data:
        pc.set_verts(r["pts_f"][r["cells"]])
        pc.set_facecolors(plt.cm.gray_r(np.clip(r["rho_f"], 0, 1)))
        cap.set_text(label)
        sub.set_text(f"perimeter {P:.1f}    compliance {C:.2f}    enclosed voids {h}")
        for _ in range(3):  # hold each design ~3 s
            w.grab_frame()
print(f"wrote {gif}")
