"""The 2-D deformable-mesh cantilever as it optimises: an animation, and the enriched nodes.

Two things are drawn, both from the saved run and neither smoothed.

**The design.** Element density on the mesh WHERE THE OPTIMISER PUT IT -- the triangles are drawn at
their moving coordinates, so the node migration that the method rests on is visible directly rather
than described. Watch the mesh lines near the members: they crowd onto the boundary.

**The enriched nodes.** This run uses FULL enrichment (`space="cover"`), so every node carries the
extra cover degrees of freedom -- that is what the panel shows, and it is the honest answer. What is
also drawn, separately and labelled as such, is the mask the paper's own SELECTIVE enrichment rule
would produce: a node is dropped when every element touching it has density below 1e-3 (Jung, Yun &
Kim, *Comput. Struct.* **331** (2026) 108403, Sec. 3.6). That is computed from the same densities,
not from a second run, so it shows where the saving WOULD come from, not a measurement of one.

    python scripts/fig_2d_evolution.py            # writes the gif and the node figure
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
        "axes.labelcolor": "#1A202C",
        "axes.edgecolor": "#94A3B8",
        "xtick.color": "#1A202C",
        "ytick.color": "#1A202C",
    }
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

TEAL, INK, GRAY, FAINT = "#0D9488", "#1A202C", "#94A3B8", "#EDF2F4"
RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
RUN = os.environ.get("RUN", "tut2d_cover_frames.npz")

r = np.load(os.path.join(RUNS, RUN))
cells, pts0 = r["cells"], r["pts0"]
F_rho, F_pts, F_C, F_it = r["frames_rho"], r["frames_pts"], r["frames_C"], r["frames_it"]
L, H = float(pts0[:, 0].max()), float(pts0[:, 1].max())
print(f"{len(F_rho)} frames, {cells.shape[0]} triangles, {pts0.shape[0]} nodes, L={L} H={H}")


def polys(pts):
    return pts[cells]


# ---------------------------------------------------------------- the animation -------------------
fig, ax = plt.subplots(figsize=(7.6, 4.0))
ax.set_xlim(-0.5, L + 0.5)
ax.set_ylim(-0.5, H + 0.5)
ax.set_aspect("equal")
ax.set_axis_off()
pc = PolyCollection(polys(F_pts[0]), facecolors="white", edgecolors=GRAY, linewidths=0.15)
ax.add_collection(pc)
cap = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color=INK)

writer = PillowWriter(fps=4)
gif = os.path.join(RUNS, "tut2d_evolution.gif")
with writer.saving(fig, gif, dpi=150):
    for k in range(len(F_rho)):
        pc.set_verts(polys(F_pts[k]))
        shade = np.clip(F_rho[k], 0.0, 1.0)
        pc.set_facecolors(plt.cm.gray_r(shade))
        cap.set_text(f"iteration {int(F_it[k]):3d}      compliance {float(F_C[k]):8.3f}")
        writer.grab_frame()
    for _ in range(8):  # hold on the final design
        writer.grab_frame()
plt.close(fig)
print(f"wrote {gif}")

# ------------------------------------------------------- nodes: enrichment and motion -------------
rho_f, pts_f = F_rho[-1], F_pts[-1]
incident = [[] for _ in range(pts0.shape[0])]
for t, tri in enumerate(cells):
    for n in tri:
        incident[int(n)].append(t)
# The paper's Sec. 3.6 rule, applied to the converged density.
keep = np.array([bool(np.any(rho_f[np.array(inc)] >= 1e-3)) if inc else False for inc in incident])
moved = np.linalg.norm(pts_f - pts0, axis=1)

fig, ax = plt.subplots(1, 2, figsize=(11.6, 3.9))

ax[0].add_collection(PolyCollection(polys(pts_f), facecolors=plt.cm.gray_r(np.clip(rho_f, 0, 1)), edgecolors="none"))
ax[0].scatter(pts_f[keep, 0], pts_f[keep, 1], s=1.6, color=TEAL, zorder=3, label=f"kept enriched ({100 * keep.mean():.0f}%)")
ax[0].scatter(pts_f[~keep, 0], pts_f[~keep, 1], s=1.6, color=GRAY, zorder=3, label=f"droppable ({100 * (~keep).mean():.0f}%)")
ax[0].legend(frameon=False, fontsize=8, loc="upper right", markerscale=3)
ax[0].set_title("every node IS enriched in this run;\ncolour = what Sec. 3.6 would keep", fontsize=9)

sc = ax[1].scatter(pts_f[:, 0], pts_f[:, 1], s=2.4, c=moved, cmap="magma_r", vmin=0.0, vmax=float(np.percentile(moved, 99)))
cb = fig.colorbar(sc, ax=ax[1], shrink=0.85)
cb.set_label("|node displacement|", fontsize=8)
cb.outline.set_visible(False)
ax[1].set_title(f"how far each node MOVED  (median {np.median(moved[moved > 1e-12]):.3f}, bound 2.0)", fontsize=9)

for a in ax:
    a.set_xlim(-0.5, L + 0.5)
    a.set_ylim(-0.5, H + 0.5)
    a.set_aspect("equal")
    a.set_axis_off()
fig.tight_layout()
dst = os.path.join(RUNS, "tut2d_nodes.png")
fig.savefig(dst, dpi=190, facecolor="white")
print(f"wrote {dst}")
print(f"  enriched kept by the Sec 3.6 rule: {int(keep.sum())} of {keep.size} ({100 * keep.mean():.1f}%)")
print(f"  nodes that moved at all: {int((moved > 1e-12).sum())}, median |dx| {np.median(moved[moved > 1e-12]):.4f}")
