"""Where the density actually sits, before and after the projection worked.

The iso-level slider in the console shows this qualitatively -- drag it on a grey design and the
structure dissolves; drag it on a binary one and nothing moves. This puts numbers on the same thing,
because "the structure dissolves" is the difference between a design that survives being made real
and one that does not.

Everything plotted is the measured density field from the run's own npz. Nothing is smoothed,
binned adaptively, or fitted. The only drawn line that is not data is the rho = 0.5 marker, which is
the conventional extraction level and is labelled as such.

    python scripts/fig_grey.py            # writes runs/grey.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")

RUNS = [
    ("h25c_Lagrange", "Lagrange, no projection", "#d1495b", 0),
    ("hv_Lagrange", "Lagrange, projected", "#d1495b", 1),
    ("h25c_cover", "E-FEM, no projection", "#0d9488", 0),
    ("hv_cover", "E-FEM, projected", "#0d9488", 1),
]


def _vols(cells, pts):
    v = pts[cells]
    return np.abs(np.linalg.det(np.stack([v[:, i + 1] - v[:, 0] for i in range(3)], -1))) / 6.0


fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
rows = []
for tag, label, colour, panel in RUNS:
    r = np.load(os.path.join(OUT, f"topo3d_{tag}.npz"))
    rho, cells, pts = r["rho_f"], r["cells"], r["pts_f"]
    vol = _vols(cells, pts)
    a = ax[panel]
    # VOLUME-weighted, not cell-weighted: a design is judged by the material it holds, and a mesh
    # with many small void cells would otherwise flatter itself.
    a.hist(rho, bins=np.linspace(0, 1, 51), weights=vol, color=colour, alpha=0.65, label=label)
    grey = float(vol[(rho > 0.1) & (rho < 0.9)].sum())
    m_nd = float(4.0 * (rho * (1.0 - rho)).mean())
    # The fraction of the design's MATERIAL that lies above the extraction level -- density-weighted
    # on both sides. Dividing raw cell volume by material instead counts a rho = 0.6 cell as fully
    # solid and can exceed 100 %, which measures the extraction's rounding-up rather than the loss.
    keep = float((rho * vol)[rho > 0.5].sum()) / float((rho * vol).sum())
    rows.append((label, m_nd, grey, 100 * keep))

for panel, title in ((0, "no projection"), (1, "with projection")):
    a = ax[panel]
    a.axvline(0.5, color="k", lw=1.0, ls="--", alpha=0.6)
    a.annotate("rho = 0.5\nextraction level", xy=(0.5, 0.9), xycoords=("data", "axes fraction"),
               ha="center", fontsize=8, color="0.3")
    a.set_yscale("log")
    a.set_xlabel("element density")
    a.set_title(title, fontsize=11)
    a.legend(fontsize=8, loc="upper center")
    a.grid(alpha=0.2)
ax[0].set_ylabel("material volume in bin (log)")

fig.suptitle(
    "Where the density sits. Material between the peaks is what thresholding destroys.", fontsize=12
)
fig.tight_layout()
dst = os.path.join(OUT, "grey.png")
fig.savefig(dst, dpi=150)
print(f"wrote {dst}\n")
print(f"{'design':<28}{'M_nd':>9}{'grey volume':>14}{'material above rho>0.5':>26}")
print("-" * 77)
for label, m_nd, grey, keep in rows:
    print(f"{label:<28}{m_nd:>9.4f}{grey:>14.4f}{keep:>25.1f}%")
