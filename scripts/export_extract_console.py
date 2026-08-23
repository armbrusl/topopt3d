"""Console for the body the REANALYSIS actually solved, not the density field it came from.

`export_console.py` draws the design: a continuous rho on the optimiser's mesh, with the iso-level
live in the browser. That is the right view of a design and the wrong view of a reanalysis, because
the reanalysis does not solve a density field at all -- it solves the SOLID that survives
thresholding at the volume-preserving level, minus whatever the manifold repair had to drop, minus
any disconnected piece. Those three steps are exactly where a design can turn out to be less than it
looked, so the object they produce is worth seeing on its own.

Refinement is not: midpoint subdivision is exact on planar facets, so the 4M-tet level-4 mesh is the
identical geometry to level 0 and shows nothing new. There is one body per design, and this is it.

Reads `runs/extract_<TAG>.npz`, written by `EXTRACT_ONLY=1 RUN=... reanalysis_boundary.py`.

    TAG=h25c_cover python scripts/export_extract_console.py
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")
TAG = os.environ.get("TAG", "h25c_cover")

src = np.load(os.path.join(OUT, f"extract_{TAG}.npz"))
mask, cells, pts = src["mask"], src["cells"], src["pts"]
design = np.load(os.path.join(OUT, f"topo3d_{TAG}.npz"))

# A BINARY field: 1 for a tet the reanalysis kept, RHO_MIN for one it did not. The console's
# iso-level slider then has nothing to interpolate -- every level between the two shows the same
# body, which is the honest way to display a design that has already been thresholded. Sliding it on
# the ORIGINAL console is a different and equally useful thing (it shows how much of the design is
# grey); this one deliberately cannot mislead that way.
rho_bin = np.where(mask, 1.0, 1e-3)
frames = np.repeat(rho_bin[None, :], design["frames_rho"].shape[0], axis=0)

dst = os.path.join(OUT, f"topo3d_extract_{TAG}.npz")
np.savez(
    dst,
    cells=cells, pts0=design["pts0"], pts_f=pts,
    rho_f=rho_bin, frames_rho=frames,
    frames_pts=np.repeat(pts[None, :, :], frames.shape[0], axis=0),
    solid=design["solid"], load_f=design["load_f"], fixed=design["fixed"],
    **{k: design[k] for k in ("it", "C", "V", "P", "S", "mnd", "ang", "asp", "vmx", "vmn")},
)
kept_frac = 100.0 * mask.sum() / mask.size
print(f"{TAG}: {int(mask.sum())} of {mask.size} tets kept ({kept_frac:.1f} %), "
      f"level rho > {float(src['tau']):.4f}, material {float(src['material_kept']):.4f} "
      f"against a target of {float(src['material_target']):.4f}")

env = dict(os.environ, RUN=os.path.basename(dst), TAG=f"extract_{TAG}")
sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "export_console.py")], env=env))
