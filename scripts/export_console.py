"""Serialise the 3-D topology-optimisation run for the interactive console.

Ships the RAW per-tet data rather than a pre-extracted surface, so the iso-level and the iteration
are both live in the browser. Every field here is computed by jNO on this run.

Follows pikimam/analysis/export_web.py: round hard, dump minified, inline into the HTML by regex.
"""

import json
import os
import re

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")
os.makedirs(OUT, exist_ok=True)
run = np.load(os.path.join(OUT, os.environ.get("RUN", "topo3d_final.npz")))
cells, pts0, pts_f, rho_f = (run[k] for k in ("cells", "pts0", "pts_f", "rho_f"))
fr_rho, fr_pts = run["frames_rho"], run["frames_pts"]
print(f"tets={cells.shape[0]} nodes={pts0.shape[0]} frames={fr_rho.shape[0]}")

# --- per-tet mesh-quality fields on the FINAL deformed mesh, through jNO's own nodes ----------
# The mesh must be rebuilt to evaluate the geometry nodes on the deformed configuration, so it has
# to match the run exactly -- hence HMESH from the environment, the same knob run.py takes. The
# assertion below is what catches a mismatch instead of silently reporting another mesh's quality.
L, H, W = 4.0, 2.0, 2.0
h = float(os.environ.get("HMESH", "0.10"))
d = jno.Shape.box(0, 0, 0, L, H, W, size=h).domain()
assert np.array_equal(np.asarray(d._cells_p1()), cells), "mesh differs from the run"
d.mesh.points[:, :3] = pts_f  # evaluate the geometry nodes ON the deformed configuration
aspect = np.asarray(d.cell_aspect().eval()).reshape(-1)
dihed = np.degrees(np.asarray(d.cell_angles().eval()))  # (n_cells, 6)
cvol = np.asarray(d.cell_volume().eval()).reshape(-1)
moved = np.linalg.norm(pts_f - pts0, axis=1)
print(f"aspect [{aspect.min():.2f},{aspect.max():.2f}]  min dihedral {dihed.min():.2f} deg  "
      f"min |K| {cvol.min():.3e}  max node move {moved.max():.4f}")

STRIDE = int(os.environ.get("FRAME_STRIDE", "1"))
r = lambda a, n: np.round(np.asarray(a, float), n).tolist()  # noqa: E731

data = {
    "meta": {
        "L": L, "H": H, "W": W, "h": h, "volfrac": float(os.environ.get("VOLFRAC", "0.20")), "penal": 3.0,
        "nt": int(cells.shape[0]), "nn": int(pts0.shape[0]), "nf": int(len(fr_rho[::STRIDE])),
        "iters": int(run["it"][-1]),
        "C0": float(run["C"][0]), "C1": float(run["C"][-1]),
        "V1": float(run["V"][-1]), "mnd": float(run["mnd"][-1]),
        "minvol": float(cvol.min()), "mindih": float(dihed.min()), "maxaspect": float(aspect.max()),
        "maxmove": float(moved.max()), "movebound": 0.30 * h,
    },
    "cells": cells.astype(int).ravel().tolist(),
    "pts": r(pts_f.ravel(), 5),
    "pts0": r(pts0.ravel(), 5),
    # per-frame: node positions + densities, so the slider drives geometry AND design
    # Frame payload dominates the file, and the artifact host caps a page at 16 MB. Stride the
    # frames and round harder: the density is what animates, and the mesh motion is bounded by the
    # per-node box (measured max |dx| = 0.048 here, about half an element), so 4 dp on coordinates
    # and 3 on density are both well below anything visible at screen resolution.
    "frames": {"rho": [r(f, 3) for f in fr_rho[::STRIDE]], "pts": [r(f.ravel(), 4) for f in fr_pts[::STRIDE]]},
    # per-tet scalar fields on the final configuration
    "fields": {
        "rho": r(rho_f, 4),
        "aspect": r(aspect, 3),
        "dihmin": r(dihed.min(axis=1), 2),
        "vol": r(cvol, 6),
    },
    "moved": r(moved, 5),
    "hist": {k: r(run[k], 6) for k in ("it", "C", "V", "mnd", "ang", "asp")},
}

blob = json.dumps(data, separators=(",", ":"))
out_json = os.path.join(OUT, "topo3d_web.json")
with open(out_json, "w") as f:
    f.write(blob)
print(f"wrote {out_json}  ({len(blob) / 1e6:.2f} MB)")

tpl = os.path.join(HERE, "console_template.html")
dst = os.path.join(OUT, "console.html")
if os.path.exists(tpl):
    html = open(tpl).read()
    new, n = re.subn(r"const BUILTIN = \{.*?\};\n", "const BUILTIN = " + blob + ";\n", html, count=1, flags=re.S)
    assert n == 1, "BUILTIN marker not found in the template"
    open(dst, "w").write(new)
    print(f"wrote {dst}  ({len(new) / 1e6:.2f} MB)")
