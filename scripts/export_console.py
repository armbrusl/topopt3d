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


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")
os.makedirs(OUT, exist_ok=True)
run = np.load(os.path.join(OUT, os.environ.get("RUN", "topo3d_final.npz")))
cells, pts0, pts_f, rho_f = (run[k] for k in ("cells", "pts0", "pts_f", "rho_f"))
fr_rho, fr_pts = run["frames_rho"], run["frames_pts"]
print(f"tets={cells.shape[0]} nodes={pts0.shape[0]} frames={fr_rho.shape[0]}")

# --- per-tet mesh-quality fields on the FINAL deformed mesh -------------------------------------
# Computed from `cells` and `pts_f` directly rather than by rebuilding the domain and evaluating
# jNO's geometry nodes. The rebuild had to reproduce the run's geometry exactly, which tied this
# script to `Shape.box`: it refused a tapered blank with bolt holes outright. The formulas below are
# the same ones `cell_volume` / `cell_angles` / `cell_aspect` implement for tetrahedra, so nothing is
# approximated -- and the extents now only set the camera and the cut-away range.
L = float(os.environ.get("LX", "4.0"))
H = float(os.environ.get("LY", "2.0"))
W = float(os.environ.get("LZ", "2.0"))
h = float(os.environ.get("HMESH", "0.10"))
_v = pts_f[cells]                                            # (n_cells, 4, 3)
cvol = np.abs(np.linalg.det(np.stack([_v[:, i + 1] - _v[:, 0] for i in range(3)], -1))) / 6.0


def _inward_normal(k):
    keep = [i for i in range(4) if i != k]
    n = np.cross(_v[:, keep[1]] - _v[:, keep[0]], _v[:, keep[2]] - _v[:, keep[0]])
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-30)
    s = np.sign(np.sum(n * (_v[:, k] - _v[:, keep[0]]), axis=-1, keepdims=True))
    return n * np.where(s == 0.0, 1.0, s)


_nrm = [_inward_normal(k) for k in range(4)]
# The dihedral across edge (i, j) is pi minus the angle between the inward normals of the two faces
# that omit the OTHER two vertices -- the same pairing `cell_angles` uses.
dihed = np.degrees(np.stack([
    np.pi - np.arccos(np.clip(np.sum(_nrm[a] * _nrm[b], axis=-1), -1.0, 1.0))
    for (i, j) in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for (a, b) in [tuple(k for k in range(4) if k not in (i, j))]
], axis=-1))
_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_longest = np.max(np.stack([np.linalg.norm(_v[:, j] - _v[:, i], axis=-1) for i, j in _pairs], -1), -1)
_surf = 0.0
for k in range(4):
    keep = [i for i in range(4) if i != k]
    e = np.stack([_v[:, keep[i + 1]] - _v[:, keep[0]] for i in range(2)], axis=-1)
    gram = np.einsum("cij,cik->cjk", e, e)
    _surf = _surf + np.sqrt(np.clip(np.linalg.det(gram), 0.0, None)) / 2.0
aspect = _longest / (2.0 * np.sqrt(6.0) * (3.0 * cvol / (_surf + 1e-30)) + 1e-30)
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
        "maxmove": float(moved.max()),
        "movebound": float(os.environ.get("MOVE_BOUND_FRAC", "2.0")) * h,
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

# --- boundary conditions, if the run recorded them ---------------------------------------------
# Taken from the ASSEMBLED right-hand side and the constrained node set, so the console draws what
# the operator carries rather than what the predicates say. Loaded nodes are sent with their force
# vector; a node with no entry in `b` simply is not loaded, whatever the region predicate claimed.
if "load_f" in run.files:
    lf = np.asarray(run["load_f"], float)
    mag = np.linalg.norm(lf, axis=1)
    live = np.flatnonzero(mag > 1e-12)
    data["bc"] = {
        # UNIT direction per node; the relative magnitude rides in `scale` and the template applies
        # it once. Normalising the vector by the global max instead would apply magnitude twice.
        "load": [[int(n)] + r(lf[n] / mag[n], 4) for n in live],
        "fixed": np.flatnonzero(np.asarray(run["fixed"])).astype(int).tolist() if "fixed" in run.files else [],
        "fmax": float(mag.max()),
        "scale": [r(mag[n] / max(mag.max(), 1e-300), 3) for n in live],
    }
    print(f"bc: {len(live)} loaded nodes, {len(data['bc']['fixed'])} fixed nodes")
if "solid" in run.files:
    data["nondesign"] = np.flatnonzero(np.asarray(run["solid"])).astype(int).tolist()

blob = json.dumps(data, separators=(",", ":"))
# TAG keeps a pseudo-2D console from overwriting the 3-D one; both are worth keeping side by side.
TAG = os.environ.get("TAG", "")
_sfx = f"_{TAG}" if TAG else ""
out_json = os.path.join(OUT, f"topo3d_web{_sfx}.json")
with open(out_json, "w") as f:
    f.write(blob)
print(f"wrote {out_json}  ({len(blob) / 1e6:.2f} MB)")

tpl = os.path.join(HERE, "console_template.html")
dst = os.path.join(OUT, f"console{_sfx}.html")
if os.path.exists(tpl):
    html = open(tpl).read()
    new, n = re.subn(r"const BUILTIN = \{.*?\};\n", "const BUILTIN = " + blob + ";\n", html, count=1, flags=re.S)
    assert n == 1, "BUILTIN marker not found in the template"
    open(dst, "w").write(new)
    print(f"wrote {dst}  ({len(new) / 1e6:.2f} MB)")
