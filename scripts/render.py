"""Static figures + animation for the 3-D topology-optimisation run.

The projection, painter sort and screen-space Lambert shading here are the SAME algorithm the
interactive console uses, reimplemented in numpy so the stills and the console cannot disagree.
Everything drawn is computed by the model on this run -- no illustrative layers. The shading is a
rendering convention for conveying geometry; it carries no data.

Style: pikimam/analysis/_common.py.
"""

import os

import numpy as np
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")
os.makedirs(OUT, exist_ok=True)
run = np.load(os.path.join(OUT, os.environ.get("RUN", "topo3d_final.npz")))
cells, pts0 = run["cells"], run["pts0"]
fr_rho, fr_pts = run["frames_rho"], run["frames_pts"]
NT, NN, NF = cells.shape[0], pts0.shape[0], fr_rho.shape[0]
L, H, W = 4.0, 2.0, 2.0


def setup_style():
    """The house style, set through matplotlib directly.

    seaborn is not in any of this repo's pixi environments, and the style is only an rcParams
    block plus despine — both reproduced here rather than adding a dependency for a figure script.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--", "grid.color": "#94A3B8",
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "Frutiger 45 Light", "FreeSans", "DejaVu Sans"],
        "font.weight": "light", "axes.titleweight": "light", "axes.labelweight": "bold",
        "axes.edgecolor": "#1A202C", "axes.labelcolor": "#1A202C", "text.color": "#1A202C",
        "xtick.color": "#1A202C", "ytick.color": "#1A202C",
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    })

    class _S:
        @staticmethod
        def despine(ax=None):
            for a in ([ax] if ax is not None else plt.gcf().axes):
                a.spines["top"].set_visible(False)
                a.spines["right"].set_visible(False)

    return _S(), plt


LF = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def surface(rho, level=0.5):
    """Boundary faces of the solid region: a face of a solid tet used by exactly one solid tet."""
    solid = np.where(rho > level)[0]
    seen = {}
    for t in solid:
        c = cells[t]
        for f in LF:
            k = tuple(sorted((int(c[f[0]]), int(c[f[1]]), int(c[f[2]]))))
            if k in seen:
                del seen[k]
            else:
                seen[k] = t
    faces = np.array(list(seen.keys()), dtype=int) if seen else np.zeros((0, 3), int)
    owner = np.array(list(seen.values()), dtype=int) if seen else np.zeros(0, int)
    return faces, owner, solid.size


def closed(faces):
    """A closed surface uses every edge exactly twice — the check that the extraction is sound."""
    if len(faces) == 0:
        return True, {}
    cnt = {}
    for tri in faces:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            k = (min(tri[a], tri[b]), max(tri[a], tri[b]))
            cnt[k] = cnt.get(k, 0) + 1
    from collections import Counter
    return all(v == 2 for v in cnt.values()), Counter(cnt.values())


def project(pts, faces, az=-0.72, el=0.42):
    """Orthographic yaw-then-pitch. Returns screen (x, y) per face vertex plus depth."""
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    w = pts - np.array([L / 2, H / 2, W / 2])
    X = w[:, 0] * ca + w[:, 2] * sa
    Z = -w[:, 0] * sa + w[:, 2] * ca
    Y = w[:, 1] * ce - Z * se
    Dp = w[:, 1] * se + Z * ce
    P = np.stack([X, Y, Dp], axis=1)
    return P[faces]  # (n_faces, 3, 3)


VIRIDIS = np.array([[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
                    [31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]]) / 255.0


def ramp(t):
    t = np.clip(np.nan_to_num(t), 0, 1) * (len(VIRIDIS) - 1)
    i = np.minimum(len(VIRIDIS) - 2, np.floor(t).astype(int))
    f = (t - i)[:, None]
    return VIRIDIS[i] * (1 - f) + VIRIDIS[i + 1] * f


ACCENT = np.array([13, 148, 136]) / 255.0  # the house teal


def draw(ax, pts, faces, vals=None, *, vmin=0.0, vmax=1.0, az=-0.72, el=0.42, solid=True):
    """Painter's algorithm + screen-space Lambert, identical to the console's fill loop.

    ``solid=True`` renders one material, which is the topology-optimisation convention and what
    these panels want: every face drawn is already above the iso-level, so mapping the remaining
    half of the density range onto a colour axis spends it on nothing. Pass ``solid=False`` with
    ``vals`` to colour by a field that actually varies over the surface.
    """
    if len(faces) == 0:
        ax.set_axis_off()
        return
    Q = project(pts, faces, az, el)
    order = np.argsort(Q[:, :, 2].mean(axis=1))  # far first
    Q = Q[order]
    A, B, C = Q[:, 0], Q[:, 1], Q[:, 2]
    n = np.cross(B - A, C - A)
    nl = np.linalg.norm(n, axis=1)
    nl[nl == 0] = 1.0
    Lt = np.array([0.34, 0.60, 0.72])
    lam = np.clip(np.abs(n @ Lt / nl) * 0.8 + 0.34, 0.3, 1.0)[:, None]
    if solid:
        rgb = np.repeat(ACCENT[None, :], len(Q), axis=0) * lam
    else:
        v = np.asarray(vals)[order]
        rgb = ramp((v - vmin) / (vmax - vmin if vmax > vmin else 1.0)) * lam
    ax.add_collection(PolyCollection(Q[:, :, :2], facecolors=rgb, edgecolors=rgb, linewidths=0.25))
    pad = 0.06
    ax.set_xlim(Q[:, :, 0].min() - pad, Q[:, :, 0].max() + pad)
    ax.set_ylim(Q[:, :, 1].min() - pad, Q[:, :, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()


# ------------------------------------------------------------------ validate the extraction
faces_f, owner_f, nsolid = surface(fr_rho[-1])
ok, hist = closed(faces_f)
print(f"final design: {nsolid}/{NT} solid tets, {len(faces_f)} boundary faces")
# An edge used twice is a manifold surface. An edge used 2k times is k surface sheets meeting
# there -- i.e. the solid region PINCHES along that edge. Not an extraction defect: it is the
# line-pinch the edge-fan criterion exists to suppress, counted directly off the design.
print(f"edge-use histogram {dict(hist)}  ->  manifold: {ok}")
pinched = sum(v for k, v in hist.items() if k > 2)
print(f"LINE PINCHES: {pinched} edges carry more than one surface sheet "
      f"({100.0 * pinched / max(sum(hist.values()), 1):.1f}% of surface edges)")

sns, plt = setup_style()

# ------------------------------------------------------------------ 1. the structure, two views
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
for ax, (az, el, lab) in zip(axes, [(-0.72, 0.42, "oblique"), (0.0, 0.0, "side  $x$–$y$"), (0.0, 1.5708, "plan  $x$–$z$")]):
    draw(ax, fr_pts[-1], faces_f, az=az, el=el)
    ax.set_title(lab)
fig.savefig(os.path.join(OUT, "topo3d_structure.png"))
plt.close(fig)
print("wrote topo3d_structure.png")

# ------------------------------------------------------------------ 2. summary panel
fig = plt.figure(figsize=(11, 6.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.30)
axS = fig.add_subplot(gs[0, :])
draw(axS, fr_pts[-1], faces_f)
axS.set_title(r"optimised structure   ·   iso-surface $\bar\rho = 0.5$ on the deformed mesh")

axC = fig.add_subplot(gs[1, 0])
axC.plot(run["it"], run["C"], color="#0D9488", lw=1.6)
axC.set_xlabel("MMA iteration"); axC.set_ylabel("compliance $C$")
axC.set_yscale("log"); sns.despine(ax=axC)

axV = fig.add_subplot(gs[1, 1])
axV.plot(run["it"], run["V"], color="#0D9488", lw=1.4, label=r"volume / budget")
axV.plot(run["it"], run["mnd"], color="#94A3B8", lw=1.4, label=r"$M_{nd}$")
axV.axhline(1.0, color="#1A202C", lw=0.8, ls="--")
axV.set_xlabel("MMA iteration"); axV.set_ylabel("constraint / grey level")
axV.legend(fontsize=7); sns.despine(ax=axV)

axH = fig.add_subplot(gs[1, 2])
axH.hist(fr_rho[-1], bins=28, range=(0, 1), color="#0D9488", alpha=0.85)
axH.set_xlabel(r"physical density $\bar\rho$"); axH.set_ylabel("tetrahedra")
axH.set_yscale("log"); sns.despine(ax=axH)
fig.savefig(os.path.join(OUT, "topo3d_summary.png"))
plt.close(fig)
print("wrote topo3d_summary.png")

# ------------------------------------------------------------------ 3. the evolution, as a GIF
from PIL import Image  # noqa: E402

tmp = os.path.join(OUT, "_frames")
os.makedirs(tmp, exist_ok=True)
paths = []
for i in range(NF):
    fa, ow, _ = surface(fr_rho[i])
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    draw(ax, fr_pts[i], fa)
    ax.set_title(f"iteration {int(run['it'][i]):3d}   ·   $C$ = {run['C'][i]:7.2f}", fontsize=9)
    # every frame on the SAME screen box, so the structure does not jitter between frames
    ax.set_xlim(-2.35, 2.35); ax.set_ylim(-1.35, 1.35)
    p = os.path.join(tmp, f"f{i:03d}.png")
    fig.savefig(p, dpi=110, facecolor="white")
    plt.close(fig)
    paths.append(p)

ims = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128) for p in paths]
gif = os.path.join(OUT, "topo3d_evolution.gif")
ims[0].save(gif, save_all=True, append_images=ims[1:] + [ims[-1]] * 8, duration=140, loop=0, optimize=True)
print(f"wrote topo3d_evolution.gif  ({os.path.getsize(gif) / 1e6:.2f} MB, {len(ims)} frames)")
