"""Does the optimiser's reported compliance match the body it actually produced?

Each design's extracted solid is re-solved on a UNIFORMLY refined mesh of the identical geometry
(midpoint subdivision is exact on planar facets, so every level is the same body). Linear tets are
stiff and refinement can only soften them, so the series is monotone and each level is a one-sided
LOWER bound on the truth -- which is what makes the comparison honest without a converged value.

Everything plotted is measured. The one computed quantity is the extrapolated limit, drawn as an
open marker and labelled as an extrapolation, never as data: it assumes the last observed increment
ratio continues, which is an assumption the series supports but does not prove.

    python scripts/fig_convergence.py            # writes runs/convergence.png
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUTDIR") or os.path.join(os.path.dirname(HERE), "runs")

# Measured: scripts/reanalysis_boundary.py, REFINE=4, the corrected h=0.25 pair.
RUNS = {
    "Lagrange": {
        "h": [0.2590, 0.1295, 0.0648, 0.0324, 0.0162],
        "C": [7.0454, 9.8143, 13.1901, 15.6882, 17.0401],
        "tets": [1605, 8828, 62744, 493518, 3948240],
        "reported": 4.8591,   # what the optimiser said, verified reproducible to 0.00 %
        "colour": "#d1495b",
    },
    "E-FEM": {
        "h": [0.2572, 0.1286, 0.0643, 0.0322, 0.0161],
        "C": [4.8111, 5.1298, 5.6298, 5.9528, 6.1108],
        "tets": [1636, 9083, 63809, 508261, 4049435],
        "reported": 6.0357,
        "colour": "#0d9488",
    },
}


def extrapolate(C):
    """Geometric tail from the LAST observed increment ratio. An estimate, not a measurement."""
    d = [C[i + 1] - C[i] for i in range(len(C) - 1)]
    r = d[-1] / d[-2]
    return (C[-1] + d[-1] * r / (1 - r), r) if 0 < r < 1 else (float("nan"), r)


fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.4))

for name, d in RUNS.items():
    lim, r = extrapolate(d["C"])
    a = ax[0]
    a.plot(d["h"], d["C"], "o-", color=d["colour"], lw=1.8, ms=6, label=f"{name}: refined solve")
    a.axhline(d["reported"], color=d["colour"], ls="--", lw=1.2, alpha=0.8)
    a.annotate(f"{name} reported {d['reported']:.3f}", xy=(d["h"][0], d["reported"]),
               xytext=(3, 4 if name == "E-FEM" else -13), textcoords="offset points",
               fontsize=8, color=d["colour"])
    a.plot([d["h"][-1] / 2], [lim], "o", mfc="none", mec=d["colour"], ms=9, mew=1.6)
    a.annotate(f"extrapolated {lim:.2f}", xy=(d["h"][-1] / 2, lim), xytext=(-6, 8),
               textcoords="offset points", fontsize=8, color=d["colour"], ha="right", style="italic")

ax[0].set_xscale("log"); ax[0].set_xlabel("element size h of the refined mesh"); ax[0].set_ylabel("compliance")
ax[0].invert_xaxis()
ax[0].set_title("The same body, re-solved on finer and finer meshes\n"
                "dashed = what the optimiser reported; open marker = extrapolated, not measured", fontsize=10)
ax[0].legend(fontsize=8, loc="center left"); ax[0].grid(alpha=0.25)

# --- right: the error each formulation makes on ONE body -----------------------------------------
# The E-FEM design, read three ways. No extrapolation enters this panel: it uses the level-4 value
# as the reference, so every bar is a LOWER bound on the error.
truth = RUNS["E-FEM"]["C"][-1]
reads = [("its own\nE-FEM solve", 6.0357, "#0d9488"), ("a plain\nP1 solve", 5.1628, "#d1495b")]
a = ax[1]
for i, (lbl, v, c) in enumerate(reads):
    err = 100 * (v / truth - 1)
    a.bar(i, err, color=c, width=0.5)
    a.text(i, err - 1.1, f"{err:+.2f} %", ha="center", va="top", fontsize=11, color="white", weight="bold")
a.axhline(0, color="k", lw=1.0)
a.set_xticks(range(len(reads))); a.set_xticklabels([r[0] for r in reads], fontsize=9)
a.set_ylabel("error against the refined solve of the same body (%)")
a.set_title(f"One design, one mesh, one density field\n"
            f"refined solve $\\geq$ {truth:.4f} at 4.05M tets", fontsize=10)
a.grid(axis="y", alpha=0.25)
a.set_ylim(-19, 3)

fig.suptitle("Reanalysis of the extracted solid — h = 0.25 pair, uniform refinement to 4M tets", fontsize=12)
fig.tight_layout()
dst = os.path.join(OUT, "convergence.png")
fig.savefig(dst, dpi=150)
print(f"wrote {dst}")
for name, d in RUNS.items():
    lim, r = extrapolate(d["C"])
    print(f"  {name:9s} bound >= {d['C'][-1]:8.4f}  extrapolated ~ {lim:7.4f}  "
          f"(ratio {r:.3f}, order O(h^{math.log2(1 / r):.2f}))  "
          f"reported {d['reported']:.4f} understates by >= {100 * (d['C'][-1] / d['reported'] - 1):.1f} %")
