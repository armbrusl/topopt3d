"""The material budget and the mesh-quality bounds, in one place with their calibration.

These stay in ``scripts/`` rather than jNO on purpose. ``cell_angles``, ``cell_aspect`` and
``cell_volume`` are jNO's -- they are properties of a mesh and mean the same thing everywhere. The
*bounds* are not: 15 deg is chosen because gmsh hands this mesh family a 15.23 deg worst dihedral,
and the volume window is the paper's ratio taken against this mesh's own median. Burying a number
like that behind a library default is how it stops being questioned.

The budget is the part that has to be shared. Four scripts still write

    volume = (rho * cellv).sum / (VOLFRAC * cellv.sum)

and ``cellv`` is a functional of the MOVING mesh, so the denominator is a moving target. With the
outline free, the bracket pushed its boundary outward and was handed 42 % more material for it:
blank 12.72 -> 18.05, material 3.81 -> 5.39, and a compliance of 3.11 against 4.07 that was bought
rather than designed. Tangential sliding already pins the blank, but the fixed denominator is what
makes the budget mean one thing regardless of what any future predicate lets move -- and ``cellv``
is unsigned (an inverted cell still reads positive), so "the boundary is pinned" is not by itself
enough to make the moving version safe.
"""

import numpy as np

from .mesh import cell_volumes


def material_budget(rho, cellv, pts0, cells, volfrac, *, log=print):
    """``(rho * cellv).sum / (volfrac * V0)`` with ``V0`` the INITIAL blank volume, a constant.

    Returns the constraint expression, to be bounded at 1. ``cellv`` is jNO's ``d.cell_volume()``
    node, so the numerator still tracks the moving mesh -- material is material wherever the nodes
    are. Only the budget is frozen.
    """
    v0 = float(cell_volumes(cells, pts0).sum())
    log(f"material budget: {volfrac} x {v0:.4f} = {volfrac * v0:.4f} (fixed, against the initial blank)")
    return (rho.reshape(-1) * cellv).sum / (volfrac * v0)


def mesh_quality(d, pts0, cells, *, theta_min_deg=15.0, aspect_max=8.0, pn=50.0,
                 v0min_frac=0.115, v0max_frac=1.732, rho=None, log=print):
    """The four geometric constraints, each normalised so ``<= 1`` is the feasible side.

    ``theta_min_deg`` -- the paper uses 20 deg on TRIANGLE interior angles, whose ideal is 60. In 3-D
    ``cell_angles()`` returns the six DIHEDRALS, ideal 70.53, so the same ratio would be 23.5. But
    gmsh hands this mesh a worst dihedral of 15.23 deg, and a bound above that starts violated --
    while the paper's Fig. 6c shows the constraint marginally active at convergence, not violated
    from iteration 0. 15 deg is the tightest bound this mesh family satisfies.

    ``v0min_frac`` / ``v0max_frac`` -- eq. (25)/(27) use [0.05, 0.75] against a nominal unit-edge
    element, i.e. [0.115, 1.732] x nominal. The RATIO is what transfers, not the numbers: gmsh's
    ``size=h`` does not make ideal elements, and the median tet here is 1.79x the regular tet of edge
    h, so against the true nominal 53 % of cells would start infeasible. Taken against the median
    instead, 0.00 % start under the floor and 0.05 % over the active cap.

    ``rho`` is optional and only used for the ACTIVE cap ``(2 - rho) * V0MAX``, which lets a void
    cell grow further than a solid one. Pass ``None`` for a fixed cap.

    Returns ``dict(g_ang=, g_asp=, g_vmx=, g_vmn=)``; a caller wraps each in ``jno.le(..., 1.0)``.
    Named, so a feasibility report can say which one is binding rather than "inequality row 2".
    """
    theta_min = np.deg2rad(theta_min_deg)
    v0 = cell_volumes(cells, pts0)
    v0_med = float(np.median(v0))
    v0max, v0min = v0max_frac * v0_med, v0min_frac * v0_med
    cellv = d.cell_volume()

    # (2*pi - angle) / (2*pi - theta_min): a p-norm over the SMALL end, so the worst dihedral is what
    # binds. `normalize=True` is the Le et al. / Jung eq. (29)-(30) form -- value is the true max,
    # gradient is the smooth p-norm.
    g_ang = ((2 * np.pi - d.cell_angles()) / (2 * np.pi - theta_min)).reshape(-1).pnorm(pn, normalize=True)
    g_asp = (d.cell_aspect() / aspect_max).reshape(-1).pnorm(pn, normalize=True)
    cap = (2.0 - rho.reshape(-1)) * v0max if rho is not None else v0max
    g_vmx = (cellv / cap).reshape(-1).pnorm(pn, normalize=True)
    g_vmn = (v0min / cellv).reshape(-1).pnorm(pn, normalize=True)

    n_lo = int((v0 < v0min).sum())
    n_hi = int((v0 > v0max).sum())
    log(
        f"mesh quality: theta_min {theta_min_deg} deg, aspect <= {aspect_max}, "
        f"V in [{v0min:.4g}, {v0max:.4g}] against a median of {v0_med:.4g}; "
        f"at iteration 0 {100 * n_lo / v0.size:.2f} % start under the floor and "
        f"{100 * n_hi / v0.size:.2f} % over the cap"
    )
    return {
        "g_ang": g_ang.name("g_ang"),
        "g_asp": g_asp.name("g_asp"),
        "g_vmx": g_vmx.name("g_vmx"),
        "g_vmn": g_vmn.name("g_vmn"),
    }
