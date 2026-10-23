# topopt3d — 3-D topology optimisation on a deformable tetrahedral mesh

A 3-D extension of Jung, Yun & Kim, *Density-based topology optimization on a deformable
unstructured mesh with enriched finite elements*, **Computers & Structures 331 (2026) 108403**,
whose §2.3.2 leaves the tetrahedral case open. Built on [jNO](https://github.com/FhG-IISB/jNO).

The design vector is `[rho, d_x, d_y, d_z]`: one density per tetrahedron plus the position of every
movable node. All sensitivities come from automatic differentiation.

## What carries over to tetrahedra

**The patch filter works on edge fans.** Eq. (18) is a geometric mean over N-2 factors, so its
contrast fades as the patch grows (hinge score relative to solid, measured on jNO's kernel):

| N | 5 | 6 | 8 | 12 | 27 |
|---|---|---|---|---|---|
| hinge / solid | 0.01 | 0.04 | 0.11 | 0.76 | 0.85 |

A 3-D vertex patch has ~27 elements, so it has no usable contrast. An edge fan has ~5.2, where the
filter is at its sharpest. Around an interior edge the fan is a cycle, so the eq. (18) walk is well
defined without an angular sort.

**The other ingredients:**
- eq. (21)/(24) become bounds on the 6 dihedral angles;
- eq. (26)/(28) keep their density-coupled element-volume bounds;
- the eq. (38)-(41) perimeter becomes an area summed over interior facets.

## Results

**Fidelity check.** Each converged, fully binary design (grey fraction 0.0000) is extracted at
rho > 0.5 and re-solved on uniformly refined meshes of the same body
(`scripts/reanalysis_boundary.py`). Linear tets only soften under refinement, so each refined value
is a lower bound on the true compliance. Settings are identical for both formulations; only the
element formulation differs.

| problem | formulation | reported | refined re-solve | reported understates by |
|---|---|---|---|---|
| rope bracket, h = 0.25, 750 it. | conventional | 4.181 | ≥ 111.6 (level 3) | ≥ 2570 % |
| | **E-FEM** | 5.415 | ≥ 6.971 (level 3) | **≥ 28.7 %** |
| L-bracket (3-D), h = 0.25, 250 it. | conventional | 214.4 | ≥ 34121 (level 2) | ≥ 15812 % |
| | **E-FEM** | 328.0 | ≥ 866.3 (level 2) | **≥ 164 %** |

- **E-FEM is ~90× more faithful than the conventional formulation on both geometries** (90× and
  96×).
- Most of the conventional gap is the material model, not the mesh. The step from SIMP field to
  extracted solid alone costs +578 % (conventional) against +14.4 % (E-FEM).
- On one body, the E-FEM rope-bracket design, E-FEM's own solve is off by −22.3 % and a plain P1
  solve by −33.3 %.
- Distortion is not the mechanism. An equal-element-count control with undistorted tets moves the
  conventional result +5.0 % and E-FEM −0.05 %.

**Connectivity** (rope bracket, h = 0.25, rho > 0.5). On a fixed mesh the conventional design reaches
one bolt through a single shared vertex.

| | elements | pieces | largest piece |
|---|---|---|---|
| fixed mesh, conventional | 4,082 | 25 | 62.3 % |
| fixed mesh, E-FEM | 4,082 | 20 | 61.2 % |
| fixed mesh h = 0.10, conventional | 52,967 | 161 | 97.1 % |
| deformable, conventional | 4,082 | 12 | 97.0 % |
| deformable, **E-FEM** | 4,082 | **1** | **100 %** |

**2-D reproduction of the paper's benchmarks** (`scripts/paper2d.py`, cantilever and L-bracket):
- The E-FEM dof count matches the paper's structure exactly (2,204 × 2 × 3 against their
  2,145 × 2 × 3).
- With one load fitted on the cantilever (+1.6 %), the L-bracket is predicted to −11.6 %.
- The conventional arm swings 24–50 % over its last 200 iterations; E-FEM swings 3 % under the same
  settings.

## Limitations and open questions

- **All refined values are one-sided bounds that are still rising.** At the last refinement step
  they still increase +33 % (conventional) and +7 % (E-FEM). Quote the bounds, not extrapolations.
- **The L-bracket pair ran 250 iterations** and its compliances still move 8–15 %. It needs 750 for
  both arms.
- **The fixed-mesh h = 0.10 comparison** is only a level-1 bound (≥ 308 %); level 2 needs more than
  62 GB.
- **The perimeter barrier is required in 3-D.** Without it, the L-bracket load path narrows to one
  tet. `PSTAR` is an area, so it is calibrated per mesh with a `PSTAR=0` probe.
- **Several inputs to the 2-D benchmarks are not stated in the paper:**
  - the applied load: compliance scales with its square, so absolute agreement is indeterminate;
  - the perimeter target P* (we use 650, from jNO's cantilever tutorial);
  - the M_nd convention: our 2-D designs stay ~5 % grey at a one-cell interface layer, where theirs
    are binary.

## Layout

```
scripts/
  bracket.py              rope bracket (main 3-D experiment)
  lbracket3d.py           the paper's L-bracket, extruded, out-of-plane load
  paper2d.py              the paper's 2-D benchmarks
  reanalysis_boundary.py  independent re-solve of the extracted design; protocol in its docstring
  cluster_report.py       checks a run for silent failure modes; run it before trusting a result
  pardiso_solver.py       MKL PARDISO via pure_callback with a hand-written adjoint
  export_console.py       interactive 3-D viewer -> runs/console_<TAG>.html
  fig_*.py, render.py     figures
  tests/                  pytest, fast, no GPU
cluster/                  SLURM jobs
vendor/jNO/               pinned jNO submodule (branch feat/topopt-3d, not yet on main)
runs/                     output, not tracked
```

## Run

```bash
git clone --recurse-submodules git@github.com:armbrusl/topopt3d.git && cd topopt3d
pixi install --manifest-path vendor/jNO/pyproject.toml -e fem
alias jrun="pixi run --manifest-path vendor/jNO/pyproject.toml -e fem python"

HMESH=0.25 ITERS=750 SPACE=cover    jrun scripts/bracket.py   # rope bracket, E-FEM (PSTAR=0)
HMESH=0.25 ITERS=750 SPACE=Lagrange jrun scripts/bracket.py   # rope bracket, conventional
HMESH=0.25 RELOCATE=0 jrun scripts/bracket.py                 # fixed-mesh SIMP baseline
PSTAR=26 SPACE=cover jrun scripts/lbracket3d.py               # L-bracket; PSTAR from a PSTAR=0 probe
```

All settings are environment variables; each run prints its resolved `CONFIG` line at startup.
`runs/` is not in git. The three `.npz` files behind the tables above (~10 MB) are enough to
reproduce every reanalysis and figure without re-optimising; ask for them separately.
`HMESH=0.10 SPACE=cover` needs more than 62 GB to JIT-compile and was run on a cluster
(2× A100 80 GB).
