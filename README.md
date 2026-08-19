# topopt3d — 3-D topology optimisation on a deformable tetrahedral mesh

The 3-D extension of Jung, Yun & Kim, *Density-based topology optimization on a deformable
unstructured mesh with enriched finite elements*, **Computers & Structures 331 (2026) 108403** —
whose §2.3.2 leaves the tetrahedral case open. Built on [jNO](https://github.com/FhG-IISB/jNO).

The design vector is `[rho, d_x, d_y, d_z]`: one density per tetrahedron plus the position of every
interior node. Every sensitivity comes from AD; nothing is hand-derived.

```bash
HMESH=0.17 PSTAR=64 BUDGET=1680 python scripts/run.py     # ~350 iterations at 14k tets
python scripts/render.py                                   # stills + evolution GIF
python scripts/export_console.py                           # interactive HTML console
```

## Requires jNO from an unmerged branch

The tetrahedral half of this does not exist on jNO `main`. It needs branch **`feat/topopt-3d`**,
which adds:

| what | why it is needed |
|---|---|
| `domain.patch_filter()` over **edge fans** | eq. (17)-(19) on tets — see *Why edges* below |
| `domain.cell_angles()` returning **6 dihedrals** on tets | the 3-D mesh-quality constraint |
| `domain.transfer_cell_field()` on tets | the reanalysis, i.e. the honesty check |
| `domain._interior_facets()` | the perimeter functional, an **area** in 3-D |
| `expr.integrate(fem, solver=...)` | choosing the linear backend at all |

```bash
git clone -b feat/topopt-3d <jNO remote> jNO && cd jNO && pixi install -e fem
```

## Why edge fans and not vertex patches

eq. (18) is a **geometric mean over N-2 factors**, so its contrast dies as the patch grows. Measured
on jNO's own kernel (`tests/test_patch_filter_scaling.py`), hinge score relative to solid:

| N | 5 | 6 | 8 | 12 | 27 |
|---|---|---|---|---|---|
| hinge / solid | 0.01 | **0.04** | 0.11 | 0.76 | **0.85** |

A 3-D **vertex** patch is `4T/V ≈ 27` elements — no usable contrast, so eq. (18) cannot be ported to
it. A 3-D **edge fan** is `6T/E ≈ 5.2`, where the formula is at its *sharpest* — better than the 2-D
vertex patch it was designed for. Around an interior edge a tet has exactly two faces containing
both endpoints, so the fan is a **cycle** and eq. (18)'s walk is well defined with no angular sort.

## Cost, and what an H200 changes

Measured on a 20-core CPU + RTX 3070 (8 GB), per design iteration:

| tets | default backend | `solver=lu(backend="pardiso")` |
|---|---|---|
| 2,847 | 620 ms | 335 ms |
| 8,802 | 2,115 ms | 665 ms |
| 14,063 | 4,014 ms | 810 ms |
| 64,816 | ~74 s (extrapolated) | 2,772 ms |

The default's SuperLU is `O(n^1.9)` on this operator against PARDISO's `O(n^1.55)`; the raw solve
differs by **46-125x** and widens with size. Two reasons: the stiffness is SPD (symmetric to
5.6e-17) while the default applies a general LU, and **the sparsity never changes** — relocation
moves nodes, not connectivity — so the symbolic analysis is valid for all 400 iterations.

**On an H200 the first thing to try is `backend="cudss"`.** It is a different backend, so it is not
subject to the blocker below; 141 GB of HBM removes the fill-in ceiling that bounds sparse-direct on
an 8 GB card; and this problem is its ideal case (cached symbolic plan, values-only re-factorisation
every iteration). `pixi install -e fem` does not bring it — install cuDSS on the cluster and pass
`SOLVER=cudss`.

**An iterative solver is not the shortcut it looks like.** SA-AMG + CG with elasticity's
near-nullspace converges (residual 1e-11) but needs 103-145 CG iterations against the 1e3 SIMP
contrast — 244-1250 ms, an order of magnitude worse than the direct factorisation. The contrast that
wrecks the multigrid rate is exactly what SIMP manufactures. `scripts/bench_solvers.py` reproduces
this on dumped matrices.

## Known blockers (measured, not guessed)

1. **`backend="pardiso"` aborts with the patch filter.** glibc `double free or corruption`, at every
   mesh size. Controlled A/B with no CUDA device visible: pardiso alone works, the filter alone works
   under the default backend, the *pair* dies. Ruled out: the GPU, MKL threading (`MKL_NUM_THREADS=1`),
   `pure_callback` concurrency (a lock did not help and cost 15-20%), the adjoint/transpose path, and
   both cache-eviction sites (instrumented; neither fires). Cause still open. **So the speed-up table
   above is not currently reachable on a design run** — it was measured without the filter.
2. **`penal` as a differentiable parameter NaNs with the patch filter**, on the first solve:
   `d/dp(rho^p) = rho^p ln(rho)` and the filter's `(prod)^(1/(N-2))` both blow up where the physical
   density reaches exactly zero. Flooring it at 1e-3 did not clear it. **So SIMP continuation is off**
   (`PENALP=0`), which is tolerable — jNO measured that it never fires in the published 2-D run.
3. **Length-scale control is not optional.** Without a perimeter barrier the continuum problem is not
   well posed and the design fragments at *any* resolution — a finer mesh gives finer speckle, not a
   cleaner truss. Set `PSTAR`; there is no sensible default because in 3-D it is an **area**.

## Calibrating PSTAR

`PSTAR=0` runs unconstrained so the natural perimeter can be measured, then set the target from it —
the 2-D tutorial's 650 is a *length* and means nothing here. Measured at 14k tets: P settles at
**84.5**, so `PSTAR=64` (the same 0.76 ratio the 2-D tutorial uses).

## Knobs

All via environment, so a cluster job needs no edits:

| var | default | meaning |
|---|---|---|
| `HMESH` | `0.10` | target edge length; 0.17 -> 14k tets, 0.10 -> 65k |
| `PSTAR` | `0` | perimeter (**area**) target; 0 disables the barrier |
| `BUDGET` | `1680` | seconds of design loop — the run is wall-clock budgeted, not iteration-counted |
| `CHUNK` | `10` | iterations between snapshots |
| `SOLVER` | `pardiso` | `""` (default backend), `host`, `pardiso`, `cudss` |
| `MOVING` | `1` | `0` freezes the mesh (density-only) |
| `PENALP` | `1` | `0` uses a literal SIMP exponent — **required** while blocker 2 stands |
| `PMASK` | `1` | solid non-designable material under the load |

## Honesty checks built in

* The traction band asserts a **non-zero load resultant**. A band narrower than the mesh spacing
  contains no whole facet and assembles exactly zero — silently. Measured: `z<0.2` gives
  `|f| = 0.000000` on an h=0.4 mesh.
* `(rho * cellv).sum` needs `rho.reshape(-1)` — a P0 parameter is `(n_cells,1)` and `cell_volume()`
  is `(n_cells,)`, so the unflattened product is an `(n_cells,n_cells)` **outer product**,
  `n_cells` times too large, and nothing raises.
* The renderer reports **line pinches** — surface edges carrying more than one sheet — straight off
  the design. That is the diagnostic that says whether 3-D pinching happens often enough to justify
  a vertex criterion at all.
