# topopt3d — 3-D topology optimisation on a deformable tetrahedral mesh

The 3-D extension of Jung, Yun & Kim, *Density-based topology optimization on a deformable
unstructured mesh with enriched finite elements*, **Computers & Structures 331 (2026) 108403** —
whose §2.3.2 leaves the tetrahedral case open. Built on [jNO](https://github.com/FhG-IISB/jNO).

The design vector is `[rho, d_x, d_y, d_z]`: one density per tetrahedron plus the position of every
interior node. Every sensitivity comes from AD; nothing is hand-derived.

## Run it

jNO comes with the repo as a **pinned submodule** (`vendor/jNO`, branch `feat/topopt-3d`), so there
is nothing to clone separately and no version to get wrong.

```bash
git clone --recurse-submodules git@github.com:armbrusl/topopt3d.git
cd topopt3d
pixi install --manifest-path vendor/jNO/pyproject.toml -e fem     # once, ~minutes

alias jrun="pixi run --manifest-path vendor/jNO/pyproject.toml -e fem python"

# 1. PROBE — no barrier, to measure the natural perimeter. PSTAR is an AREA in 3-D, so the
#    2-D tutorial's value is meaningless and there is no sensible default.
HMESH=0.17 PSTAR=0  PENALP=0 BUDGET=330  TAG=probe jrun scripts/run.py

# 2. RUN — set PSTAR to ~0.76 x the P the probe settled at (84.5 -> 64 at this mesh).
HMESH=0.17 PSTAR=64 PENALP=0 BUDGET=1680 TAG=final jrun scripts/run.py

# 3. FIGURES + the interactive 3-D console
RUN=topo3d_final.npz jrun scripts/render.py          # stills + evolution GIF
RUN=topo3d_final.npz jrun scripts/export_console.py  # runs/console.html
jrun scripts/show_mesh.py                            # refinement preview, no run needed
```

Already cloned without `--recurse-submodules`? `git submodule update --init`.

!!! warning "Watch the shell spaces, and set `SOLVER`"
    Every knob is an environment variable, so `CUDA_VISIBLE_DEVICES=1HMESH=0.17` (no space) sets
    **neither** — `HMESH` falls back to `0.10`, i.e. 64,816 tets instead of 14,063. The run prints
    its resolved `CONFIG` line at startup; check it against what you asked for.

    And a GPU does **not** speed this up on its own. The linear solve dominates — at 65k tets it is
    ~95 % of an iteration — and the default backend factorises on the host. Measured on an A100
    with `SOLVER` unset: **82 s/iteration at 64,816 tets**, which is what the same problem costs on
    a CPU. Pick a backend, or the accelerator is idle for the part that matters.

Everything lands in `runs/`. Set `OUTDIR=/scratch/...` to send it elsewhere.

### On the H200 cluster

```bash
git clone --recurse-submodules git@github.com:armbrusl/topopt3d.git && cd topopt3d
pixi install --manifest-path vendor/jNO/pyproject.toml -e fem
sbatch cluster/h200.sbatch                            # SOLVER=cudss, HMESH=0.10, PSTAR=0 (probe)
PSTAR=<0.76*P> BUDGET=3000 TAG=h200 sbatch cluster/h200.sbatch
```

Try **`SOLVER=cudss` first** — see *Cost* below for why. If it is unavailable or misbehaves,
`SOLVER=""` uses the default backend, which always works.

## The pinned jNO branch

`vendor/jNO` is pinned to an exact commit on **`feat/topopt-3d`**, which is not merged to jNO `main`
and carries two prerequisites beneath it (`feat/p-adaptivity` -> `feat/fem-functional`). It provides
the tetrahedral half of the method:

| what | why it is needed |
|---|---|
| `domain.patch_filter()` over **edge fans** | eq. (17)-(19) on tets — see *Why edge fans* below |
| `domain.cell_angles()` returning **6 dihedrals** on tets | the 3-D mesh-quality constraint |
| `domain.transfer_cell_field()` on tets | the reanalysis, i.e. the honesty check |
| `domain._interior_facets()` | the perimeter functional, an **area** in 3-D |
| `expr.integrate(fem, solver=...)` | choosing the linear backend at all |

To move to a newer jNO: `cd vendor/jNO && git pull && cd - && git add vendor/jNO && git commit`.

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

## The solver: use `SOLVER=custom`

`scripts/pardiso_solver.py` drives MKL PARDISO through one `pure_callback` with a hand-written
adjoint, and is the default. Measured against jNO's default backend, per design iteration, **with
the patch filter on**:

| tets | default | `SOLVER=custom` | speed-up |
|---|---|---|---|
| 8,802 | 2,116 ms | 933 ms | 2.3x |
| 14,063 | 4,048 ms | 1,247 ms | 3.2x |
| 24,787 | 12,713 ms | 1,991 ms | **6.4x** |
| 64,816 | ~74 s (an A100 measured 82 s) | **4,263 ms** | **~17x** |

Compliance agrees with the default backend to 4-5 digits at every size; the solution matches to
5e-13 and the adjoint to 3.5e-14 against jNO's own (`scripts/`, correctness test). So a 30-minute
budget is now **~400 iterations at 64,816 tets** rather than ~24.

It exists because **jNO's own `lu(backend="pardiso")` aborts the process** — glibc `double free or
corruption` — whenever a P0 density carries `rho.constrain(d.patch_filter())`. That was chased at
length and the cause was NOT found; ruled out were the GPU, MKL threading, `pure_callback`
concurrency, the adjoint path, both cache-eviction sites, `_check_A`, buffer aliasing, and matrix
rebinding. The decisive clue is that **standalone pypardiso solves the very same pathological
operator in 2.5 ms with a 1.2e-12 residual** — so the fault is in that backend's caching and phase
driving, not in MKL, and going around it costs only the symbolic-analysis reuse (~20 %).

Conditioning is what triggers it: with `E = EMIN + rho^3(E0-EMIN)`, `EMIN=1e-4` and `1e-3` both
abort while `1e-2` survives. Note that flooring `rho_bar` is **not** a conditioning lever — at
`rho_bar=1e-3` the SIMP term contributes 1e-9 and EMIN swamps it entirely.

**An iterative solver is not the shortcut it looks like.** SA-AMG + CG with elasticity's
near-nullspace converges (residual 1e-11) but needs 103-145 CG iterations against the 1e3 SIMP
contrast — 244-1250 ms, an order of magnitude worse than the direct factorisation. The contrast
that wrecks the multigrid rate is exactly what SIMP manufactures. `scripts/bench_solvers.py`
reproduces this on dumped matrices.

## Other known blockers (measured, not guessed)

1. **`penal` as a differentiable parameter NaNs with the patch filter**, on the first solve:
   `d/dp(rho^p) = rho^p ln(rho)` and the filter's `(prod)^(1/(N-2))` both blow up where the physical
   density reaches exactly zero. Flooring it at 1e-3 did not clear it. **So SIMP continuation is off**
   (`PENALP=0`), which is tolerable — jNO measured that it never fires in the published 2-D run.
2. **Length-scale control is not optional.** Without a perimeter barrier the continuum problem is not
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
| `SOLVER` | `custom` | MKL PARDISO via `scripts/pardiso_solver.py` — **recommended**. Also `""` (jNO default), `host`, `device`. Not `pardiso`: it aborts, see above |
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

## A long fine run on another machine

```bash
bash cluster/bracket_fine.sh                     # h=0.05 (~450k tets), 140 iterations, 10 h ceiling
HMESH=0.07 HOURS=6 bash cluster/bracket_fine.sh  # smaller, if memory binds
```

Two phases, because one setting cannot be carried over from a coarser mesh. The perimeter
functional sums the density jump across every interior facet weighted by its **area** in 3-D, and
its natural value GROWS with refinement -- measured 51 at `h=0.15` and 65 at `h=0.10` on the same
bracket, because a finer mesh resolves finer features. So the script **probes** for the natural
perimeter at the chosen resolution and sets `PSTAR` to a fraction of it, rather than copying a
number that would silently over-constrain.

It checkpoints every chunk, so a wall-clock stop, a queue kill or a crash still leaves a usable
design and its full history in `runs/topo3d_<TAG>.npz`.

**Sizing.** Measured on an i5-13600K over 250 iterations: 16,691 tets took 9.0 min and 52,967 took
25.9 min (2.16 and 6.22 s/iteration). Extrapolating linearly puts `h=0.05` at ~50 s/iteration, about
2 h for 140 -- treat that as a **floor**. A sparse direct factorisation on a 3-D mesh fills in faster
than linearly and the `h=0.10` run already came in 60 % above the linear prediction. Memory binds
before wall clock: at ~285k DOFs the factor is several GB. On a GPU node prefer `SOLVER=cudss`,
which is this problem's ideal case -- the sparsity pattern is identical at every iteration, so the
symbolic plan is computed once.

**140 iterations is enough** on this problem: compliance was flat from about iteration 120 at
`h=0.10`. The Heaviside projection ramps to `BETA_H` at `BETA_AT` (default 0.8) of *the run*, so a
shorter budget still projects fully rather than stopping half-projected.
