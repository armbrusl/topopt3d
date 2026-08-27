# topopt3d — 3-D topology optimisation on a deformable tetrahedral mesh

The 3-D extension of Jung, Yun & Kim, *Density-based topology optimization on a deformable
unstructured mesh with enriched finite elements*, **Computers & Structures 331 (2026) 108403** —
whose §2.3.2 leaves the tetrahedral case open. Built on [jNO](https://github.com/FhG-IISB/jNO).

The design vector is `[rho, d_x, d_y, d_z]`: one density per tetrahedron plus the position of every
interior node. Every sensitivity comes from AD; nothing is hand-derived.

## Where things are

```
scripts/
  bracket.py              the main experiment: tapered bracket, 2 bolt holes, oblique pull.
                          Everything quoted in this README comes from here.
  torsion.py  heattree.py  minimal.py  run.py     the other four problems.
  reanalysis_boundary.py  the independent check -- extracts the design's own boundary as a
                          SOLID and re-solves it on refined meshes. Read its docstring first;
                          it is where the measurement protocol lives.
  reanalysis3d.py         the older density-resampling variant. Superseded by the above.
  pardiso_solver.py       MKL PARDISO through one pure_callback with a hand-written adjoint.
  export_console.py       -> runs/console_<TAG>.html, the interactive 3-D viewer.
  fig_*.py                the figures.
  cluster_report.py       reads a run's npz + logs and checks four SILENT failure modes.
                          Run it on anything before believing it.
  topopt/                 shared layer -- WRITTEN AND TESTED BUT NOT YET WIRED IN.
                          See the STATUS block in its __init__.py before building on it.
  problems/               empty placeholder. See its README.md.
  tests/                  pytest; fast, no GPU, no data files needed.
cluster/                  SLURM jobs and a calibration script.
vendor/jNO/               the pinned jNO submodule (branch feat/topopt-3d).
runs/                     ALL OUTPUT. Not tracked -- see below.
```

**`runs/` is deliberately not in git** (`.gitignore`), so a fresh clone contains **no results**: no
`.npz`, no figures, no consoles. Nothing in this README can be re-checked without first re-running
the optimisation that produced it, which for the fine cases is hours on a cluster. If you have been
handed this repo to continue the work, ask for the `.npz` files as a separate transfer — the binary
pair (`topo3d_hv_Lagrange.npz`, `topo3d_hv_cover.npz`, `topo3d_h10b_Lagrange.npz`) is ~10 MB and is
enough to reproduce every reanalysis, figure and console here without re-optimising anything.

**Start by reading**, in this order: this README's retraction sections (they explain why several
obvious-looking numbers are wrong), then `scripts/reanalysis_boundary.py`'s module docstring, then
`scripts/bracket.py` top to bottom. The docstrings carry the reasoning; the code alone will not tell
you why a bound is chosen.

## Retraction: every 3-D result before commit `477bccd` is void

The relocation predicate was a bounding box over the interior nodes, so it also promoted **175
boundary nodes** — the bolt hole walls and the outer taper. Because a Dirichlet tag binds node IDS
once, at setup, the supports kept clamping nodes that had slid off the hole wall: **146 clamped
nodes at setup, 106 still on the wall at convergence.** Nothing failed. The optimiser simply stopped
reporting the compliance of the mesh it was saving — measured **4.6412 against 5.2876** from an
independent P1 solve of the same mesh and density, a **+13.9 %** error that grew every iteration.

Concretely, these previously published numbers are withdrawn:

| claim | where | status |
|---|---|---|
| reanalysis "+29.0 % attributable" | `477bccd` message | void — computed against a drifting support |
| "E-FEM softer at every iteration, ratio 1.36" | `477bccd` message | void — same cause |
| "the aspect constraint now ends at 0.47" | `477bccd` message | void — same cause |
| every `.npz`, console and figure in `runs/` | — | regenerate; all predate the fix |

The direction of the E-FEM conclusion survived re-measurement and, on a fully binary design (see
below), got stronger still — though the specific pre-projection numbers this paragraph originally
pointed to are themselves superseded. No number from before the support-drift fix should be quoted.
The acceptance test that guards this is in *Honesty checks built in*: a Lagrange run must reproduce
its own reported compliance under an independent solve to **0.00 %** at every frame.

## What the reanalysis measures

`scripts/reanalysis_boundary.py` re-solves the design's own extracted solid on uniformly refined
meshes of the identical geometry. Midpoint subdivision is exact on planar facets, so every level is
the same body; linear tets are stiff and refinement can only soften them, so each level is a
one-sided LOWER bound and no converged value is needed to draw a conclusion.

**A second retraction, found while pursuing the first.** Every result below the fold measured before
20 Aug used `BETA_H > 0` and none of it should be trusted either: the projection's sharpness `beta`
was a Python float captured in a `rho.constrain(...)` closure, and since jNO's step-program cache
started reusing compiled steps across `solve()` calls, mutating that closure to ramp `beta` changed
nothing the cache could see. The gradient stayed at the STARTING sharpness for the whole run while
`crux.eval` recompiled and reported the ramped one — a rising `beta_H` and a falling grey-level
indicator in the log, describing a design optimised at beta = 1 throughout. Fixed in jNO
(`ModelCall.project` — beta as a traced parameter node — plus a step-key fix so a mutated
reparameterisation constant is actually seen); `bracket.py` now uses it via
`jno.optimizers.heaviside_continuation`.

With that fixed, the h = 0.25 pair converges genuinely binary — `M_nd` 0.0001 / 0.0000, and the grey
fraction (density strictly between 0.1 and 0.9) is **exactly 0.0000** for both, which was never true
of any earlier result. That matters because it closes the obvious objection to the number below: it
cannot be an artifact of the Lagrange design being grey, because it is not grey at all.

Reanalysed at level 3 (~500k tets), same mesh, load, supports, material budget and iteration count,
the element formulation the only difference:

| | reported | refined solve (level 3, ~430-480k tets) | understates by |
|---|---|---|---|
| conventional (Lagrange) | 4.1814 | **≥ 111.6403** | **≥ 2570 %** |
| **E-FEM** (`SPACE=cover`) | 5.4154 | **≥ 6.9713** | **≥ 28.7 %** |

**A third retraction, and the reason this table moved.** The numbers here previously read ≥ 567 %
and ≥ 14.4 %. They were measured with an extraction that flattered both designs, and flattered the
conventional one 2.6x harder. `reanalysis_boundary.py` chose its iso-level by bisecting on retained
VOLUME until the extracted body carried the design's own material. That is correct on a grey design.
On a BINARY one it is degenerate: every level in (0,1) selects the same tets, so the only way the
search can meet the volume target is to drive the level toward zero and admit VOID tets — which the
reanalysis then re-solves as fully solid. The void wisps a SIMP design leaves behind sit exactly
where its members almost touch, so they became load-bearing bridges the design does not contain.
Measured on the h = 0.10 binary Lagrange design, the bisection settled at rho > 0.000016 on a body
9.6 % void by volume; extracting the same design honestly at rho > 0.5 made it 3.2x more compliant.
Binary designs are now extracted at rho > 0.5 and the material the manifold repair drops is REPORTED
as the fragmentation measure it is (`scripts/tests/test_extraction_level.py`). Grey designs keep the
volume-preserving path, which is right for them.

The cleanest form needs no cross-design comparison at all. Take ONE body — the E-FEM design — and
read it two ways against its own refined solve (≥ 6.9713):

* its own **E-FEM** solve, 5.4154 → **−22.3 %**
* a plain **P1** solve of the identical mesh and density, 4.6485 → **−33.3 %**

**E-FEM's error is 1.49× smaller on the same configuration.** That factor is weaker than the ~2×
this section claimed before the extraction fix, and it is the one number in this README that the fix
made WORSE rather than better — recorded here rather than quietly dropped. The material-model step in
isolation (SIMP field → extracted solid, same mesh, no refinement) is where the separation actually
lives: **+578 %** for Lagrange against **+14.4 %** for E-FEM. For reference their 2-D Fig. 7 is
−0.5 % with E-FEM and +17.6 % without; our conventional case is far worse than theirs even on a
fully binary design.

**Retraction of the retraction's own headline number.** The pre-projection level-3/level-4 pass
(below) reported E-FEM's own solve as −1.2 %, "faithful", and called that number stronger than the
grey one. It was not stronger — it was flattered by greyness in exactly the way the material-model
control now shows: a design with 4–6 % grey volume reads better than the same formulation on a
binary design, for the same reason Lagrange's grey design read far better than it should have. The
correct comparison is always binary-vs-binary, above.

Distortion is still not the mechanism: the equal-element-count control (same geometry, same
resolution, undistorted tets) reads **+4.98 %** for Lagrange and **−0.05 %** for E-FEM — both small
next to the 578 % and 14.4 % the material model alone contributes. Read that control with its own
caveat, which the script prints: the extracted solid is 0.3-0.5 % interior nodes at h = 0.25, so a
refill has almost no alternative tetrahedralisation to find. It says distortion had no room to
operate at this resolution, NOT that distortion is harmless.

**Read the bounds as bounds — both are still climbing.** At level 3 the Lagrange series is moving
+32.55 % between the last two levels, E-FEM +7.28 %, so neither is converged and the true gap is
almost certainly wider than the table shows. The increment ratio is now measured on the binary pair
and it does NOT reproduce the ~0.5 (O(h¹)) of the earlier grey pass: the increments are non-monotone
— 20.45, 33.99, 27.42 for Lagrange and 0.479, 0.704, 0.473 for E-FEM — so the series is not yet in
its asymptotic regime and the level 2→3 ratios (0.807 and 0.672) are the first that could mean
anything. Extrapolating geometrically from them gives ~226 (Lagrange) and ~7.94 (E-FEM), but a ratio
of 0.807 sits close enough to 1 that the Lagrange figure is not trustworthy — a small error in the
ratio moves it enormously. **Quote the bounds, not the extrapolations**, and treat the Lagrange
series as unconverged rather than assigning it a value.

`python scripts/fig_convergence.py` draws the earlier (grey, level-4) convergence series;
`python scripts/fig_grey.py` draws the density histogram that motivated re-running it, and shows
directly why the iso-level slider on a projected console does nothing — there is no grey left for it
to threshold through.

**A warm start does not substitute for running E-FEM throughout.** Loading a converged Lagrange
design into an E-FEM run (`WARMSTART=<npz>`, same mesh, same connectivity, no transfer) and
optimising 250 further iterations at Lagrange's own cost profile recovers most of the gap but not
all of it. Re-measured with the corrected extraction, at level 3:

| | reported | truth ≥ | understates ≥ |
|---|---|---|---|
| conventional, cold | 4.1814 | 111.6403 | 2570 % |
| E-FEM, warm-started from it | 6.9955 | 18.4096 | 163 % |
| E-FEM, from scratch | 5.4154 | 6.9713 | 28.7 % |

The warm start is a 6.1× improvement on the cold conventional design and still lands **2.6× short**
of running E-FEM from the beginning, while keeping Lagrange's perimeter and roughness rather than
converging toward E-FEM's own. Read the middle row with one caveat: it ran 250 iterations against
the other two rows' 750, so some of its remaining gap is unspent budget rather than the warm start
as such. The direction is not in doubt — the formulation has to be right while the topology is still
forming — but the exact 2.6× would need a 750-iteration warm run to pin down.

### Superseded (pre-projection) pass — kept for the convergence-rate measurement only

The 4M-tet convergence series and the O(h¹) rate above were measured before the projection fix, on a
grey pair (`M_nd` 0.06 / 0.04). The compliance numbers from that pass are superseded by the binary
table above and must not be quoted on their own:

| | reported | refined solve (4M tets) | understates by |
|---|---|---|---|
| conventional (Lagrange) | 4.8591 | ≥ 17.0401 | ≥ 251 % (superseded — see above) |
| E-FEM (`SPACE=cover`) | 6.0357 | ≥ 6.1108 | ≥ 1.2 % (superseded — flattered by greyness) |

Two further defects found while re-measuring, both fixed:

* **the design domain could inflate.** With the outline free to move and the material budget written
  as a fraction of the *current* cell volumes, the optimiser pushed the boundary out and was handed
  more material for it: blank 12.72 → 18.05 (**+42 %**), material 3.81 → 5.39, and a compliance of
  3.11 against 4.07 that was bought rather than designed. The budget is now absolute
  (`VOLFRAC × V₀`, printed at startup) and boundary nodes only slide **tangentially**, so the blank
  is invariant to machine precision (12.717462 → 12.717462 with 665 nodes moving up to 0.331).
* **`RELOCATE=0` is not a valid density-only ablation** on this problem — see *Knobs*.

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

### On the cluster

The machine actually used for the fine runs is **`ki2`: 32-core Xeon Gold 6334, 2x NVIDIA A100 80GB
PCIe** — not an H200. `cluster/h200.sbatch` keeps its name for history; read it as "the cluster job".

```bash
git clone --recurse-submodules git@github.com:armbrusl/topopt3d.git && cd topopt3d
pixi install --manifest-path vendor/jNO/pyproject.toml -e fem
sbatch cluster/h200.sbatch                            # SOLVER=cudss, HMESH=0.10, PSTAR=0 (probe)
PSTAR=<0.76*P> BUDGET=3000 TAG=h200 sbatch cluster/h200.sbatch
```

`SOLVER` is read by **`scripts/run.py` only**, which is what `h200.sbatch` launches. It has NO effect
on `scripts/bracket.py`, which hardcodes `pardiso_lu` (see `scripts/bracket.py:497`) — setting
`SOLVER=cudss` for a bracket run does nothing at all. Try **`SOLVER=cudss` first** for `run.py`; if
it is unavailable or misbehaves, `SOLVER=""` uses the default backend, which always works.

**Two measured limits worth knowing before you burn a slot** (24 Aug 2026):

* **The GPU is slower than the CPU at real problem size.** On a small calibration mesh (h=0.35) GPU
  assembly won, 3.6 vs 4.4 s/iter, and `cluster/calibrate.sh` will tell you so. At h=0.10 with
  `SPACE=cover` the same configuration ran **80 s/iter on GPU against 53 s/iter on CPU**. Calibrate
  at the size you intend to run, not at a cheap one — and note `calibrate.sh` uses `CHUNK=$ITERS`, a
  single chunk, so it does not see the per-chunk recompilation a real `CHUNK=50` run pays.
* **`HMESH=0.10 SPACE=cover` will not compile on a 62 GB workstation.** It is OOM-killed (exit 137)
  during "JIT compiling step function", right after iteration 0 — reproduced twice with nothing else
  running and 47 GB free. Steady-state RSS once running is only ~12.8 GB, so sizing headroom from
  what you observe after iteration 0 badly under-predicts the compile-time peak. Run it on the
  cluster. The h=0.10 *Lagrange* run is fine locally: 3.78 s/iter, 750 iterations in 57 min.

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

## Cost, and what a datacenter GPU changes

Measured on a 20-core CPU + RTX 3070 (8 GB), per design iteration. These are **solver** benchmarks
on `run.py`'s path; for what a GPU does to a whole `bracket.py` iteration see the measured limits
under *On the cluster* above, which point the other way at h=0.10.

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

**On a datacenter card the first thing to try is `backend="cudss"`.** It is a different backend, so
it is not subject to the blocker below; the A100's 80 GB of HBM removes the fill-in ceiling that
bounds sparse-direct on an 8 GB card; and this problem is its ideal case (cached symbolic plan,
values-only re-factorisation every iteration). `pixi install -e fem` does not bring it — install
cuDSS on the cluster and pass `SOLVER=cudss` (to `run.py`; `bracket.py` ignores it). **Untested** —
this remains a recommendation, not a measurement.

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
| `RELOCATE` | `1` | `bracket.py`: `0` skips `.trainable()` on the coordinates entirely. Note this problem is **not** solvable density-only as configured — the initial gmsh mesh violates the element-volume bound and only nodal motion can fix it, so MMA spends the run in feasibility restoration and compliance ends at 1.6e3–2.4e3. Relax `V0MAX_FRAC` before reading anything into a pinned run |
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
* `record()` checks that the **clamped support set has not drifted**. A Dirichlet tag is
  materialised once from the *initial* coordinates and clamps those node IDs for the whole run, so a
  support that slides off the bolt hole keeps working and nothing raises — only an independent
  re-solve of the saved mesh ever notices. It bit us: the relocation predicate was a bounding box
  over the interior nodes, which promoted 175 **boundary** nodes, and the clamp went 146 nodes ->
  106 still on the wall. The reported compliance was 4.6412 against 5.2876 from an independent P1
  solve of the same mesh and density — a **13.9 %** gap that grew every iteration and was invisible
  from inside the run. Now the BC-carrying boundary is frozen and the set is invariant, 146 -> 146.
* `scripts/reanalysis_boundary.py` is the acceptance test for that: a Lagrange run must reproduce
  its own reported compliance under an independent solve to **0.00 %** at every frame. It does.

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
