# `scripts/problems/` — not written yet

This directory is a placeholder for the per-problem descriptors of the shared layer in
`scripts/topopt/`. **It is empty on purpose and nothing depends on it.**

## What belongs here

One module per problem, each returning the things that are *about the problem* and nothing about how
it is optimised: geometry constants, the blank `jno.Shape`, boundary-condition predicates, the load
vector, the non-design (solid) mask, and the `jno.fem` term list. Planned members: `bracket`,
`torsion`, `lbracket`.

## Why it is worth doing

Not to remove duplication for its own sake. The concrete bug it closes is a **desync between the
optimiser and its own validation**: `scripts/reanalysis_boundary.py` and `scripts/reanalysis3d.py`
hardcode

```python
PULL = np.array([0.30, -0.85, 0.43])
```

while `scripts/bracket.py` reads `PULL` from the environment. Change the load on a run and the
reanalysis keeps validating the *old* problem — silently, with no error and a plausible number. A
shared descriptor imported by both makes that impossible to express.

The same applies to geometry: `LX, LY, LZ, SHOULDER, TOPL, TOPR, BX1, BX2, BY, RB` are currently
transcribed in at least three files.

## Where the rest of the plan lives

See the STATUS block at the top of `scripts/topopt/__init__.py`, and the acceptance test named
there: `bracket.py` must reproduce iteration 0 digit-for-digit after the port.
