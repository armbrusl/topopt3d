"""Shared machinery for the deformable-mesh topology-optimisation problems.

Each module here exists because a copy of it diverged and caused a bug, and the docstrings say
which. The split from the problem descriptors in ``scripts/problems/`` is deliberate: this package
knows about meshes, budgets and design variables, and nothing about brackets.

STATUS -- READ THIS BEFORE BUILDING ON IT
=========================================
**This package is a foundation, not yet the architecture. Nothing imports it.**

``mesh``, ``coords`` and ``constraints`` are written and tested
(``scripts/tests/test_topopt_layer.py``, 13 tests), and ``coords.free_axes`` was checked against
``bracket.py``'s hand-rolled predicate on a real mesh -- 2468/2383/2320 free per axis, zero nodes
classified differently. But the five problem scripts (``bracket``, ``torsion``, ``run``, ``minimal``,
``heattree``) each still carry their own copy of this logic and none of them has been ported.

Two pieces are missing before that port can happen:

* ``scripts/problems/`` -- the per-problem descriptors (geometry, blank, BC predicates, load,
  non-design mask, term list). The directory exists and is empty. These are what let
  ``reanalysis_boundary.py`` share a problem definition with the driver, which is the point: today
  the reanalysis scripts hardcode ``PULL`` while ``bracket.py`` reads it from the environment, so
  changing the load makes the reanalysis silently validate a DIFFERENT problem.
* ``record.py`` and ``driver.py`` -- the npz schema and the chunked solve loop.

So: if you are changing how a problem is set up **today**, edit the problem script, not this package
-- editing here changes nothing that runs. If you are continuing the refactor, the acceptance test is
that ``bracket.py`` reproduces iteration 0 digit-for-digit after being ported.
"""
