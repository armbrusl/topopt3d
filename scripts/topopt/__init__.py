"""Shared machinery for the deformable-mesh topology-optimisation problems.

Each module here exists because a copy of it diverged and caused a bug, and the docstrings say
which. The split from the problem descriptors in ``scripts/problems/`` is deliberate: this package
knows about meshes, budgets and design variables, and nothing about brackets.
"""
